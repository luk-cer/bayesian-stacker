"""
ground_truth_test.py
====================
End-to-end pipeline validation against synthetic ground truth.

Runs the complete Bayesian Astro Stacker pipeline on synthetic data where
all true values (PSF, shifts, transparency, scene) are known exactly, then
measures the accuracy of each pipeline stage.

Pipeline stages tested
----------------------
  Phase 0  InstrumentModel calibration accuracy
           → bias MAE, read noise MAE, dark rate MAE, flat gain MAE

  Phase 2  FrameCharacterizer per-frame accuracy
           → FWHM recovery error  (recovered vs true seeing)
           → Transparency recovery error
           → Shift recovery error  (dx, dy in LR pixels)

  Phase 3  SufficientStats fast stack quality
           → fast_stack vs true LR scene:  RMSE, PSNR, Pearson r

  Phase 4  MAP super-resolution quality   (if PyTorch available)
           → lambda_hr vs true HR scene:  RMSE, PSNR, Pearson r
           → SR frequency test: PSD above native Nyquist

All results are collected into a ValidationReport with a structured summary
and optional FITS/PNG output.

Usage
-----
    # Quick smoke test (small image, few frames)
    from ground_truth_test import run_ground_truth_test, TestConfig
    report = run_ground_truth_test(TestConfig.fast())
    print(report.summary())

    # Full test at ASI533 sub-resolution
    report = run_ground_truth_test(TestConfig.full())
    report.save("validation_results/")
    print(report.summary())

    # CLI
    python ground_truth_test.py --fast        # quick smoke test
    python ground_truth_test.py --full        # thorough test
    python ground_truth_test.py --output-dir results/

Dependencies
------------
    numpy  scipy  astropy  matplotlib (optional)
    synthetic_starfield  frame_characterizer  sufficient_statistics
    map_stacker  optics  instrument_model_artifact
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
try:
    from synthetic_starfield import StarfieldConfig, StarfieldGroundTruth, generate_starfield
    _STARFIELD_OK = True
except ImportError:
    _STARFIELD_OK = False
    logger.warning("synthetic_starfield.py not found")

try:
    from frame_characterizer import FrameCharacterizer
    _FC_OK = True
except ImportError:
    _FC_OK = False
    logger.warning("frame_characterizer.py not found")

try:
    from sufficient_statistics import SufficientStatsAccumulator, SufficientStats
    _SS_OK = True
except ImportError:
    _SS_OK = False

try:
    from map_stacker import MapConfig, MapResult, solve, _TORCH_OK
except ImportError:
    _TORCH_OK = False
    MapConfig = None
    solve     = None

try:
    from optics import ScopeGeometry
    _OPTICS_OK = True
except ImportError:
    _OPTICS_OK = False

try:
    from instrument_model_artifact import InstrumentModel
    _MODEL_OK = True
except ImportError:
    _MODEL_OK = False


# ============================================================================
# TestConfig
# ============================================================================

@dataclass
class TestConfig:
    """
    Configuration for the ground truth validation test.

    Attributes
    ----------
    shape : (H, W)
        Native sensor resolution.  Smaller = faster test.
    scale_factor : int
        SR upscaling factor.
    n_stars : int
        Number of stars in the synthetic scene.
    n_frames : int
        Number of light frames to stack.
    exposure_s : float
        Exposure per frame.
    seed : int
        Random seed.
    map_n_iter : int
        MAP optimisation iterations (0 = skip MAP).
    map_alpha_tv : float
        TV regularisation weight for MAP.
    output_dir : Path | None
        Where to write results.  None = auto temp dir.
    save_fits : bool
        Write validation FITS files.
    save_plots : bool
        Write PNG plots (requires matplotlib).
    """
    shape:        Tuple[int, int] = (128, 128)
    scale_factor: int             = 2
    n_stars:      int             = 80
    n_frames:     int             = 15
    exposure_s:   float           = 300.0
    seed:         int             = 42
    map_n_iter:   int             = 150
    map_alpha_tv: float           = 5e-3
    output_dir:   Optional[Path]  = None
    save_fits:    bool            = True
    save_plots:   bool            = True

    @classmethod
    def fast(cls) -> "TestConfig":
        """Quick smoke test: 64×64, 8 frames, no MAP."""
        return cls(
            shape        = (64, 64),
            scale_factor = 2,
            n_stars      = 30,
            n_frames     = 8,
            exposure_s   = 300.0,
            map_n_iter   = 0,    # skip MAP (fast mode)
        )

    @classmethod
    def full(cls) -> "TestConfig":
        """Thorough test: 256×256, 30 frames, MAP included."""
        return cls(
            shape        = (256, 256),
            scale_factor = 2,
            n_stars      = 150,
            n_frames     = 30,
            exposure_s   = 300.0,
            map_n_iter   = 200,
        )


# ============================================================================
# Metric helpers
# ============================================================================

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean squared error between two arrays."""
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64))**2)))


def _psnr(a: np.ndarray, b: np.ndarray, data_range: Optional[float] = None) -> float:
    """
    Peak signal-to-noise ratio [dB].  Higher = better.
    data_range defaults to max(b) - min(b).
    """
    if data_range is None:
        data_range = float(b.max()) - float(b.min())
        if data_range < 1e-9:
            return 0.0
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    if mse < 1e-15:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / mse))


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient between flattened arrays."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _psd_ratio_above_nyquist(
    hr: np.ndarray,
    fast: np.ndarray,
    scale: int,
) -> float:
    """
    Compute the ratio of power spectral density above the native Nyquist
    frequency in the MAP result vs the fast stack (upsampled).

    The native Nyquist is at spatial frequency f_nyq = 0.5 cycles/LR-pixel.
    In HR-pixel units this is f_nyq_hr = 0.5 / scale.

    ratio > 1 means the MAP result has MORE high-frequency content than the
    fast stack → super-resolution is working.

    Parameters
    ----------
    hr   : [S·H, S·W] HR MAP result
    fast : [H, W]     fast stack (LR)
    scale: SR scale factor
    """
    sH, sW = hr.shape
    H,  W  = fast.shape

    # Upsample fast stack to HR by nearest-neighbour (preserves frequencies)
    fast_hr = np.repeat(np.repeat(fast, scale, axis=0), scale, axis=1)

    # 2D power spectrum
    psd_hr   = np.abs(np.fft.rfft2(hr   - hr.mean()))**2
    psd_fast = np.abs(np.fft.rfft2(fast_hr - fast_hr.mean()))**2

    # Radial frequency grid in HR-pixel units [0, 0.5]
    fy = np.fft.fftfreq(sH).reshape(-1, 1)
    fx = np.fft.rfftfreq(sW).reshape(1, -1)
    f_rad = np.sqrt(fy**2 + fx**2)

    # Native Nyquist in HR pixels = 0.5/scale (LR Nyquist mapped to HR grid)
    f_nyq_hr = 0.5 / scale

    above = f_rad > f_nyq_hr

    power_hr_above   = float(psd_hr[above].mean())   if above.any() else 0.0
    power_fast_above = float(psd_fast[above].mean()) if above.any() else 1.0

    return power_hr_above / max(power_fast_above, 1e-12)


def _fwhm_from_psf(psf: np.ndarray) -> float:
    """
    Estimate FWHM of a PSF kernel from the central row using subpixel
    linear interpolation at the half-maximum crossings.
    Returns FWHM in pixels.
    """
    K        = psf.shape[0]
    half     = K // 2
    row      = psf[half, :].astype(np.float64)
    peak     = row.max()
    if peak < 1e-12:
        return float(K)
    half_max = 0.5 * peak

    # Left crossing: last index below half_max, left of centre
    left_below  = np.where((row[:half+1] < half_max))[0]
    right_above = np.where((row[half:]   < half_max))[0] + half

    if len(left_below) == 0 or len(right_above) == 0:
        # PSF too wide for the kernel — return kernel size as upper bound
        return float(K)

    li = int(left_below[-1])     # last pixel below half-max on left
    ri = int(right_above[0])     # first pixel below half-max on right

    # Subpixel interpolation
    if li + 1 <= half and row[li + 1] - row[li] > 1e-12:
        left_x = li + (half_max - row[li]) / (row[li + 1] - row[li])
    else:
        left_x = float(li)

    if ri > 0 and row[ri - 1] - row[ri] > 1e-12:
        right_x = (ri - 1) + (row[ri - 1] - half_max) / (row[ri - 1] - row[ri])
    else:
        right_x = float(ri)

    return max(right_x - left_x, 1.0)


# ============================================================================
# ValidationReport
# ============================================================================

@dataclass
class ValidationReport:
    """
    Structured results from the ground truth validation.

    Each metric is a dict mapping a descriptive name to a scalar or array.
    Pass / fail thresholds are defined in _THRESHOLDS.
    """

    # Phase 0 calibration
    calib_bias_mae_pct:      float   = float("nan")
    calib_rn_mae_pct:        float   = float("nan")
    calib_dark_mae_pct:      float   = float("nan")
    calib_flat_mae_pct:      float   = float("nan")

    # Phase 2 frame characterisation
    fwhm_error_px:           List[float] = field(default_factory=list)  # recovered - true
    transparency_error:      List[float] = field(default_factory=list)
    shift_dx_error_px:       List[float] = field(default_factory=list)
    shift_dy_error_px:       List[float] = field(default_factory=list)
    n_stars_detected:        List[int]   = field(default_factory=list)

    # Phase 3 fast stack
    fast_rmse:               float   = float("nan")
    fast_psnr_db:            float   = float("nan")
    fast_pearson_r:          float   = float("nan")

    # Phase 4 MAP
    map_rmse:                float   = float("nan")
    map_psnr_db:             float   = float("nan")
    map_pearson_r:           float   = float("nan")
    map_sr_psd_ratio:        float   = float("nan")   # > 1 → SR working
    map_n_iter:              int     = 0
    map_converged:           bool    = False
    map_elapsed_s:           float   = 0.0

    # Metadata
    n_frames:                int     = 0
    n_frames_characterised:  int     = 0
    elapsed_total_s:         float   = 0.0
    torch_available:         bool    = False
    warnings:                List[str] = field(default_factory=list)

    # Threshold definitions for pass/fail
    _THRESHOLDS: dict = field(default_factory=lambda: {
        "calib_bias_mae_pct":  2.0,   # bias MAE < 2%
        "calib_rn_mae_pct":    5.0,   # read noise MAE < 5%
        "calib_dark_mae_pct": 10.0,   # dark rate MAE < 10% (on normal pixels)
        "calib_flat_mae_pct":  3.0,   # flat gain MAE < 3%
        "fwhm_error_median":   0.5,   # median |FWHM error| < 0.5 LR px
        "transparency_error":  0.05,  # median |t error| < 0.05
        "shift_error_median":  0.5,   # median |shift error| < 0.5 LR px
        "fast_pearson_r":      0.90,  # fast stack r > 0.90
        "map_pearson_r":       0.90,  # MAP r > 0.90
        "map_sr_psd_ratio":    1.1,   # MAP has > 10% more HF power than fast stack
    }, repr=False)

    def pass_fail(self) -> Dict[str, bool]:
        """Return dict of {metric_name: passed}."""
        t = self._THRESHOLDS
        results: Dict[str, bool] = {}

        results["calib_bias_mae"]       = self.calib_bias_mae_pct  < t["calib_bias_mae_pct"]
        results["calib_rn_mae"]         = self.calib_rn_mae_pct    < t["calib_rn_mae_pct"]
        results["calib_dark_mae"]       = self.calib_dark_mae_pct  < t["calib_dark_mae_pct"]
        results["calib_flat_mae"]       = self.calib_flat_mae_pct  < t["calib_flat_mae_pct"]

        if self.fwhm_error_px:
            med_fwhm = float(np.median(np.abs(self.fwhm_error_px)))
            results["fwhm_recovery"]    = med_fwhm < t["fwhm_error_median"]
        if self.transparency_error:
            med_t = float(np.median(np.abs(self.transparency_error)))
            results["transparency"]     = med_t < t["transparency_error"]
        if self.shift_dx_error_px:
            dx_err = float(np.median(np.abs(self.shift_dx_error_px)))
            dy_err = float(np.median(np.abs(self.shift_dy_error_px)))
            results["shift_dx"]         = dx_err < t["shift_error_median"]
            results["shift_dy"]         = dy_err < t["shift_error_median"]

        if not np.isnan(self.fast_pearson_r):
            results["fast_stack"]       = self.fast_pearson_r > t["fast_pearson_r"]
        if not np.isnan(self.map_pearson_r):
            results["map_result"]       = self.map_pearson_r  > t["map_pearson_r"]
        if not np.isnan(self.map_sr_psd_ratio):
            results["sr_frequency"]     = self.map_sr_psd_ratio > t["map_sr_psd_ratio"]

        return results

    def n_passed(self) -> Tuple[int, int]:
        """Return (n_passed, n_total) for all applicable tests."""
        pf = self.pass_fail()
        return sum(pf.values()), len(pf)

    def summary(self) -> str:
        pf      = self.pass_fail()
        n_pass, n_total = self.n_passed()

        def _status(key: str) -> str:
            if key not in pf:
                return "n/a"
            return "✓" if pf[key] else "✗"

        def _flt(v: float, fmt: str = ".3f") -> str:
            return "n/a" if np.isnan(v) else format(v, fmt)

        fwhm_err_med = (float(np.median(np.abs(self.fwhm_error_px)))
                        if self.fwhm_error_px else float("nan"))
        t_err_med    = (float(np.median(np.abs(self.transparency_error)))
                        if self.transparency_error else float("nan"))
        dx_err_med   = (float(np.median(np.abs(self.shift_dx_error_px)))
                        if self.shift_dx_error_px else float("nan"))
        dy_err_med   = (float(np.median(np.abs(self.shift_dy_error_px)))
                        if self.shift_dy_error_px else float("nan"))

        lines = [
            "",
            "╔══════════════════════════════════════════════════════╗",
            "║          BAYESIAN ASTRO STACKER — VALIDATION         ║",
            "╠══════════════════════════════════════════════════════╣",
            f"║  Frames characterised : {self.n_frames_characterised:>3} / {self.n_frames:<3}"
            + " " * 23 + "║",
            f"║  PyTorch available    : {'yes' if self.torch_available else 'no'}"
            + " " * 28 + "║",
            f"║  Total elapsed        : {self.elapsed_total_s:.1f} s"
            + " " * max(0, 27 - len(f"{self.elapsed_total_s:.1f}")) + "║",
            "╠══════════════════════════════════════════════════════╣",
            "║  PHASE 0 — Calibration accuracy                      ║",
            f"║    Bias MAE          {_flt(self.calib_bias_mae_pct):>7}%   "
            f"[thresh <2%]   {_status('calib_bias_mae'):>2}   ║",
            f"║    Read noise MAE    {_flt(self.calib_rn_mae_pct):>7}%   "
            f"[thresh <5%]   {_status('calib_rn_mae'):>2}   ║",
            f"║    Dark rate MAE     {_flt(self.calib_dark_mae_pct):>7}%   "
            f"[thresh <10%]  {_status('calib_dark_mae'):>2}   ║",
            f"║    Flat gain MAE     {_flt(self.calib_flat_mae_pct):>7}%   "
            f"[thresh <3%]   {_status('calib_flat_mae'):>2}   ║",
            "╠══════════════════════════════════════════════════════╣",
            "║  PHASE 2 — Frame characterisation accuracy           ║",
            f"║    |ΔFWHM| median    {_flt(fwhm_err_med):>7} px  "
            f"[thresh <0.5]  {_status('fwhm_recovery'):>2}   ║",
            f"║    |Δt| median       {_flt(t_err_med):>7}     "
            f"[thresh <0.05] {_status('transparency'):>2}   ║",
            f"║    |Δdx| median      {_flt(dx_err_med):>7} px  "
            f"[thresh <0.5]  {_status('shift_dx'):>2}   ║",
            f"║    |Δdy| median      {_flt(dy_err_med):>7} px  "
            f"[thresh <0.5]  {_status('shift_dy'):>2}   ║",
            "╠══════════════════════════════════════════════════════╣",
            "║  PHASE 3 — Fast stack quality                        ║",
            f"║    Pearson r         {_flt(self.fast_pearson_r):>7}     "
            f"[thresh >0.90] {_status('fast_stack'):>2}   ║",
            f"║    RMSE              {_flt(self.fast_rmse):>7}           "
            f"                ║",
            f"║    PSNR              {_flt(self.fast_psnr_db):>7} dB              "
            f"      ║",
            "╠══════════════════════════════════════════════════════╣",
            "║  PHASE 4 — MAP super-resolution                      ║",
            f"║    Pearson r         {_flt(self.map_pearson_r):>7}     "
            f"[thresh >0.90] {_status('map_result'):>2}   ║",
            f"║    RMSE              {_flt(self.map_rmse):>7}                     "
            f"║",
            f"║    PSNR              {_flt(self.map_psnr_db):>7} dB              "
            f"      ║",
            f"║    SR PSD ratio      {_flt(self.map_sr_psd_ratio):>7}     "
            f"[thresh >1.10] {_status('sr_frequency'):>2}   ║",
            f"║    Iterations        {self.map_n_iter:>7}     "
            f"converged={self.map_converged}       ║",
            "╠══════════════════════════════════════════════════════╣",
            f"║  RESULT: {n_pass}/{n_total} tests passed"
            + (" " * max(0, 43 - len(f"{n_pass}/{n_total} tests passed"))) + "║",
            "╚══════════════════════════════════════════════════════╝",
        ]
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)

    def save(self, output_dir: str | Path) -> None:
        """Save report as text and optionally plots to output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        report_path = out / "validation_report.txt"
        report_path.write_text(self.summary())
        logger.info("Report saved → %s", report_path)


# ============================================================================
# Phase 0 metric: calibration accuracy
# ============================================================================

def _measure_calibration(
    model:  "InstrumentModel",
    instr:  object,              # TrueInstrument
    report: ValidationReport,
) -> None:
    """Compare fitted InstrumentModel against true instrument parameters."""
    eps = 1e-9

    # Bias
    if model.bias_mean is not None and hasattr(instr, "bias_mean"):
        true_bias = instr.bias_mean.astype(np.float64)
        fit_bias  = model.bias_mean.astype(np.float64)
        mae = float(np.mean(np.abs(fit_bias - true_bias)))
        report.calib_bias_mae_pct = 100.0 * mae / (float(np.mean(np.abs(true_bias))) + eps)

    # Read noise
    if model.read_noise is not None and hasattr(instr, "read_noise"):
        true_rn = instr.read_noise.astype(np.float64)
        fit_rn  = model.read_noise.astype(np.float64)
        mae = float(np.mean(np.abs(fit_rn - true_rn)))
        report.calib_rn_mae_pct = 100.0 * mae / (float(np.mean(true_rn)) + eps)

    # Dark rate (on normal pixels only)
    if model.dark_rate is not None and hasattr(instr, "dark_rate"):
        true_dr  = instr.dark_rate.astype(np.float64)
        fit_dr   = model.dark_rate.astype(np.float64)
        if hasattr(instr, "hot_pixel_mask"):
            normal = ~instr.hot_pixel_mask
        else:
            normal = np.ones(true_dr.shape, dtype=bool)
        mae = float(np.mean(np.abs(fit_dr[normal] - true_dr[normal])))
        report.calib_dark_mae_pct = 100.0 * mae / (float(np.mean(true_dr[normal])) + eps)

    # Flat gain
    if model.flat_gain is not None and hasattr(instr, "flat_gain"):
        true_fg = instr.flat_gain.astype(np.float64)
        fit_fg  = model.flat_gain.astype(np.float64)
        mae = float(np.mean(np.abs(fit_fg - true_fg)))
        report.calib_flat_mae_pct = 100.0 * mae / (float(np.mean(np.abs(true_fg))) + eps)

    logger.info(
        "Phase 0: bias_MAE=%.2f%%  rn_MAE=%.2f%%  dark_MAE=%.2f%%  flat_MAE=%.2f%%",
        report.calib_bias_mae_pct, report.calib_rn_mae_pct,
        report.calib_dark_mae_pct, report.calib_flat_mae_pct,
    )


# ============================================================================
# Phase 2 metric: frame characterisation accuracy
# ============================================================================

def _measure_characterisation(
    truth:   StarfieldGroundTruth,
    metas:   list,                 # List[FrameMetadata]
    report:  ValidationReport,
) -> None:
    """
    Compare FrameCharacterizer outputs against true per-frame values.

    Notes
    -----
    FWHM error is only measured when at least min_stars_for_psf stars were
    used (n_stars_used > 0); frames that fell back to the Gaussian/prior PSF
    are excluded to avoid masking the PSF estimator's true accuracy.

    Shift error is only measured when solve_status == 'wcs', meaning the
    FrameCharacterizer successfully extracted a WCS solution.  Frames with
    solve_status == 'failed' (no plate solution) contribute shift=None and
    are excluded.  Note that the synthetic WCS headers produced by
    _make_wcs_header() are minimal (no distortion, no ASTAP solve) so shift
    accuracy from this path is limited by the WCS model fidelity.
    """
    min_stars_for_psf = 3   # same as FrameCharacterizer default

    for i, meta in enumerate(metas):
        # FWHM error: only count frames where PSF was actually fitted
        true_fwhm = truth.true_fwhm_lr_px[i]
        if meta.n_stars_used >= min_stars_for_psf:
            report.fwhm_error_px.append(meta.fwhm_pixels - true_fwhm)

        # Transparency error (all frames)
        true_t = truth.true_transparency[i]
        report.transparency_error.append(meta.transparency - true_t)

        # Shift error: only frames with a real WCS solution (not reference)
        if i > 0 and meta.shift is not None and                 getattr(meta, "solve_status", "failed") == "wcs":
            true_dx, true_dy = truth.true_shifts[i]
            report.shift_dx_error_px.append(meta.shift.dx_px - true_dx)
            report.shift_dy_error_px.append(meta.shift.dy_px - true_dy)

        report.n_stars_detected.append(meta.n_stars_used)

    fwhm_med  = float(np.median(np.abs(report.fwhm_error_px)))
    t_med     = float(np.median(np.abs(report.transparency_error)))
    sh_med_dx = float(np.median(np.abs(report.shift_dx_error_px))) \
                if report.shift_dx_error_px else float("nan")
    sh_med_dy = float(np.median(np.abs(report.shift_dy_error_px))) \
                if report.shift_dy_error_px else float("nan")

    logger.info(
        "Phase 2: |ΔFWHM|=%.3f px  |Δt|=%.4f  |Δdx|=%.3f px  |Δdy|=%.3f px  "
        "stars: min=%d max=%d",
        fwhm_med, t_med, sh_med_dx, sh_med_dy,
        min(report.n_stars_detected) if report.n_stars_detected else 0,
        max(report.n_stars_detected) if report.n_stars_detected else 0,
    )


# ============================================================================
# Phase 3 metric: fast stack quality
# ============================================================================

def _measure_fast_stack(
    stats:   SufficientStats,
    truth:   StarfieldGroundTruth,
    report:  ValidationReport,
    cfg:     TestConfig,
    out_dir: Path,
) -> None:
    """Compare fast stack (weighted_mean) against true LR scene."""
    fast  = stats.weighted_mean.astype(np.float64)       # [H, W]
    # Scale true scene by exposure to get total ADU (same units as fast stack)
    true_lr = (truth.true_scene_lr.astype(np.float64)
               * cfg.exposure_s * float(np.mean(truth.true_transparency)))

    # Normalise both to zero-mean for fair comparison (sky is subtracted)
    fast_zm  = fast  - fast.mean()
    true_zm  = true_lr - true_lr.mean()

    report.fast_rmse      = _rmse(fast_zm, true_zm)
    report.fast_psnr_db   = _psnr(fast_zm, true_zm)
    report.fast_pearson_r = _pearson_r(fast_zm, true_zm)

    logger.info(
        "Phase 3: fast stack  r=%.4f  RMSE=%.2f  PSNR=%.1f dB",
        report.fast_pearson_r, report.fast_rmse, report.fast_psnr_db,
    )

    if cfg.save_fits:
        fast_path = out_dir / "fast_stack_validation.fits"
        hdr = fits.Header()
        hdr["COMMENT"] = "Phase 3 fast stack (zero-mean)"
        hdr["PEARSONR"] = round(report.fast_pearson_r, 5)
        fits.writeto(str(fast_path), fast_zm.astype(np.float32), hdr, overwrite=True)

        true_path = out_dir / "true_scene_lr.fits"
        fits.writeto(str(true_path), true_zm.astype(np.float32), overwrite=True)


# ============================================================================
# Phase 4 metric: MAP super-resolution quality
# ============================================================================

def _measure_map(
    map_result: MapResult,
    stats:      SufficientStats,
    truth:      StarfieldGroundTruth,
    report:     ValidationReport,
    cfg:        TestConfig,
    out_dir:    Path,
) -> None:
    """Compare MAP lambda_hr against true HR scene."""
    S        = cfg.scale_factor
    lam_hr   = map_result.lambda_hr.astype(np.float64)   # [S·H, S·W]
    true_hr  = (truth.true_scene_hr.astype(np.float64)
                * cfg.exposure_s * float(np.mean(truth.true_transparency)))

    # Zero-mean for fair comparison
    lam_zm  = lam_hr  - lam_hr.mean()
    true_zm = true_hr - true_hr.mean()

    report.map_rmse      = _rmse(lam_zm, true_zm)
    report.map_psnr_db   = _psnr(lam_zm, true_zm)
    report.map_pearson_r = _pearson_r(lam_zm, true_zm)
    report.map_n_iter    = map_result.n_iter
    report.map_converged = map_result.converged
    report.map_elapsed_s = map_result.elapsed_s

    # SR frequency content test
    fast_lr  = stats.weighted_mean.astype(np.float64)
    report.map_sr_psd_ratio = _psd_ratio_above_nyquist(lam_hr, fast_lr, S)

    logger.info(
        "Phase 4: MAP  r=%.4f  RMSE=%.2f  PSNR=%.1f dB  SR_PSD_ratio=%.3f  "
        "iters=%d  converged=%s",
        report.map_pearson_r, report.map_rmse, report.map_psnr_db,
        report.map_sr_psd_ratio, report.map_n_iter, report.map_converged,
    )

    if cfg.save_fits:
        map_path  = out_dir / "map_result_validation.fits"
        hdr = fits.Header()
        hdr["COMMENT"] = "Phase 4 MAP lambda_hr (zero-mean)"
        hdr["PEARSONR"] = round(report.map_pearson_r, 5)
        hdr["PSNR"]    = round(report.map_psnr_db, 2)
        hdr["SRPSD"]   = round(report.map_sr_psd_ratio, 4)
        fits.writeto(str(map_path), lam_zm.astype(np.float32), hdr, overwrite=True)

        true_hr_path = out_dir / "true_scene_hr.fits"
        fits.writeto(str(true_hr_path), true_zm.astype(np.float32), overwrite=True)


# ============================================================================
# Save diagnostic plots
# ============================================================================

def _save_plots(
    stats:      SufficientStats,
    map_result: Optional[MapResult],
    truth:      StarfieldGroundTruth,
    report:     ValidationReport,
    cfg:        TestConfig,
    out_dir:    Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    S = cfg.scale_factor
    true_lr  = truth.true_scene_lr.astype(np.float64)
    fast_lr  = stats.weighted_mean.astype(np.float64)
    fast_lr -= fast_lr.mean()
    true_lr -= true_lr.mean()

    # ── 1. LR comparison ──────────────────────────────────────────────────
    ncols = 4 if map_result is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    vm = float(np.percentile(np.abs(true_lr), 99))

    ax = axes[0]
    ax.imshow(true_lr, cmap="inferno", vmin=-vm, vmax=vm, origin="lower")
    ax.set_title("True LR scene")
    ax.axis("off")

    ax = axes[1]
    ax.imshow(fast_lr, cmap="inferno", vmin=-vm, vmax=vm, origin="lower")
    ax.set_title(f"Fast stack  r={report.fast_pearson_r:.3f}")
    ax.axis("off")

    ax = axes[2]
    resid = fast_lr - true_lr
    vm_r  = float(np.percentile(np.abs(resid), 99))
    im = ax.imshow(resid, cmap="RdBu_r", vmin=-vm_r, vmax=vm_r, origin="lower")
    ax.set_title(f"Residual (fast−true)\nRMSE={report.fast_rmse:.2f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if map_result is not None and ncols == 4:
        true_hr   = truth.true_scene_hr.astype(np.float64)
        true_hr  -= true_hr.mean()
        lam_hr    = map_result.lambda_hr.astype(np.float64)
        lam_hr   -= lam_hr.mean()
        vm_hr     = float(np.percentile(np.abs(true_hr), 99))
        ax = axes[3]
        ax.imshow(lam_hr, cmap="inferno", vmin=-vm_hr, vmax=vm_hr, origin="lower")
        ax.set_title(f"MAP λ_hr ({S}×)  r={report.map_pearson_r:.3f}")
        ax.axis("off")

    fig.suptitle(
        f"Ground Truth Validation  —  {cfg.n_frames} frames  "
        f"{cfg.shape[0]}×{cfg.shape[1]} LR",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(str(out_dir / "validation_comparison.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Per-frame metrics ───────────────────────────────────────────────
    if report.fwhm_error_px:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        ax = axes[0]
        ax.plot(report.fwhm_error_px, "o-", ms=4, color="steelblue")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Frame index"); ax.set_ylabel("FWHM error [LR px]")
        ax.set_title("FWHM recovery error")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(report.transparency_error, "o-", ms=4, color="darkorange")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Frame index"); ax.set_ylabel("Δt")
        ax.set_title("Transparency error")
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        if report.shift_dx_error_px:
            ax.plot(report.shift_dx_error_px, "o-", ms=4,
                    label="Δdx", color="steelblue")
            ax.plot(report.shift_dy_error_px, "s--", ms=4,
                    label="Δdy", color="crimson")
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.legend(fontsize=9)
        ax.set_xlabel("Frame index"); ax.set_ylabel("Shift error [LR px]")
        ax.set_title("Shift recovery error")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(out_dir / "per_frame_metrics.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ── 3. PSD comparison ──────────────────────────────────────────────────
    if map_result is not None:
        from scipy.ndimage import gaussian_filter as gf
        lam_hr   = map_result.lambda_hr.astype(np.float64)
        fast_hr  = np.repeat(np.repeat(stats.weighted_mean, S, 0), S, 1).astype(np.float64)

        psd_map  = np.abs(np.fft.rfft2(lam_hr  - lam_hr.mean())) **2
        psd_fast = np.abs(np.fft.rfft2(fast_hr - fast_hr.mean()))**2

        sH, sW = lam_hr.shape
        fy = np.fft.fftfreq(sH).reshape(-1, 1)
        fx = np.fft.rfftfreq(sW).reshape(1, -1)
        f_rad = np.sqrt(fy**2 + fx**2).ravel()
        f_nyq_hr = 0.5 / S

        bins   = np.linspace(0, 0.5, 60)
        bc     = 0.5 * (bins[:-1] + bins[1:])
        idx    = np.digitize(f_rad, bins) - 1
        valid  = (idx >= 0) & (idx < len(bc))

        p_map  = np.zeros(len(bc))
        p_fast = np.zeros(len(bc))
        for k in range(len(bc)):
            m = valid & (idx == k)
            if m.any():
                p_map[k]  = psd_map.ravel()[m].mean()
                p_fast[k] = psd_fast.ravel()[m].mean()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(bc, p_map  + 1, lw=1.5, label="MAP λ_hr",  color="steelblue")
        ax.semilogy(bc, p_fast + 1, lw=1.5, label="Fast stack (upsampled)",
                    color="darkorange", ls="--")
        ax.axvline(f_nyq_hr, color="red", ls=":", lw=1.2,
                   label=f"Native Nyquist (f={f_nyq_hr:.3f})")
        ax.set_xlabel("Spatial frequency [cycles/HR pixel]")
        ax.set_ylabel("Mean PSD")
        ax.set_title(f"Power spectrum: SR PSD ratio above Nyquist = "
                     f"{report.map_sr_psd_ratio:.3f}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out_dir / "power_spectrum.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

    logger.info("Validation plots saved → %s", out_dir)


# ============================================================================
# Main entry point
# ============================================================================

def run_ground_truth_test(
    test_cfg: Optional[TestConfig] = None,
    seed:     int = 42,
) -> ValidationReport:
    """
    Run the complete ground truth validation pipeline.

    Parameters
    ----------
    test_cfg : TestConfig  (defaults to TestConfig.fast() if None)
    seed     : random seed

    Returns
    -------
    ValidationReport
    """
    if test_cfg is None:
        test_cfg = TestConfig.fast()

    if not _STARFIELD_OK:
        raise ImportError("synthetic_starfield.py required")
    if not _FC_OK:
        raise ImportError("frame_characterizer.py required")
    if not _SS_OK:
        raise ImportError("sufficient_statistics.py required")

    report   = ValidationReport()
    report.torch_available = _TORCH_OK
    t0_total = time.time()

    # ── Resolve output dir ────────────────────────────────────────────────
    _own_tmpdir = None
    if test_cfg.output_dir is None:
        import tempfile as _tf
        _own_tmpdir   = _tf.TemporaryDirectory()
        out_dir       = Path(_own_tmpdir.name)
    else:
        out_dir = Path(test_cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Ground truth validation: shape=%s  n_frames=%d  seed=%d",
                test_cfg.shape, test_cfg.n_frames, seed)
    logger.info("Output → %s", out_dir)
    logger.info("=" * 60)

    # ── Phase 0+1: Generate synthetic data ───────────────────────────────
    sf_cfg = StarfieldConfig(
        shape           = test_cfg.shape,
        scale_factor    = test_cfg.scale_factor,
        n_stars         = test_cfg.n_stars,
        n_frames        = test_cfg.n_frames,
        exposure_s      = test_cfg.exposure_s,
        output_dir      = out_dir / "starfield",
    )
    logger.info("Generating synthetic star field …")
    truth = generate_starfield(sf_cfg, seed=seed)
    report.n_frames = test_cfg.n_frames

    # ── Load fitted model ────────────────────────────────────────────────
    model = InstrumentModel.load(truth.model_path)

    # ── Phase 0 metrics ───────────────────────────────────────────────────
    logger.info("Measuring calibration accuracy …")
    _measure_calibration(model, truth.true_instrument, report)

    # ── Phase 2+3: Frame characterisation + accumulation ──────
    # ScopeGeometry is required by FrameCharacterizer. Build a real one when
    # optics.py is available; otherwise use a duck-typed SimpleNamespace.
    if _OPTICS_OK:
        scope_geometry = ScopeGeometry(
            aperture_mm     = 100.0,
            focal_length_mm = 550.0,
            pixel_size_um   = 3.76,
        )
    else:
        from types import SimpleNamespace
        scope_geometry = SimpleNamespace(
            aperture_mm               = 100.0,
            focal_length_mm           = 550.0,
            pixel_size_um             = 3.76,
            plate_scale_arcsec_per_px = 1.41,
        )
        report.warnings.append(
            "optics.py not importable — WCS shift recovery skipped"
        )

    # Adapt FrameCharacterizer parameters to image size — small test images
    # need smaller stamps and isolation radii so stars can actually be found.
    H_img = test_cfg.shape[0]
    _stamp   = min(25, max(11, H_img // 8) | 1)   # odd, ≤ 25
    _psf_sz  = min(21, max(9,  H_img // 10) | 1)   # odd, ≤ 21
    _sep     = max(8., H_img / 8.)                  # ≥ 8, scales with image
    _snr     = 5.0 if H_img <= 128 else 8.0         # lower SNR for small images

    fc  = FrameCharacterizer(
        scope_geometry    = scope_geometry,
        snr_threshold     = _snr,
        min_stars_for_psf = 3,
        psf_size          = _psf_sz,
        stamp_size        = _stamp,
        min_sep_px        = _sep,
    )
    acc = SufficientStatsAccumulator(frame_shape=test_cfg.shape)

    metas  = []
    n_char = 0
    for i, path in enumerate(truth.light_fits_paths):
        is_ref = (n_char == 0)
        try:
            # Load raw frame, calibrate, characterise, then accumulate
            raw  = fits.getdata(str(path)).astype(np.float32).squeeze()
            cal  = model.calibrate_frame(raw, test_cfg.exposure_s)
            hdr  = fits.getheader(str(path))
            meta = fc.characterize_calibrated(
                cal,
                header       = hdr,
                exposure_s   = test_cfg.exposure_s,
                is_reference = is_ref,
                frame_path   = path,
            )
            acc.add_calibrated(cal, meta)
            metas.append(meta)
            n_char += 1
        except Exception as exc:
            logger.warning("Frame %d characterisation failed: %s", i, exc)
            report.warnings.append(f"Frame {i} failed: {exc}")

    report.n_frames_characterised = n_char
    logger.info("%d/%d frames characterised", n_char, test_cfg.n_frames)

    # ── Phase 2 metrics ───────────────────────────────────────────────────
    logger.info("Measuring frame characterisation accuracy …")
    _measure_characterisation(truth, metas, report)

    # ── Phase 3 metrics ───────────────────────────────────────────────────
    logger.info("Measuring fast stack quality …")
    stats = acc.finalize()
    _measure_fast_stack(stats, truth, report, test_cfg, out_dir)

    # ── Phase 4 (MAP) ─────────────────────────────────────────────────────
    map_result: Optional[MapResult] = None

    if test_cfg.map_n_iter > 0:
        if not _TORCH_OK:
            logger.warning("PyTorch not available — skipping MAP test")
            report.warnings.append("MAP test skipped: PyTorch not installed")
        else:
            logger.info("Running MAP super-resolution …")
            map_cfg = MapConfig(
                scale_factor = test_cfg.scale_factor,
                mode         = "fast",
                n_iter       = test_cfg.map_n_iter,
                alpha_tv     = test_cfg.map_alpha_tv,
                alpha_kl     = 0.0,
                alpha_wav    = 0.0,
            )
            map_result = solve(stats, config=map_cfg)
            _measure_map(map_result, stats, truth, report, test_cfg, out_dir)

    # ── Plots ─────────────────────────────────────────────────────────────
    if test_cfg.save_plots:
        _save_plots(stats, map_result, truth, report, test_cfg, out_dir)

    # ── Final ─────────────────────────────────────────────────────────────
    report.elapsed_total_s = time.time() - t0_total
    truth.cleanup()
    if _own_tmpdir:
        # Copy report to CWD so user can read it
        (Path(".") / "validation_report.txt").write_text(report.summary())

    logger.info("Validation complete in %.1f s", report.elapsed_total_s)
    return report


# ============================================================================
# CLI
# ============================================================================

def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Bayesian Astro Stacker — ground truth validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fast",  action="store_true",
                        help="Quick smoke test (64×64, 8 frames)")
    parser.add_argument("--full",  action="store_true",
                        help="Full test (256×256, 30 frames)")
    parser.add_argument("--shape", nargs=2, type=int, default=None,
                        metavar=("H", "W"))
    parser.add_argument("--n-frames",  type=int,   default=15)
    parser.add_argument("--n-stars",   type=int,   default=80)
    parser.add_argument("--n-iter",    type=int,   default=150,
                        help="MAP iterations (0 = skip)")
    parser.add_argument("--alpha-tv",  type=float, default=5e-3)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level   = logging.DEBUG if args.verbose else logging.INFO,
        format  = "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    if args.fast:
        cfg = TestConfig.fast()
    elif args.full:
        cfg = TestConfig.full()
    else:
        shape = tuple(args.shape) if args.shape else (128, 128)
        cfg   = TestConfig(
            shape        = shape,
            n_frames     = args.n_frames,
            n_stars      = args.n_stars,
            map_n_iter   = args.n_iter,
            map_alpha_tv = args.alpha_tv,
            output_dir   = Path(args.output_dir) if args.output_dir else None,
        )

    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)

    report = run_ground_truth_test(cfg, seed=args.seed)
    print(report.summary())

    n_pass, n_total = report.n_passed()
    import sys
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    _cli()
