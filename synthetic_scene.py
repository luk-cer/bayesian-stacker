"""
synthetic_scene.py
==================
Synthetic sky scene generator and calibration pipeline validator.

Purpose
-------
Generate a realistic synthetic "sky" containing:
  - Radial vignetting gradient  (centre bright, corners ~60% brightness)
  - Smooth sky background with linear top-to-bottom gradient
  - Filled disc at known position and flux  (hard-edge object)
  - Gaussian blob at known position and flux (smooth extended object)

Apply the full forward model via CalibrationFrameGenerator.generate_raw_light_frame()
to produce a raw FITS frame.  Then calibrate with a fitted InstrumentModel and
compare the result to ground truth.

Validation produces three outputs:
  1. Residual map   (calibrated_adu − true_sky_adu) as float32 + FITS file
  2. Per-parameter error table  median|estimated − true| for bias, RN, dark, flat
  3. Noise budget check  empirical residual variance vs theoretical expectation

Noise budget
------------
For a perfectly calibrated frame the residual per pixel is pure noise:

    residual^(p) ≈ shot_noise / g  +  dark_shot / g  +  read_noise / g

    σ²_expected^(p) = sky_adu^(p) / g^(p)           ← signal shot noise
                    + dark_rate^(p)·t / g^(p)²       ← dark shot (after /g)
                    + σ_r^(p)² / g^(p)²              ← read noise (after /g)

We estimate empirical variance from N_repeat independent frames.
The ratio  χ² = σ²_empirical / σ²_expected  should be ≈ 1.0:
  > 1  →  excess noise from calibration error
  < 1  →  impossible with a correct pipeline

Usage
-----
    from synthetic_calibration import sample_instrument_from_priors, CalibrationFrameGenerator
    from synthetic_scene import SceneParams, SyntheticScene, CalibrationValidator

    instr  = sample_instrument_from_priors(priors, shape=(256, 256))
    gen    = CalibrationFrameGenerator(instr)
    scene  = SyntheticScene(SceneParams.default((256, 256)))
    report = CalibrationValidator(model, instr, scene, gen).run(exposure_s=300., n_repeat=50)
    print(report.summary())
    report.save_fits("validation_output/")

    # Or run everything end-to-end:
    from synthetic_scene import run_full_validation
    report = run_full_validation(shape=(256, 256))
    print(report.summary())

Dependencies
------------
    numpy  astropy  synthetic_calibration  instrument_model_artifact
"""

from __future__ import annotations

import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy project imports (avoid hard-wiring relative paths)
# ---------------------------------------------------------------------------
try:
    from synthetic_calibration import TrueInstrument, CalibrationFrameGenerator
    _SYNTH_CAL_AVAILABLE = True
except ImportError:
    _SYNTH_CAL_AVAILABLE = False
    TrueInstrument = None
    CalibrationFrameGenerator = None

try:
    from instrument_model_artifact import InstrumentModel
    _MODEL_AVAILABLE = True
except ImportError:
    try:
        from instrument_model import InstrumentModel
        _MODEL_AVAILABLE = True
    except ImportError:
        _MODEL_AVAILABLE = False
        InstrumentModel = None


# ============================================================================
# SceneParams
# ============================================================================

@dataclass
class SceneParams:
    """
    All parameters that define a synthetic sky scene.

    Scene formula
    -------------
    vignetting^(p)    = cos^4(theta(p))        cos^4 radial falloff
    sky_background^(p)= sky_level × gradient(y) × vignetting^(p)
    disc^(p)          = disc_peak  if dist(p, disc_centre) <= disc_r else 0
    gauss^(p)         = gauss_peak × exp(-r²/2σ²)

    true_sky^(p) = sky_background^(p) + disc^(p) + gauss^(p)

    All values in units of  ADU / second  (equivalent photon rate per pixel).
    The light-frame generator multiplies by exposure_s to get ADU.

    Parameters
    ----------
    shape                  Sensor shape (H, W)
    sky_level_adu          Median sky background [ADU/s]
    sky_gradient_strength  Fractional top-to-bottom change (0.15 = ±7.5%)
    vignetting_edge_frac   Relative brightness at the image corners (0.60)
    disc_cx / disc_cy      Disc centre as fraction of width / height
    disc_r_frac            Disc radius as fraction of min(H, W)
    disc_peak_adu          Peak disc signal above background [ADU/s]
    gauss_cx / gauss_cy    Gaussian centre as fraction of width / height
    gauss_sigma_frac       Gaussian sigma as fraction of min(H, W)
    gauss_peak_adu         Peak Gaussian signal above background [ADU/s]
    """
    shape:                  Tuple[int, int] = (256, 256)
    sky_level_adu:          float = 500.0
    sky_gradient_strength:  float = 0.15
    vignetting_edge_frac:   float = 0.60
    disc_cx:                float = 0.35
    disc_cy:                float = 0.50
    disc_r_frac:            float = 0.08
    disc_peak_adu:          float = 2000.0
    gauss_cx:               float = 0.65
    gauss_cy:               float = 0.50
    gauss_sigma_frac:       float = 0.04
    gauss_peak_adu:         float = 3000.0

    @classmethod
    def default(cls, shape: Tuple[int, int] = (256, 256)) -> "SceneParams":
        return cls(shape=shape)

    @classmethod
    def for_asi533(cls) -> "SceneParams":
        return cls(shape=(3008, 3008), sky_level_adu=800.,
                   disc_peak_adu=5000., gauss_peak_adu=8000.)


# ============================================================================
# SyntheticScene
# ============================================================================

class SyntheticScene:
    """
    Builds the true sky signal map and region masks.

    Attributes
    ----------
    params       SceneParams
    true_sky     [H, W] float64  total sky signal [ADU/s equivalent]
    sky_bg       [H, W] float64  background component only
    disc_map     [H, W] float64  disc signal above background
    gauss_map    [H, W] float64  Gaussian blob above background
    vignetting   [H, W] float64  vignetting map (values 0–1)
    disc_mask    [H, W] bool     pixels inside the disc
    """

    def __init__(self, params: SceneParams) -> None:
        self.params = params
        self._build()

    def _build(self) -> None:
        H, W = self.params.shape
        p    = self.params
        Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
        cx, cy = W / 2.0, H / 2.0

        # ---- Vignetting: cos^4 law -----------------------------------------
        # Scaled so the corner brightness equals vignetting_edge_frac.
        # Corner distance from centre = sqrt(cx^2 + cy^2).
        r_corner = np.sqrt(cx**2 + cy**2)
        # theta_max: angle at corner such that cos^4(theta_max) = edge_frac
        theta_max    = np.arccos(np.clip(p.vignetting_edge_frac ** 0.25, 0.0, 1.0))
        r_norm       = np.sqrt((X - cx)**2 + (Y - cy)**2) / r_corner
        theta        = r_norm * theta_max
        self.vignetting = np.cos(np.minimum(theta, np.pi / 2.0)) ** 4

        # ---- Sky background: level × linear gradient × vignetting ----------
        gradient   = 1.0 + p.sky_gradient_strength * (Y / H - 0.5)
        self.sky_bg = p.sky_level_adu * gradient * self.vignetting

        # ---- Disc -----------------------------------------------------------
        disc_cx_px = p.disc_cx * W
        disc_cy_px = p.disc_cy * H
        disc_r_px  = p.disc_r_frac * min(H, W)
        dist_disc  = np.sqrt((X - disc_cx_px)**2 + (Y - disc_cy_px)**2)
        self.disc_mask = dist_disc <= disc_r_px
        self.disc_map  = np.where(self.disc_mask, p.disc_peak_adu, 0.0)

        # ---- Gaussian blob --------------------------------------------------
        gauss_cx_px    = p.gauss_cx * W
        gauss_cy_px    = p.gauss_cy * H
        gauss_sigma_px = p.gauss_sigma_frac * min(H, W)
        r2             = (X - gauss_cx_px)**2 + (Y - gauss_cy_px)**2
        self.gauss_map = p.gauss_peak_adu * np.exp(-r2 / (2.0 * gauss_sigma_px**2))

        # ---- Composite ------------------------------------------------------
        self.true_sky = self.sky_bg + self.disc_map + self.gauss_map

        disc_r_px_val    = disc_r_px
        gauss_sigma_val  = gauss_sigma_px
        logger.info(
            "SyntheticScene built: shape=%s  bg=[%.0f,%.0f]  "
            "disc_r=%.1fpx  gauss_sigma=%.1fpx  "
            "sky_max=%.0f  disc_flux=%.0f  gauss_flux=%.0f",
            (H, W),
            float(self.sky_bg.min()), float(self.sky_bg.max()),
            disc_r_px_val, gauss_sigma_val,
            float(self.true_sky.max()),
            float(self.disc_map.sum()),
            float(self.gauss_map.sum()),
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return self.params.shape

    def disc_region_mask(self, expand_px: int = 0) -> np.ndarray:
        if expand_px == 0:
            return self.disc_mask.copy()
        H, W = self.shape
        Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
        cx   = self.params.disc_cx * W
        cy   = self.params.disc_cy * H
        r    = self.params.disc_r_frac * min(H, W)
        return np.sqrt((X - cx)**2 + (Y - cy)**2) <= (r + expand_px)

    def gauss_region_mask(self, sigma_mult: float = 2.0) -> np.ndarray:
        H, W = self.shape
        Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
        cx   = self.params.gauss_cx * W
        cy   = self.params.gauss_cy * H
        sig  = self.params.gauss_sigma_frac * min(H, W)
        return np.sqrt((X - cx)**2 + (Y - cy)**2) <= sigma_mult * sig

    def background_mask(self) -> np.ndarray:
        return ~(self.disc_region_mask(expand_px=3) | self.gauss_region_mask(2.0))

    def summary(self) -> str:
        p    = self.params
        H, W = p.shape
        return "\n".join([
            "SyntheticScene",
            "=" * 52,
            f"  Shape                : {H} × {W} px",
            f"  Sky background       : {self.sky_bg.min():.0f}–{self.sky_bg.max():.0f} ADU/s",
            f"  Vignetting range     : {self.vignetting.min():.3f}–{self.vignetting.max():.3f}",
            f"  Sky gradient         : ±{p.sky_gradient_strength*100:.0f}% top-to-bottom",
            f"  Disc centre          : ({p.disc_cx*W:.0f}, {p.disc_cy*H:.0f}) px",
            f"  Disc radius          : {p.disc_r_frac*min(H,W):.1f} px",
            f"  Disc peak above bg   : {p.disc_peak_adu:.0f} ADU/s",
            f"  Disc total flux      : {self.disc_map.sum():.0f} ADU/s·px",
            f"  Gaussian centre      : ({p.gauss_cx*W:.0f}, {p.gauss_cy*H:.0f}) px",
            f"  Gaussian sigma       : {p.gauss_sigma_frac*min(H,W):.1f} px",
            f"  Gaussian peak        : {p.gauss_peak_adu:.0f} ADU/s",
            f"  Gaussian total flux  : {self.gauss_map.sum():.0f} ADU/s·px",
            f"  true_sky range       : {self.true_sky.min():.0f}–{self.true_sky.max():.0f} ADU/s",
        ])


# ============================================================================
# Data classes for report
# ============================================================================

@dataclass
class ParameterError:
    name:        str
    true_median: float
    est_median:  float
    mae:         float    # median absolute error
    mae_frac:    float    # mae / |true_median|
    p95_error:   float    # 95th-percentile absolute error


@dataclass
class RegionStats:
    name:           str
    n_pixels:       int
    mean_residual:  float   # mean(calibrated - true_sky_adu)
    std_residual:   float   # std of residuals
    sigma_expected: float   # sqrt(median(var_expected)) in region
    chi2_ratio:     float   # median(var_emp / var_exp) in region
    bias_frac:      float   # |mean_residual| / median(sky_bg)


@dataclass
class ValidationReport:
    """
    Output of CalibrationValidator.run().

    residual_map        (calibrated − true_sky_adu) for frame 0  [H,W] float32
    variance_map_emp    empirical per-pixel variance across N_repeat frames
    variance_map_exp    theoretical noise budget variance
    chi2_map            variance_emp / variance_exp  (ideal ≈ 1.0)
    param_errors        list[ParameterError] — bias, RN, dark, flat
    region_stats        list[RegionStats]    — disc, gaussian, background
    global_chi2_median  headline number  (median of chi2_map)
    """
    residual_map:        np.ndarray
    variance_map_emp:    np.ndarray
    variance_map_exp:    np.ndarray
    chi2_map:            np.ndarray
    param_errors:        List[ParameterError]
    region_stats:        List[RegionStats]
    exposure_s:          float
    n_repeat:            int
    global_chi2_median:  float

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "  Calibration Validation Report",
            "=" * 62,
            f"  Exposure         : {self.exposure_s:.0f} s",
            f"  Repeat frames    : {self.n_repeat}",
            f"  Global χ² median : {self.global_chi2_median:.4f}  (ideal = 1.00)",
            "",
            "── Parameter recovery ───────────────────────────────────────",
        ]
        fmt = "  {:<17} true={:>9.4f}  est={:>9.4f}  MAE={:>8.4f} ({:5.2f}%)  p95={:.4f}"
        for pe in self.param_errors:
            lines.append(fmt.format(
                pe.name,
                pe.true_median, pe.est_median,
                pe.mae, pe.mae_frac * 100, pe.p95_error,
            ))
        lines += [
            "",
            "── Spatial region residuals ─────────────────────────────────",
        ]
        for rs in self.region_stats:
            ok = "✓" if 0.75 <= rs.chi2_ratio <= 1.35 else "✗"
            lines.append(
                f"  {rs.name:<14} n={rs.n_pixels:>6}  "
                f"mean={rs.mean_residual:+7.3f}  std={rs.std_residual:7.3f}  "
                f"σ_exp={rs.sigma_expected:7.3f}  "
                f"χ²={rs.chi2_ratio:.3f} {ok}"
            )
        lines.append("=" * 62)
        return "\n".join(lines)

    def save_fits(self, output_dir) -> None:
        """
        Write four FITS files to output_dir:
          residual.fits             calibrated − true_sky_adu (one frame)
          variance_empirical.fits   empirical variance across N_repeat frames
          variance_expected.fits    theoretical noise budget
          chi2_map.fits             variance ratio (ideal = 1.0 everywhere)
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        def _w(fname, data, comment):
            hdr = fits.Header()
            hdr["EXPTIME"]  = self.exposure_s
            hdr["NREPEAT"]  = self.n_repeat
            hdr["CHI2MED"]  = round(self.global_chi2_median, 5)
            hdr["COMMENT"]  = comment
            fits.writeto(str(out / fname), data.astype(np.float32),
                         hdr, overwrite=True)
            logger.info("Saved %s", out / fname)

        _w("residual.fits",          self.residual_map,
           "calibrated - true_sky_adu for frame 0; should be zero-mean noise")
        _w("variance_empirical.fits", self.variance_map_emp,
           f"empirical residual variance from {self.n_repeat} frames")
        _w("variance_expected.fits",  self.variance_map_exp,
           "theoretical noise budget: signal_shot + dark_shot + read_noise")
        _w("chi2_map.fits",           self.chi2_map,
           "var_empirical / var_expected; ideal = 1.0 everywhere")


# ============================================================================
# CalibrationValidator
# ============================================================================

class CalibrationValidator:
    """
    Validate a fitted InstrumentModel against known TrueInstrument parameters.

    Parameters
    ----------
    model     : InstrumentModel            the fitted model to test
    instr     : TrueInstrument             ground truth parameters
    scene     : SyntheticScene             sky signal map
    generator : CalibrationFrameGenerator  draws raw frames from instr+scene
    """

    def __init__(self, model, instr, scene: SyntheticScene, generator) -> None:
        self.model     = model
        self.instr     = instr
        self.scene     = scene
        self.generator = generator

    # ---- Parameter recovery -------------------------------------------------

    def _param_errors(self) -> List[ParameterError]:
        errors = []

        def _pe(name, true, est, exclude_mask=None):
            """
            exclude_mask : [H,W] bool — pixels to exclude from MAE
                           (used for dark rate to skip hot pixels, which
                           inflate the fractional error denominator)
            """
            t  = true.astype(np.float64)
            e  = est.astype(np.float64)
            ae = np.abs(e - t)

            if exclude_mask is not None:
                valid = ~exclude_mask
                t_med  = float(np.median(t[valid]))
                e_med  = float(np.median(e[valid]))
                mae    = float(np.median(ae[valid]))
                p95    = float(np.percentile(ae[valid], 95))
            else:
                t_med  = float(np.median(t))
                e_med  = float(np.median(e))
                mae    = float(np.median(ae))
                p95    = float(np.percentile(ae, 95))

            return ParameterError(
                name        = name,
                true_median = t_med,
                est_median  = e_med,
                mae         = mae,
                mae_frac    = mae / max(abs(t_med), 1e-12),
                p95_error   = p95,
            )

        if self.model.bias_mean is not None:
            errors.append(_pe("bias_mean [ADU]",
                              self.instr.bias_mean, self.model.bias_mean))
        if self.model.read_noise is not None:
            errors.append(_pe("read_noise [ADU]",
                              self.instr.read_noise, self.model.read_noise))
        if self.model.dark_rate is not None:
            # Exclude hot pixels: they inflate mae_frac denominator (tiny median)
            # and the dark regression is not expected to be accurate on outliers
            errors.append(_pe("dark_rate [ADU/s]",
                              self.instr.dark_rate, self.model.dark_rate,
                              exclude_mask=self.instr.hot_pixel_mask))
        if self.model.flat_gain is not None:
            # flat_gain_est has median=1.0 by design; true also has median=1.0
            # (normalised in sample_instrument_from_priors).  Compare shapes.
            errors.append(_pe("flat_gain",
                              self.instr.flat_gain, self.model.flat_gain))
        return errors

    # ---- Region statistics --------------------------------------------------

    def _region_stats(
        self,
        residuals_all: np.ndarray,   # [N, H, W] float64
        var_emp:       np.ndarray,   # [H, W] float32
        var_exp:       np.ndarray,   # [H, W] float32
    ) -> List[RegionStats]:
        sky_bg_med = float(np.median(self.scene.sky_bg))
        chi2 = (var_emp / np.maximum(var_exp, 1e-12)).astype(np.float64)

        regions = {
            "disc":       self.scene.disc_region_mask(),
            "gaussian":   self.scene.gauss_region_mask(),
            "background": self.scene.background_mask(),
        }
        stats = []
        for name, mask in regions.items():
            if not mask.any():
                continue
            r_flat = residuals_all[0][mask]   # single frame residuals in region
            stats.append(RegionStats(
                name           = name,
                n_pixels       = int(mask.sum()),
                mean_residual  = float(np.mean(r_flat)),
                std_residual   = float(np.std(r_flat)),
                sigma_expected = float(np.sqrt(np.median(var_exp[mask]))),
                chi2_ratio     = float(np.median(chi2[mask])),
                bias_frac      = abs(float(np.mean(r_flat))) / max(sky_bg_med, 1.0),
            ))
        return stats

    # ---- Main ---------------------------------------------------------------

    def run(
        self,
        exposure_s: float = 300.0,
        n_repeat:   int   = 50,
        rng:        Optional[np.random.Generator] = None,
    ) -> ValidationReport:
        """
        Run the full validation and return a ValidationReport.

        Parameters
        ----------
        exposure_s : float   Light frame exposure time [s]
        n_repeat   : int     Number of independent frames for variance estimate
        rng        : optional RNG for reproducibility
        """
        if rng is not None:
            self.generator.rng = rng

        sky = self.scene.true_sky       # [H, W] float64, units = ADU/s

        logger.info(
            "Validation: exposure=%.0f s  n_repeat=%d  "
            "sky_range=[%.0f, %.0f] ADU/s",
            exposure_s, n_repeat, float(sky.min()), float(sky.max()),
        )

        # Generate + calibrate N_repeat light frames
        calibrated_stack = []
        for i in range(n_repeat):
            raw = self.generator.generate_raw_light_frame(sky, exposure_s)
            cal = self.model.calibrate_frame(
                raw.astype(np.float32), exposure_s
            ).astype(np.float64)
            calibrated_stack.append(cal)
            if (i + 1) % 10 == 0:
                logger.info("  Calibrated %d / %d frames", i + 1, n_repeat)

        stack = np.stack(calibrated_stack, axis=0)   # [N, H, W]

        # Residuals vs true sky in ADU (not ADU/s — multiply by exposure)
        true_sky_adu   = sky * exposure_s            # [H, W]
        residuals_all  = stack - true_sky_adu        # [N, H, W]

        # Empirical variance across N frames
        var_emp = np.var(residuals_all, axis=0, ddof=1).astype(np.float32)

        # Theoretical noise budget using TRUE instrument values
        g  = np.maximum(self.instr.flat_gain.astype(np.float64), 1e-6)
        rn = self.instr.read_noise.astype(np.float64)
        dk = self.instr.dark_rate.astype(np.float64)

        # sky is [ADU/s]; multiply by t to get expected ADU before flat division
        var_exp = (
            (sky * exposure_s) / g          # signal shot: Poisson(g·sky·t)/g → var = sky·t/g
            + dk * exposure_s / g**2        # dark shot after /g
            + rn**2 / g**2                  # read noise after /g
        ).astype(np.float32)

        chi2_map = (var_emp / np.maximum(var_exp, 1e-12)).astype(np.float32)
        chi2_med = float(np.median(chi2_map))

        param_errors = self._param_errors()
        region_stats = self._region_stats(residuals_all, var_emp, var_exp)

        logger.info(
            "Validation done. χ²_median=%.4f  "
            "bias_MAE=%.3f  dark_MAE=%.5f  flat_MAE=%.5f",
            chi2_med,
            next((p.mae for p in param_errors if "bias" in p.name), 0),
            next((p.mae for p in param_errors if "dark" in p.name), 0),
            next((p.mae for p in param_errors if "flat" in p.name), 0),
        )

        return ValidationReport(
            residual_map       = residuals_all[0].astype(np.float32),
            variance_map_emp   = var_emp,
            variance_map_exp   = var_exp,
            chi2_map           = chi2_map,
            param_errors       = param_errors,
            region_stats       = region_stats,
            exposure_s         = exposure_s,
            n_repeat           = n_repeat,
            global_chi2_median = chi2_med,
        )


# ============================================================================
# Convenience runner
# ============================================================================

def run_full_validation(
    shape:          Tuple[int, int] = (256, 256),
    n_bias:         int   = 30,
    n_flat:         int   = 25,
    dark_exposures: Optional[List[float]] = None,
    dark_repeats:   int   = 3,
    light_exposure: float = 300.0,
    n_repeat_light: int   = 50,
    output_dir:     Optional = None,
    bayes_mode:     bool  = True,
    seed:           int   = 0,
) -> ValidationReport:
    """
    Full end-to-end pipeline validation in one call:

      1. Sample true instrument from SensorPriors (cold start)
      2. Generate synthetic calibration frames
      3. Fit InstrumentModel (with or without BayesCalibrationState)
      4. Build a SyntheticScene
      5. Run CalibrationValidator

    Returns ValidationReport.  If output_dir is given, also saves FITS files.
    """
    # Ensure project modules are importable
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    from synthetic_calibration import (
        sample_instrument_from_priors,
        CalibrationFrameGenerator,
    )
    try:
        from instrument_model_artifact import InstrumentModel as _IM
    except ImportError:
        from instrument_model import InstrumentModel as _IM

    if dark_exposures is None:
        dark_exposures = [60., 120., 300., 300., 600.]

    rng = np.random.default_rng(seed)

    # 1. Priors + sampled true instrument
    try:
        from bayes_calibration import SensorPriors, BayesCalibrationState
        priors = SensorPriors.for_asi533_gain100()
        _bayes_ok = True
    except ImportError:
        priors     = None
        _bayes_ok  = False

    instr = sample_instrument_from_priors(priors, shape=shape, rng=rng)
    logger.info("Sampled instrument:\n%s", instr.summary())

    # 2. Generate calibration frames and fit model
    gen = CalibrationFrameGenerator(instr, rng=rng)

    with tempfile.TemporaryDirectory() as tmpdir:
        bias_f  = gen.generate_bias_frames(n_bias)
        dark_f  = gen.generate_dark_frames(dark_exposures, dark_repeats)
        flat_f  = gen.generate_flat_frames(n_flat)
        dflat_f = gen.generate_dark_flat_frames(10)
        gen.write_fits_folder(tmpdir, bias_f, dark_f, flat_f, dflat_f)

        # 3. Fit model
        if bayes_mode and _bayes_ok:
            state = BayesCalibrationState.from_priors(priors, shape=shape)
            model = _IM.fit_all(tmpdir, bayes_state=state)
        else:
            model = _IM.fit_all(tmpdir)

    logger.info("Fitted model:\n%s", model.summary())

    # 4. Build scene
    scene = SyntheticScene(SceneParams.default(shape=shape))
    logger.info("\n%s", scene.summary())

    # 5. Validate
    validator = CalibrationValidator(model, instr, scene, gen)
    report    = validator.run(
        exposure_s = light_exposure,
        n_repeat   = n_repeat_light,
        rng        = rng,
    )

    if output_dir is not None:
        report.save_fits(output_dir)

    return report
