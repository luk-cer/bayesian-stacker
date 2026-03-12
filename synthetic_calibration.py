"""
synthetic_calibration.py
========================
Bayesian sampler and synthetic calibration frame generator.

Purpose
-------
Draw a coherent set of per-pixel instrument parameters from the posterior
distributions stored in a BayesCalibrationState (or from a SensorPriors
cold start), then generate realistic synthetic FITS calibration frames
(bias, dark, flat, dark_flat) that obey those parameters exactly.

Because the true parameter values are returned alongside the frames, this
module is the foundation of the pipeline validation loop:

    sample  →  generate frames  →  fit InstrumentModel  →  compare

Design
------
Sampling is per-pixel and spatially correlated: hot pixels are placed
at specific locations rather than independently per pixel, matching
what a real sensor would show.

Forward model for each frame type
----------------------------------

  Bias frame:
      b^(p) = μ^(p) + ε,   ε ~ N(0, σ_r^(p)²)
      where μ^(p) ~ N(mu_n^(p), tau2_n^(p))

  Dark frame (exposure t):
      d^(p) = μ^(p) + Poisson(λ^(p) · t) + ε,   ε ~ N(0, σ_r^(p)²)
      where λ^(p) ~ Gamma(α_n^(p), β_n^(p))

  Flat frame (sky level S):
      f^(p) = g^(p) · S + Poisson(g^(p) · S) - g^(p)·S + ε,   ε ~ N(0, σ_r^(p)²)
            ≈ g^(p) · S + N(0, g^(p)·S + σ_r^(p)²)
      where g^(p) ~ N(g_n^(p), v_n^(p))
      Note: exact Poisson sampling is used, not the Gaussian approx.

  Dark flat frame (exposure t):
      df^(p) = μ^(p) + Poisson(λ^(p) · t) + A^(p) + ε
      where A^(p) is the amp glow profile (near-zero for ASI533)

All frames are returned as float32 arrays matching real sensor output.
FITS headers include EXPTIME, IMAGETYP, CCD-TEMP, GAIN, BAYERPAT.

Usage
-----
    from bayes_calibration import BayesCalibrationState, SensorPriors
    from synthetic_calibration import sample_instrument, CalibrationFrameGenerator

    # From posteriors (after one or more real sessions)
    state = BayesCalibrationState.load("instrument.h5")
    instr = sample_instrument(state, rng=np.random.default_rng(42))

    # Or from priors (cold start — no real data yet)
    priors = SensorPriors.for_asi533_gain100()
    instr  = sample_instrument_from_priors(priors, shape=(256, 256))

    gen = CalibrationFrameGenerator(instr)
    bias_frames  = gen.generate_bias_frames(n=20)
    dark_frames  = gen.generate_dark_frames(exposures=[120., 300., 600.])
    flat_frames  = gen.generate_flat_frames(n=15, sky_adu=30_000.)
    dflat_frames = gen.generate_dark_flat_frames(n=10, exposure=2.)

    # Save as FITS to a folder (for InstrumentModel.fit_all)
    gen.write_fits_folder(output_dir, bias_frames, dark_frames,
                          flat_frames, dflat_frames)

Dependencies
------------
    numpy  astropy  bayes_calibration  scipy.special (for InvGamma sampling)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from project modules (optional — degrade gracefully)
# ---------------------------------------------------------------------------
try:
    from bayes_calibration import BayesCalibrationState, SensorPriors
    _BAYES_AVAILABLE = True
except ImportError:
    _BAYES_AVAILABLE = False
    BayesCalibrationState = None  # type: ignore
    SensorPriors = None           # type: ignore


# ============================================================================
# TrueInstrument — ground truth parameter maps
# ============================================================================

@dataclass
class TrueInstrument:
    """
    Ground-truth per-pixel instrument parameters drawn from posteriors.

    All arrays are float64, shape (H, W).

    These are the *actual* values used to generate synthetic frames —
    not the posteriors, not the point estimates, but the specific draw.
    The validator compares InstrumentModel.fit_* output against these.

    Attributes
    ----------
    bias_mean       Per-pixel bias pedestal μ^(p) [ADU]
    read_noise      Per-pixel read noise σ_r^(p) [ADU]
    dark_rate       Per-pixel dark current λ^(p) [ADU/s]
    flat_gain       Per-pixel throughput g^(p)  (median ≈ 1.0)
    amp_glow        Amp glow spatial profile [ADU] — near zero for ASI533
    hot_pixel_mask  True locations of hot pixels
    shape           (H, W)
    gain_setting    Camera gain [ADU/e-]
    temperature_c   Sensor temperature [°C]
    bayer_pattern   'RGGB' | None
    """
    bias_mean:       np.ndarray
    read_noise:      np.ndarray
    dark_rate:       np.ndarray
    flat_gain:       np.ndarray
    amp_glow:        np.ndarray
    hot_pixel_mask:  np.ndarray
    shape:           Tuple[int, int]
    gain_setting:    float = 100.0
    temperature_c:   float = -10.0
    bayer_pattern:   Optional[str] = "RGGB"

    def summary(self) -> str:
        lines = ["TrueInstrument (ground truth)", "=" * 42]
        lines.append(f"  Shape          : {self.shape}")
        lines.append(f"  Gain setting   : {self.gain_setting}")
        lines.append(f"  Temperature    : {self.temperature_c} °C")
        lines.append(f"  Bayer pattern  : {self.bayer_pattern}")
        lines.append(f"  bias_mean      : median={np.median(self.bias_mean):.2f}  "
                     f"std={self.bias_mean.std():.3f} ADU")
        lines.append(f"  read_noise     : median={np.median(self.read_noise):.3f} ADU")
        lines.append(f"  dark_rate      : median={np.median(self.dark_rate):.5f} ADU/s  "
                     f"max={self.dark_rate.max():.4f}")
        lines.append(f"  flat_gain      : median={np.median(self.flat_gain):.4f}  "
                     f"range=[{self.flat_gain.min():.3f}, {self.flat_gain.max():.3f}]")
        lines.append(f"  hot pixels     : {self.hot_pixel_mask.sum():,} "
                     f"({100*self.hot_pixel_mask.mean():.3f}%)")
        return "\n".join(lines)


# ============================================================================
# Sampling from posteriors
# ============================================================================

def sample_instrument(
    state: "BayesCalibrationState",
    rng:   Optional[np.random.Generator] = None,
    hot_pixel_rate_threshold_sigma: float = 5.0,
) -> TrueInstrument:
    """
    Draw one realisation of instrument parameters from the posterior.

    Each pixel's parameter is drawn independently from its marginal posterior:

      μ^(p)  ~ N(mu_n^(p), tau2_n^(p))             bias mean
      σ_r^(p) ~ sqrt(InvGamma(alpha_n^(p), beta_n^(p)))  read noise
      λ^(p)  ~ Gamma(alpha_n^(p), 1/beta_n^(p))    dark rate
      g^(p)  ~ N(g_n^(p), v_n^(p))                 flat gain

    Parameters
    ----------
    state : BayesCalibrationState
        Fitted Bayesian state from one or more real (or synthetic) sessions.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    hot_pixel_rate_threshold_sigma : float
        Pixels whose sampled dark_rate exceeds median + k·MAD are flagged
        as hot in the TrueInstrument.

    Returns
    -------
    TrueInstrument with all arrays at float64.
    """
    if rng is None:
        rng = np.random.default_rng()

    if state.shape is None:
        raise ValueError("BayesCalibrationState has no shape — "
                         "was it initialised with from_priors()?")
    H, W = state.shape

    # ---- Bias mean: N(mu_n, tau2_n) ----------------------------------------
    if state.bias_acc is not None:
        mu_n   = state.bias_acc.posterior_mean          # [H, W]
        tau2_n = state.bias_acc.posterior_variance      # [H, W]
        bias_mean = rng.normal(mu_n, np.sqrt(np.maximum(tau2_n, 0.0)))
    else:
        logger.warning("No bias posterior — using flat 300 ADU bias")
        bias_mean = np.full((H, W), 300.0)

    # ---- Read noise: sqrt(InvGamma(alpha_n, beta_n)) -----------------------
    # InvGamma(α,β): sample x ~ Gamma(α, 1/β) → σ² = 1/x
    if state.read_noise_acc is not None:
        alpha_rn = state.read_noise_acc.posterior_alpha   # [H, W]
        beta_rn  = state.read_noise_acc.posterior_beta    # [H, W]
        # Gamma(alpha, scale=1/beta) — numpy uses shape,scale parameterisation
        var_sample = 1.0 / rng.gamma(
            shape=np.maximum(alpha_rn, 1e-6),
            scale=1.0 / np.maximum(beta_rn, 1e-6),
        )
        read_noise = np.sqrt(np.maximum(var_sample, 0.01))  # floor at 0.1 ADU
    else:
        logger.warning("No read noise posterior — using flat 6 ADU")
        read_noise = np.full((H, W), 6.0)

    # ---- Dark rate: Gamma(alpha_n, 1/beta_n) --------------------------------
    if state.dark_acc is not None:
        alpha_dk = state.dark_acc.posterior_alpha   # [H, W]
        beta_dk  = state.dark_acc.posterior_beta    # [H, W]
        dark_rate = rng.gamma(
            shape=np.maximum(alpha_dk, 1e-6),
            scale=1.0 / np.maximum(beta_dk, 1e-6),
        )
        dark_rate = np.maximum(dark_rate, 0.0)
    else:
        logger.warning("No dark posterior — using flat 0.002 ADU/s")
        dark_rate = np.full((H, W), 0.002)

    # ---- Flat gain: N(g_n, v_n) --------------------------------------------
    if state.flat_acc is not None:
        g_n = state.flat_acc.posterior_mean      # [H, W]
        v_n = state.flat_acc.posterior_variance  # [H, W]
        flat_gain = rng.normal(g_n, np.sqrt(np.maximum(v_n, 0.0)))
        flat_gain = np.maximum(flat_gain, 0.05)  # physically must be > 0
    else:
        logger.warning("No flat posterior — using flat gain 1.0")
        flat_gain = np.ones((H, W))

    # ---- Amp glow: near-zero for ASI533 ------------------------------------
    amp_glow = np.zeros((H, W), dtype=np.float64)

    # ---- Hot pixel mask from sampled dark rate ------------------------------
    med = float(np.median(dark_rate))
    mad = float(np.median(np.abs(dark_rate - med)))
    threshold = med + hot_pixel_rate_threshold_sigma * 1.4826 * mad
    hot_pixel_mask = dark_rate > threshold

    logger.info(
        "sample_instrument: bias=%.1f±%.2f ADU  RN=%.2f ADU  "
        "dark=%.4f ADU/s  hot=%d px",
        float(np.median(bias_mean)), float(bias_mean.std()),
        float(np.median(read_noise)),
        float(np.median(dark_rate)),
        int(hot_pixel_mask.sum()),
    )

    return TrueInstrument(
        bias_mean      = bias_mean.astype(np.float64),
        read_noise     = read_noise.astype(np.float64),
        dark_rate      = dark_rate.astype(np.float64),
        flat_gain      = flat_gain.astype(np.float64),
        amp_glow       = amp_glow,
        hot_pixel_mask = hot_pixel_mask,
        shape          = (H, W),
    )


def sample_instrument_from_priors(
    priors: "SensorPriors",
    shape:  Tuple[int, int],
    rng:    Optional[np.random.Generator] = None,
    flat_gain_spatial_std: float = 0.03,
    hot_pixel_fraction:    float = 0.005,
    cold_pixel_fraction:   float = 0.001,
) -> TrueInstrument:
    """
    Draw instrument parameters from sensor spec priors (no session data).

    Used for a cold-start validation where no real calibration data exists.
    Generates spatially structured parameters that mimic a real sensor:

    - Bias: smooth 2-D low-frequency variation + per-pixel noise
    - Read noise: slight spatial gradient (e.g. amp glow corner effect)
    - Dark rate: mostly uniform + exponential tail + planted hot/cold pixels
    - Flat gain: radial vignetting × per-channel Bayer offset × pixel noise

    Parameters
    ----------
    priors : SensorPriors
        Sensor spec priors (mean values and uncertainties).
    shape : (H, W)
        Sensor dimensions.
    flat_gain_spatial_std : float
        Pixel-to-pixel gain standard deviation (default 0.03 = 3%).
    hot_pixel_fraction : float
        Fraction of pixels to plant as hot (default 0.005 = 0.5%).
    cold_pixel_fraction : float
        Fraction of pixels to plant as cold (default 0.001 = 0.1%).
    """
    if rng is None:
        rng = np.random.default_rng()
    H, W = shape

    # ---- Bias: smooth 2-D structure + per-pixel noise ----------------------
    # Real sensors have a spatially correlated bias pattern from amp layout
    Y, X = np.mgrid[0:H, 0:W]
    bias_structure = (priors.bias_mean_adu
                      + 5.0 * np.sin(2 * np.pi * X / W)         # horizontal banding
                      + 3.0 * np.cos(2 * np.pi * Y / H * 0.5))  # slow vertical drift
    bias_mean = bias_structure + rng.normal(0, 2.0, shape)

    # ---- Read noise: slightly higher at corners (amp proximity) ------------
    r_norm     = np.sqrt(((X - W/2)/(W/2))**2 + ((Y - H/2)/(H/2))**2)
    read_noise = (priors.read_noise_adu * (1.0 + 0.05 * r_norm)
                  + rng.normal(0, 0.3, shape))
    read_noise = np.maximum(read_noise, 1.0)

    # ---- Dark rate: log-normal body + planted hot/cold pixels --------------
    # Log-normal gives the right heavy tail for dark current distributions
    sigma_ln   = 0.3   # log-normal shape
    mu_ln      = np.log(priors.dark_rate_adu_per_s) - 0.5 * sigma_ln**2
    dark_rate  = rng.lognormal(mu_ln, sigma_ln, shape)
    dark_rate  = np.maximum(dark_rate, 0.0)

    # Plant hot pixels — clearly separated from normal: 20-50× median rate
    # Keep fraction small enough that MAD is not distorted (~0.3%)
    n_hot = int(H * W * hot_pixel_fraction)
    if n_hot > 0:
        hot_y = rng.integers(0, H, n_hot)
        hot_x = rng.integers(0, W, n_hot)
        dark_rate[hot_y, hot_x] *= rng.uniform(20, 50, n_hot)

    # Plant cold pixels — 0.05-0.2× median rate
    n_cold = int(H * W * cold_pixel_fraction)
    if n_cold > 0:
        cold_y = rng.integers(0, H, n_cold)
        cold_x = rng.integers(0, W, n_cold)
        dark_rate[cold_y, cold_x] *= rng.uniform(0.05, 0.2, n_cold)

    # ---- Flat gain: Bayer channel offsets + pixel-to-pixel QE variation ------
    # Physical convention: flat_gain has median = 1.0.
    # Vignetting is NOT included here — it belongs in the scene (sky_bg already
    # applies a cos^4 vignetting in SyntheticScene).  flat_gain captures only
    # sensor-level pixel-to-pixel effects: per-channel Bayer QE and dust/defects.
    # This matches what fit_flat recovers (fit_flat always normalises so
    # median(flat_gain_est) = 1.0), ensuring calibrate_frame correctly returns sky×t.

    # Bayer per-channel QE offset (R and B slightly less sensitive than G)
    bayer_qe = np.ones(shape)
    bayer_qe[0::2, 0::2] = 0.97   # R — slightly lower QE
    bayer_qe[0::2, 1::2] = 1.00   # G0
    bayer_qe[1::2, 0::2] = 1.00   # G1
    bayer_qe[1::2, 1::2] = 0.96   # B — lowest QE

    # Pixel-to-pixel variation (dust, QE non-uniformity)
    pixel_noise = rng.normal(0, flat_gain_spatial_std, shape)
    flat_gain   = bayer_qe * (1.0 + pixel_noise)
    flat_gain   = np.maximum(flat_gain, 0.05)

    # Normalise to median = 1.0 — matches fit_flat convention exactly
    flat_gain  /= np.median(flat_gain)

    # ---- Amp glow (negligible for ASI533) ----------------------------------
    amp_glow = np.zeros(shape)

    # ---- Hot pixel mask -----------------------------------------------------
    med = float(np.median(dark_rate))
    mad = float(np.median(np.abs(dark_rate - med)))
    hot_pixel_mask = dark_rate > (med + 5.0 * 1.4826 * mad)

    logger.info(
        "sample_instrument_from_priors: shape=%s  bias=%.1f ADU  "
        "RN=%.2f ADU  dark_median=%.5f ADU/s  hot=%d px",
        shape,
        float(np.median(bias_mean)),
        float(np.median(read_noise)),
        float(np.median(dark_rate)),
        int(hot_pixel_mask.sum()),
    )

    return TrueInstrument(
        bias_mean      = bias_mean.astype(np.float64),
        read_noise     = read_noise.astype(np.float64),
        dark_rate      = dark_rate.astype(np.float64),
        flat_gain      = flat_gain.astype(np.float64),
        amp_glow       = amp_glow,
        hot_pixel_mask = hot_pixel_mask,
        shape          = shape,
    )


# ============================================================================
# Calibration frame generator
# ============================================================================

@dataclass
class SyntheticFrame:
    """One synthetic calibration frame with its metadata."""
    data:        np.ndarray    # [H, W] float32
    exposure_s:  float
    frame_type:  str           # 'bias' | 'dark' | 'flat' | 'dark_flat'
    sky_adu:     float = 0.0   # sky level used for flat normalisation (flat only)


class CalibrationFrameGenerator:
    """
    Generates synthetic calibration frames from a TrueInstrument.

    All noise is drawn from the physically correct distributions:
      - Read noise: Gaussian(0, σ_r²) per pixel
      - Dark counts: Poisson(λ·t) per pixel
      - Flat shot noise: Poisson(g·S) per pixel (exact, not Gaussian approx)

    This matches the generative model assumed by the calibration pipeline,
    so a correctly implemented pipeline should recover the true parameters
    within the expected noise bounds.
    """

    def __init__(
        self,
        instrument: TrueInstrument,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.instr = instrument
        self.rng   = rng if rng is not None else np.random.default_rng()

    # ---- Individual frame generators ----------------------------------------

    def _read_noise_frame(self) -> np.ndarray:
        """Draw one read noise realisation [H, W]."""
        return self.rng.normal(0.0, self.instr.read_noise)

    def _bias_frame(self) -> np.ndarray:
        """bias_mean + read_noise."""
        return self.instr.bias_mean + self._read_noise_frame()

    def _dark_frame(self, exposure_s: float) -> np.ndarray:
        """bias + Poisson(dark_rate · t) + read_noise."""
        dark_counts = self.rng.poisson(
            np.maximum(self.instr.dark_rate * exposure_s, 0.0)
        ).astype(np.float64)
        return self.instr.bias_mean + dark_counts + self._read_noise_frame()

    def _flat_frame(self, sky_adu: float) -> np.ndarray:
        """
        g · S + Poisson(g · S) - g·S + read_noise
        = g · S + shot_noise + read_noise

        The expected value is g·S.  We use exact Poisson sampling.
        For very large g·S (>1e7) the Poisson can be approximated by
        N(g·S, g·S) — numpy handles this automatically via normal approx.
        """
        expected   = np.maximum(self.instr.flat_gain * sky_adu, 0.0)
        # numpy clips Poisson lambda at ~1e9 internally and falls back to normal
        shot_noise = (self.rng.poisson(expected).astype(np.float64) - expected)
        return (self.instr.bias_mean
                + expected
                + shot_noise
                + self._read_noise_frame())

    def _dark_flat_frame(self, exposure_s: float) -> np.ndarray:
        """bias + Poisson(dark·t) + amp_glow + read_noise."""
        dark_counts = self.rng.poisson(
            np.maximum(self.instr.dark_rate * exposure_s, 0.0)
        ).astype(np.float64)
        return (self.instr.bias_mean
                + dark_counts
                + self.instr.amp_glow
                + self._read_noise_frame())

    # ---- Batch generators ---------------------------------------------------

    def generate_bias_frames(self, n: int = 20) -> List[SyntheticFrame]:
        """Generate n bias frames."""
        frames = [
            SyntheticFrame(
                data       = self._bias_frame().astype(np.float32),
                exposure_s = 0.0,
                frame_type = "bias",
            )
            for _ in range(n)
        ]
        logger.info("Generated %d bias frames", n)
        return frames

    def generate_dark_frames(
        self,
        exposures: List[float],
        repeats:   int = 1,
    ) -> List[SyntheticFrame]:
        """
        Generate dark frames at each specified exposure time.

        Parameters
        ----------
        exposures : list of float
            Exposure times in seconds.  Pass at least 2 distinct values
            to enable dark rate regression in InstrumentModel.fit_dark().
        repeats : int
            Number of frames per exposure time.
        """
        frames = []
        for exp in exposures:
            for _ in range(repeats):
                frames.append(SyntheticFrame(
                    data       = self._dark_frame(exp).astype(np.float32),
                    exposure_s = exp,
                    frame_type = "dark",
                ))
        logger.info("Generated %d dark frames (%d exposures × %d repeats)",
                    len(frames), len(exposures), repeats)
        return frames

    def generate_flat_frames(
        self,
        n:       int   = 20,
        sky_adu: float = 30_000.0,
    ) -> List[SyntheticFrame]:
        """
        Generate n flat frames at sky_adu counts.

        sky_adu should be high enough that shot noise dominates read noise
        (sky_adu >> σ_r²).  30 000 ADU is a typical well-exposed flat.
        """
        frames = [
            SyntheticFrame(
                data       = self._flat_frame(sky_adu).astype(np.float32),
                exposure_s = 2.0,
                frame_type = "flat",
                sky_adu    = sky_adu,
            )
            for _ in range(n)
        ]
        logger.info("Generated %d flat frames at %.0f ADU", n, sky_adu)
        return frames

    def generate_dark_flat_frames(
        self,
        n:          int   = 10,
        exposure_s: float = 2.0,
    ) -> List[SyntheticFrame]:
        """Generate n dark flat frames at the flat exposure time."""
        frames = [
            SyntheticFrame(
                data       = self._dark_flat_frame(exposure_s).astype(np.float32),
                exposure_s = exposure_s,
                frame_type = "dark_flat",
            )
            for _ in range(n)
        ]
        logger.info("Generated %d dark flat frames at %.1f s", n, exposure_s)
        return frames

    # ---- FITS I/O -----------------------------------------------------------

    def _make_header(self, frame_type: str, exposure_s: float) -> fits.Header:
        """Build a FITS header matching what InstrumentModel expects."""
        type_map = {
            "bias":      "Bias Frame",
            "dark":      "Dark Frame",
            "flat":      "Flat Frame",
            "dark_flat": "Dark Flat Frame",
        }
        hdr = fits.Header()
        hdr["INSTRUME"] = "SyntheticCamera"
        hdr["IMAGETYP"] = type_map.get(frame_type, frame_type)
        hdr["EXPTIME"]  = exposure_s
        hdr["GAIN"]     = self.instr.gain_setting
        hdr["CCD-TEMP"] = self.instr.temperature_c
        if self.instr.bayer_pattern is not None:
            hdr["BAYERPAT"] = self.instr.bayer_pattern
        hdr["SYNTH"]    = True
        hdr["COMMENT"]   = "Generated by synthetic_calibration.py"
        return hdr

    def write_fits_folder(
        self,
        output_dir:   str | Path,
        bias_frames:  List[SyntheticFrame],
        dark_frames:  List[SyntheticFrame],
        flat_frames:  List[SyntheticFrame],
        dflat_frames: List[SyntheticFrame],
    ) -> Dict[str, List[Path]]:
        """
        Write all frame sets to subfolder structure expected by InstrumentModel.

        Layout::

            output_dir/
              bias/        bias_0000.fits ...
              dark/        dark_0000.fits ...
              flat/        flat_0000.fits ...
              dark_flat/   dark_flat_0000.fits ...

        Returns
        -------
        dict mapping frame type → list of written Paths.
        """
        root = Path(output_dir)
        written: Dict[str, List[Path]] = {}

        frame_sets = {
            "bias":      bias_frames,
            "dark":      dark_frames,
            "flat":      flat_frames,
            "dark_flat": dflat_frames,
        }

        for ftype, frames in frame_sets.items():
            subdir = root / ftype
            subdir.mkdir(parents=True, exist_ok=True)
            paths = []
            for i, sf in enumerate(frames):
                path = subdir / f"{ftype}_{i:04d}.fits"
                hdr  = self._make_header(sf.frame_type, sf.exposure_s)
                fits.writeto(str(path), sf.data, hdr, overwrite=True)
                paths.append(path)
            written[ftype] = paths
            logger.info("Wrote %d %s frames → %s", len(frames), ftype, subdir)

        return written

    def generate_raw_light_frame(
        self,
        sky_signal: np.ndarray,
        exposure_s: float,
    ) -> np.ndarray:
        """
        Apply the full forward model to a true sky signal map.

        This is NOT a calibration frame — it simulates a real light frame
        for use in synthetic_scene.py.

        Forward model:
            raw = bias + Poisson(flat_gain · sky_signal · t) + dark + read_noise

        Parameters
        ----------
        sky_signal : [H, W] float64
            True photon flux map [ADU/s after flat correction].
            Values represent the sky signal at each pixel before flat gain.
        exposure_s : float
            Exposure duration in seconds.

        Returns
        -------
        [H, W] float32 — raw ADU frame as the camera would produce it.
        """
        # Expected signal after flat gain modulation and exposure time
        expected_signal = np.maximum(
            self.instr.flat_gain * sky_signal * exposure_s, 0.0
        )
        # Poisson shot noise on the photon signal
        signal_counts = self.rng.poisson(expected_signal).astype(np.float64)
        # Dark current
        dark_counts = self.rng.poisson(
            np.maximum(self.instr.dark_rate * exposure_s, 0.0)
        ).astype(np.float64)
        # Compose
        raw = (self.instr.bias_mean
               + signal_counts
               + dark_counts
               + self._read_noise_frame())
        return raw.astype(np.float32)
