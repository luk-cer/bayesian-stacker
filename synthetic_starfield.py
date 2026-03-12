"""
synthetic_starfield.py
======================
Realistic synthetic star-field generator for end-to-end pipeline validation.

Generates a super-resolved (HR) sky scene containing point stars and optional
extended emission (Gaussian nebula blobs), then simulates N light frames by:

  1. Applying a known sub-pixel dither shift (phase-shift in Fourier space)
  2. Convolving with a known per-frame PSF (Moffat with varying seeing)
  3. Downsampling to native sensor resolution (average-pool)
  4. Applying the sensor forward model via CalibrationFrameGenerator:
         raw = bias + Poisson(flat_gain · sky · t) + dark + read_noise
  5. Injecting a synthetic WCS header encoding the dither shift

All true values are returned in StarfieldGroundTruth so the pipeline
output can be compared against known quantities.

Design
------
Stars are rendered at the HR grid resolution.  A star at position (y_hr, x_hr)
with flux F_hr [HR-pixel ADU/s] produces a point source that, after convolution
with the PSF and average-pool downsampling, deposits flux F_hr / S² at the
corresponding LR pixel (S = scale_factor).  This conserves total photon counts.

Transparency variation simulates passing thin cloud: t_i ~ Beta(α, β) clipped
to [0.3, 1.0].  Seeing variation: FWHM_i ~ Gamma(k, θ) ≈ N(μ_seeing, σ_seeing²).

Dither pattern: random uniform sub-pixel offsets in [-max_dither_px, +max_dither_px]
at native (LR) resolution, applied as Fourier phase shifts at HR resolution.

Usage
-----
    from synthetic_starfield import StarfieldConfig, generate_starfield

    cfg    = StarfieldConfig.for_asi533(n_stars=200, n_frames=20)
    truth  = generate_starfield(cfg, seed=42)

    # Files on disk
    truth.light_fits_paths   # list[Path] — raw FITS frames
    truth.calib_fits_paths   # dict  bias/dark/flat paths
    truth.model_path         # Path — fitted InstrumentModel HDF5

    # Ground truth arrays
    truth.true_scene_hr      # [S·H, S·W] float32 — sky in HR ADU/s
    truth.true_shifts        # list[(dx, dy)] in LR pixels
    truth.true_fwhm_px       # list[float]  per-frame FWHM at LR pixel scale
    truth.true_transparency  # list[float]

    print(truth.summary())

Dependencies
------------
    numpy  scipy  astropy
    synthetic_calibration  instrument_model_artifact  bayes_calibration
    optics (for ScopeGeometry)
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
try:
    from synthetic_calibration import (
        TrueInstrument,
        CalibrationFrameGenerator,
        sample_instrument_from_priors,
    )
    _SYNTH_CAL_OK = True
except ImportError:
    _SYNTH_CAL_OK = False
    logger.warning("synthetic_calibration.py not found")

try:
    from bayes_calibration import SensorPriors, BayesCalibrationState
    _BAYES_OK = True
except ImportError:
    _BAYES_OK = False
    logger.warning("bayes_calibration.py not found")

try:
    from instrument_model_artifact import InstrumentModel
    _MODEL_OK = True
except ImportError:
    _MODEL_OK = False
    logger.warning("instrument_model_artifact.py not found")


# ============================================================================
# StarfieldConfig
# ============================================================================

@dataclass
class StarfieldConfig:
    """
    Configuration for the synthetic star-field generator.

    Scene
    -----
    shape : (H, W)
        Native sensor resolution in pixels.
    scale_factor : int
        Super-resolution upscaling factor.  HR grid is (S·H, S·W).
    n_stars : int
        Number of point stars to inject.
    star_flux_min_adu_s : float
        Minimum star flux at LR pixel scale [ADU/s].  Faint limit.
    star_flux_max_adu_s : float
        Maximum star flux at LR pixel scale [ADU/s].  Bright limit.
    star_flux_power : float
        Power-law exponent for flux distribution (negative → more faint stars).
        Typical stellar LF: power ≈ -1.5 to -2.0.
    n_nebula_blobs : int
        Number of Gaussian extended-emission blobs to add.
    nebula_flux_adu_s : float
        Peak flux of each nebula blob [ADU/s per HR pixel].
    nebula_radius_lr_px : float
        Gaussian sigma of each blob in LR pixels.

    Frames
    ------
    n_frames : int
        Number of light frames to simulate.
    exposure_s : float
        Exposure time per frame [s].
    max_dither_lr_px : float
        Maximum dither offset in each direction [LR pixels].
        A uniform dither pattern is generated in [-max, +max].
    sky_bg_adu_s : float
        Uniform sky background [ADU/s].

    Seeing / transparency
    ---------------------
    fwhm_mean_lr_px : float
        Mean seeing FWHM in LR pixels.
    fwhm_std_lr_px : float
        Std-dev of seeing FWHM across frames.
    transparency_mean : float
        Mean frame transparency ∈ (0, 1].
    transparency_std : float
        Std-dev of transparency.

    Calibration
    -----------
    n_bias : int
    n_dark : int
    n_flat : int
    dark_exposures : list[float]
        Exposure times for dark frames.

    Output
    ------
    output_dir : Path | None
        Directory to write FITS files.  If None, a TemporaryDirectory is used
        and caller is responsible for cleanup via StarfieldGroundTruth.cleanup().
    """
    # Scene
    shape:               Tuple[int, int] = (256, 256)
    scale_factor:        int             = 2
    n_stars:             int             = 150
    star_flux_min_adu_s: float           = 50.0
    star_flux_max_adu_s: float           = 5_000.0
    star_flux_power:     float           = -1.5
    n_nebula_blobs:      int             = 2
    nebula_flux_adu_s:   float           = 30.0
    nebula_radius_lr_px: float           = 8.0

    # Frames
    n_frames:           int   = 20
    exposure_s:         float = 300.0
    max_dither_lr_px:   float = 3.0
    sky_bg_adu_s:       float = 5.0

    # Seeing / transparency
    fwhm_mean_lr_px:    float = 3.0
    fwhm_std_lr_px:     float = 0.4
    transparency_mean:  float = 0.95
    transparency_std:   float = 0.05

    # Calibration frames
    n_bias:         int        = 20
    n_dark:         int        = 5
    n_flat:         int        = 20
    dark_exposures: List[float] = field(default_factory=lambda: [120., 300., 600.])

    # Output
    output_dir: Optional[Path] = None

    @classmethod
    def for_asi533(
        cls,
        n_stars:  int = 150,
        n_frames: int = 20,
        **kwargs,
    ) -> "StarfieldConfig":
        """
        Config tuned for ZWO ASI533MC-Pro + Esprit 100ED.

        Native resolution 256×256 (sub-region for fast tests).
        Plate scale 1.41″/px → FWHM ≈ 2.5–3.5 LR pixels at good sites.
        """
        return cls(
            shape               = (256, 256),
            scale_factor        = 2,
            n_stars             = n_stars,
            n_frames            = n_frames,
            fwhm_mean_lr_px     = 3.0,
            fwhm_std_lr_px      = 0.3,
            star_flux_max_adu_s = 3_000.0,
            sky_bg_adu_s        = 8.0,
            exposure_s          = 300.0,
            max_dither_lr_px    = 3.5,
            **kwargs,
        )


# ============================================================================
# StarfieldGroundTruth  —  everything the validator needs
# ============================================================================

@dataclass
class StarfieldGroundTruth:
    """
    Complete ground truth for one synthetic imaging session.

    Attributes
    ----------
    config : StarfieldConfig

    true_scene_hr : [S·H, S·W] float32
        True sky scene at HR resolution [ADU/s per HR pixel].
        Does NOT include sky background, calibration artefacts, or noise.

    true_scene_lr : [H, W] float32
        True sky scene downsampled to LR [ADU/s per LR pixel].
        = average_pool(true_scene_hr, S)
        Use for comparing against fast_stack output.

    star_positions_hr : [N, 2] float32
        (y, x) centre positions of each star in HR pixel coordinates.

    star_fluxes_hr : [N] float32
        True flux of each star [ADU/s per HR pixel].

    star_positions_lr : [N, 2] float32
        Star positions in LR pixel coordinates (= positions_hr / S).

    nebula_positions_lr : [B, 2] float32
        Centre of each Gaussian nebula blob in LR pixels.

    true_shifts : list of (dx, dy) float tuples
        Sub-pixel dither offset of each frame in LR pixels.
        Frame 0 is the reference: (0, 0).

    true_fwhm_lr_px : list[float]
        True seeing FWHM in LR pixels for each frame.

    true_psf_kernels : list of [K, K] float32
        Per-frame Moffat PSF kernels (sum=1, at LR pixel scale).

    true_transparency : list[float]
        Per-frame transparency t_i ∈ (0, 1].

    true_instrument : TrueInstrument
        Exact per-pixel sensor parameters used to generate the frames.

    light_fits_paths : list[Path]
        Paths to synthetic raw FITS light frames.

    calib_fits_paths : dict[str, list[Path]]
        Paths to calibration FITS frames (bias / dark / flat / dark_flat).

    model_path : Path
        Path to fitted InstrumentModel HDF5 (from Phase 0 pipeline).

    output_dir : Path
        Root output directory for all FITS and model files.

    _tmpdir : object | None
        TemporaryDirectory handle; non-None when output_dir was auto-created.
    """
    config:             StarfieldConfig

    true_scene_hr:      np.ndarray          # [S·H, S·W] float32
    true_scene_lr:      np.ndarray          # [H, W] float32
    star_positions_hr:  np.ndarray          # [N, 2] float32
    star_fluxes_hr:     np.ndarray          # [N] float32
    star_positions_lr:  np.ndarray          # [N, 2] float32
    nebula_positions_lr: np.ndarray         # [B, 2] float32

    true_shifts:        List[Tuple[float, float]]
    true_fwhm_lr_px:    List[float]
    true_psf_kernels:   List[np.ndarray]
    true_transparency:  List[float]

    true_instrument:    object              # TrueInstrument

    light_fits_paths:   List[Path]
    calib_fits_paths:   Dict[str, List[Path]]
    model_path:         Path
    output_dir:         Path
    _tmpdir:            object = field(default=None, repr=False)

    def cleanup(self) -> None:
        """Remove the temporary directory (if auto-created)."""
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def summary(self) -> str:
        cfg = self.config
        H, W = cfg.shape
        S    = cfg.scale_factor
        lines = [
            "StarfieldGroundTruth",
            "=" * 56,
            f"  LR shape        : {H} × {W}",
            f"  HR shape        : {H*S} × {W*S}  (scale {S}×)",
            f"  Stars           : {len(self.star_fluxes_hr)}",
            f"  Nebula blobs    : {len(self.nebula_positions_lr)}",
            f"  Light frames    : {len(self.light_fits_paths)}",
            f"  Exposure        : {cfg.exposure_s:.0f} s per frame",
            f"  Max dither      : ±{cfg.max_dither_lr_px:.1f} LR px",
            f"  Seeing FWHM     : {float(np.mean(self.true_fwhm_lr_px)):.2f} ± "
                                 f"{float(np.std(self.true_fwhm_lr_px)):.2f} LR px",
            f"  Transparency    : {float(np.mean(self.true_transparency)):.3f} ± "
                                 f"{float(np.std(self.true_transparency)):.3f}",
            f"  Star flux range : [{float(self.star_fluxes_hr.min()):.1f},"
                                f" {float(self.star_fluxes_hr.max()):.1f}] HR ADU/s",
            f"  Scene HR        : min={self.true_scene_hr.min():.2f} "
                                f"max={self.true_scene_hr.max():.2f} "
                                f"mean={self.true_scene_hr.mean():.2f}",
            f"  Output dir      : {self.output_dir}",
            "=" * 56,
        ]
        return "\n".join(lines)


# ============================================================================
# Internal helpers
# ============================================================================

def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _moffat_psf(size: int, fwhm_px: float, beta: float = 2.5) -> np.ndarray:
    """
    Normalised 2-D Moffat PSF kernel.

    Moffat(r) = (β-1)/(π·α²) · (1 + r²/α²)^(-β)
    α = FWHM / (2·sqrt(2^(1/β) - 1))

    Parameters
    ----------
    size   : odd integer kernel side length
    fwhm_px: FWHM in pixels
    beta   : Moffat β parameter (2.0–4.0 for atmosphere)

    Returns
    -------
    [size, size] float32, sum=1
    """
    if size % 2 == 0:
        size += 1
    alpha = fwhm_px / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    half  = size // 2
    y, x  = np.mgrid[-half:half+1, -half:half+1].astype(np.float64)
    r2    = x**2 + y**2
    psf   = (1.0 + r2 / alpha**2) ** (-beta)
    psf  /= psf.sum()
    return psf.astype(np.float32)


def _power_law_fluxes(
    n:       int,
    f_min:   float,
    f_max:   float,
    power:   float,
    rng:     np.random.Generator,
) -> np.ndarray:
    """
    Sample N fluxes from a truncated power-law distribution.

    p(F) ∝ F^power,  F ∈ [f_min, f_max],  power < -1.

    Uses inverse CDF method.
    """
    p1  = power + 1.0
    if abs(p1) < 1e-6:
        # Uniform in log space
        log_f = rng.uniform(np.log(f_min), np.log(f_max), size=n)
        return np.exp(log_f).astype(np.float32)
    lo  = f_min ** p1
    hi  = f_max ** p1
    u   = rng.uniform(0.0, 1.0, size=n)
    return (lo + u * (hi - lo)).astype(np.float64) ** (1.0 / p1)


def _render_stars_hr(
    shape_hr:     Tuple[int, int],
    positions_hr: np.ndarray,    # [N, 2]  (y, x)
    fluxes_hr:    np.ndarray,    # [N]     ADU/s per HR px (peak before normalise)
    psf_hr:       np.ndarray,    # [K, K]  PSF at HR pixel scale, sum=1
) -> np.ndarray:
    """
    Render star point sources on the HR grid by placing scaled PSF stamps.

    Each star at (y_hr, x_hr) with flux F contributes F·PSF(·-r) to the scene.
    The PSF is used as a stamp — stars are rendered at sub-pixel accuracy by
    using the nearest HR grid pixel (HR pixel = 1/S LR pixel, so ½ LR pixel
    accuracy is already at the full PSF precision).

    Returns
    -------
    [S·H, S·W] float32  star-only scene, units: ADU/s per HR pixel
    """
    sH, sW = shape_hr
    K      = psf_hr.shape[0]
    half   = K // 2
    scene  = np.zeros((sH, sW), dtype=np.float64)

    for (y, x), f in zip(positions_hr, fluxes_hr):
        cy, cx = int(round(y)), int(round(x))
        r0 = max(0, cy - half);  r1 = min(sH, cy + half + 1)
        c0 = max(0, cx - half);  c1 = min(sW, cx + half + 1)
        pr0 = r0 - (cy - half);  pr1 = pr0 + (r1 - r0)
        pc0 = c0 - (cx - half);  pc1 = pc0 + (c1 - c0)
        scene[r0:r1, c0:c1] += f * psf_hr[pr0:pr1, pc0:pc1]

    return scene.astype(np.float32)


def _render_nebula_hr(
    shape_hr:     Tuple[int, int],
    n_blobs:      int,
    peak_adu_s:   float,
    radius_hr_px: float,
    rng:          np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add Gaussian blob(s) to a scene at random positions.

    Returns
    -------
    blobs : [S·H, S·W] float32
    positions_hr : [n_blobs, 2] float32   (y, x) centres in HR pixels
    """
    sH, sW = shape_hr
    scene  = np.zeros((sH, sW), dtype=np.float64)
    # Margin: keep blobs at least 3-sigma from edges, but never exceed half the image
    margin = min(int(radius_hr_px * 3), sH // 4, sW // 4)
    margin = max(margin, 1)
    positions = []

    for _ in range(n_blobs):
        cy = int(rng.integers(margin, sH - margin))
        cx = int(rng.integers(margin, sW - margin))
        positions.append([float(cy), float(cx)])
        y, x = np.mgrid[0:sH, 0:sW].astype(np.float64)
        r2   = (y - cy)**2 + (x - cx)**2
        blob = peak_adu_s * np.exp(-0.5 * r2 / radius_hr_px**2)
        scene += blob

    pos_arr = np.array(positions, dtype=np.float32) if positions else \
              np.zeros((0, 2), dtype=np.float32)
    return scene.astype(np.float32), pos_arr


def _phase_shift_hr(
    scene_hr:  np.ndarray,
    dx_lr:     float,
    dy_lr:     float,
    scale:     int,
) -> np.ndarray:
    """
    Apply sub-pixel shift (dx_lr, dy_lr) in LR pixels to the HR scene.
    The equivalent HR shift is (dx_lr/scale, dy_lr/scale) HR pixels.
    Uses Fourier phase-shift theorem (exact for periodic signals).
    """
    sH, sW = scene_hr.shape
    dx_hr  = dx_lr / scale
    dy_hr  = dy_lr / scale
    fy     = np.fft.fftfreq(sH).reshape(-1, 1)
    fx     = np.fft.rfftfreq(sW).reshape(1, -1)
    phase  = np.exp(-2j * np.pi * (fy * dy_hr + fx * dx_hr))
    spec   = np.fft.rfft2(scene_hr.astype(np.float64))
    return np.real(np.fft.irfft2(spec * phase, s=(sH, sW))).astype(np.float32)


def _convolve_psf(scene_hr: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Zero-padded FFT convolution of scene_hr with psf."""
    sH, sW = scene_hr.shape
    K      = psf.shape[0]
    pH     = _next_power_of_2(sH + K)
    pW     = _next_power_of_2(sW + K)

    pad    = np.zeros((pH, pW), dtype=np.float64)
    pad[:sH, :sW] = scene_hr

    psf_pad  = np.zeros((pH, pW), dtype=np.float64)
    cy, cx   = K // 2, K // 2
    psf_pad[:K, :K] = psf
    psf_pad  = np.roll(np.roll(psf_pad, -cy, 0), -cx, 1)

    result   = np.real(np.fft.irfft2(
        np.fft.rfft2(pad) * np.fft.rfft2(psf_pad), s=(pH, pW)
    ))
    return np.maximum(result[:sH, :sW], 0.0).astype(np.float32)


def _downsample(arr: np.ndarray, scale: int) -> np.ndarray:
    """Average-pool downsampling by integer scale factor."""
    H, W = arr.shape
    return arr.reshape(H // scale, scale, W // scale, scale).mean(axis=(1, 3))


def _make_wcs_header(
    dx_lr:    float,
    dy_lr:    float,
    shape:    Tuple[int, int],
    plate_scale_deg_per_px: float = 1.41 / 3600.0,
) -> fits.Header:
    """
    Create a minimal FITS WCS header encoding a frame shift.

    The reference pixel is the frame centre.  The shift (dx, dy) is encoded
    by offsetting CRVAL (field centre RA/Dec) by the equivalent angular amount.

    This is a simplified WCS — suitable for the FrameCharacterizer's
    extract_wcs_geometry() → compute_frame_shift() path.

    Parameters
    ----------
    dx_lr, dy_lr : shift in LR pixels (positive dx = rightward = increasing RA)
    shape        : (H, W) of the frame
    plate_scale_deg_per_px : deg/px at LR scale (default 1.41″/px = 0.000392 deg/px)
    """
    H, W  = shape
    hdr   = fits.Header()

    # Base pointing
    ra0, dec0 = 83.8221, -5.3911   # Orion Nebula centre (fixed reference)

    # Shift in degrees
    ra_shift  =  dx_lr * plate_scale_deg_per_px / np.cos(np.radians(dec0))
    dec_shift = -dy_lr * plate_scale_deg_per_px

    hdr["NAXIS"]  = 2
    hdr["NAXIS1"] = W
    hdr["NAXIS2"] = H
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = W / 2.0
    hdr["CRPIX2"] = H / 2.0
    hdr["CRVAL1"] = ra0  + ra_shift
    hdr["CRVAL2"] = dec0 + dec_shift
    hdr["CD1_1"]  = -plate_scale_deg_per_px
    hdr["CD1_2"]  =  0.0
    hdr["CD2_1"]  =  0.0
    hdr["CD2_2"]  =  plate_scale_deg_per_px
    hdr["EXPTIME"] = 0.0   # filled later
    hdr["IMAGETYP"] = "LIGHT"
    hdr["BAYERPAT"]  = "RGGB"
    hdr["COMMENT"] = f"Synthetic frame dx={dx_lr:.4f} dy={dy_lr:.4f} LR-px"
    return hdr


# ============================================================================
# Main generator
# ============================================================================

def generate_starfield(
    cfg:  StarfieldConfig,
    seed: int = 42,
) -> StarfieldGroundTruth:
    """
    Generate a complete synthetic imaging session.

    Parameters
    ----------
    cfg  : StarfieldConfig
    seed : int   random seed for full reproducibility

    Returns
    -------
    StarfieldGroundTruth  (see class docstring for all fields)
    """
    if not (_SYNTH_CAL_OK and _BAYES_OK and _MODEL_OK):
        raise ImportError(
            "generate_starfield() requires synthetic_calibration, "
            "bayes_calibration, and instrument_model_artifact modules."
        )

    rng = np.random.default_rng(seed)
    H, W = cfg.shape
    S    = cfg.scale_factor
    sH, sW = H * S, W * S

    # ── 1. True instrument parameters ────────────────────────────────────
    logger.info("Sampling true instrument parameters …")
    priors = SensorPriors.for_asi533_gain100()
    instr  = sample_instrument_from_priors(priors, shape=(H, W), rng=rng)

    # ── 2. Generate calibration frames ───────────────────────────────────
    logger.info("Generating calibration frames …")
    # Resolve output directory
    _tmpdir = None
    if cfg.output_dir is None:
        _tmpdir   = tempfile.TemporaryDirectory()
        out_root  = Path(_tmpdir.name)
    else:
        out_root  = Path(cfg.output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

    gen = CalibrationFrameGenerator(instr, rng=rng)
    bias_frames  = gen.generate_bias_frames(n=cfg.n_bias)
    dark_frames  = gen.generate_dark_frames(
        exposures=cfg.dark_exposures, repeats=cfg.n_dark
    )
    flat_frames  = gen.generate_flat_frames(n=cfg.n_flat)
    dflat_frames = gen.generate_dark_flat_frames(n=max(cfg.n_flat // 2, 5))

    calib_paths = gen.write_fits_folder(
        out_root / "calibration",
        bias_frames, dark_frames, flat_frames, dflat_frames,
    )
    logger.info("Calibration frames written.")

    # ── 3. Fit InstrumentModel from calibration frames ───────────────────
    logger.info("Fitting InstrumentModel from calibration frames …")
    model     = InstrumentModel.fit_all(out_root / "calibration")
    model_path = out_root / "instrument_model.h5"
    model.save(model_path)
    logger.info("InstrumentModel saved → %s", model_path)

    # ── 4. Build true HR scene ────────────────────────────────────────────
    logger.info("Building true HR scene (%d × %d) …", sH, sW)

    # Seeing PSF for rendering stars: use mean seeing at HR scale
    fwhm_hr_mean = cfg.fwhm_mean_lr_px * S
    psf_render   = _moffat_psf(size=max(31, int(fwhm_hr_mean * 6) | 1),
                                fwhm_px=fwhm_hr_mean)

    # Star positions: random, with border margin
    margin_hr  = int(fwhm_hr_mean * 3)
    star_y_hr  = rng.uniform(margin_hr, sH - margin_hr, size=cfg.n_stars)
    star_x_hr  = rng.uniform(margin_hr, sW - margin_hr, size=cfg.n_stars)
    pos_hr     = np.stack([star_y_hr, star_x_hr], axis=1).astype(np.float32)

    # Star fluxes: power law, at HR pixel scale
    # HR pixel is 1/S × LR pixel, so flux per HR px = flux_lr / S²
    fluxes_lr  = _power_law_fluxes(
        cfg.n_stars,
        cfg.star_flux_min_adu_s,
        cfg.star_flux_max_adu_s,
        cfg.star_flux_power,
        rng,
    ).astype(np.float32)
    fluxes_hr  = fluxes_lr / (S * S)

    # Render stars on HR grid
    stars_hr   = _render_stars_hr((sH, sW), pos_hr, fluxes_hr, psf_render)

    # Nebula blobs
    radius_hr  = cfg.nebula_radius_lr_px * S
    nebula_hr, neb_pos_hr = _render_nebula_hr(
        (sH, sW), cfg.n_nebula_blobs,
        cfg.nebula_flux_adu_s / (S * S),
        radius_hr,
        rng,
    )
    neb_pos_lr = neb_pos_hr / S

    # True scene HR (stars + nebula, no sky background)
    scene_hr = (stars_hr + nebula_hr).astype(np.float32)
    scene_lr = _downsample(scene_hr.astype(np.float64), S).astype(np.float32)

    # LR star positions
    pos_lr = pos_hr / S

    logger.info(
        "True HR scene: n_stars=%d  n_blobs=%d  "
        "scene_hr max=%.1f  scene_lr max=%.1f",
        cfg.n_stars, cfg.n_nebula_blobs,
        float(scene_hr.max()), float(scene_lr.max()),
    )

    # ── 5. Generate per-frame variations ─────────────────────────────────
    fwhm_list   : List[float]               = []
    t_list      : List[float]               = []
    shift_list  : List[Tuple[float, float]] = []
    psf_list    : List[np.ndarray]          = []

    for i in range(cfg.n_frames):
        # Seeing FWHM (clip to [1.5, 8.0] LR px)
        fwhm_px = float(np.clip(
            rng.normal(cfg.fwhm_mean_lr_px, cfg.fwhm_std_lr_px), 1.5, 8.0
        ))
        fwhm_list.append(fwhm_px)

        # Transparency (clip to [0.3, 1.0])
        t = float(np.clip(
            rng.normal(cfg.transparency_mean, cfg.transparency_std), 0.3, 1.0
        ))
        t_list.append(t)

        # Dither shift in LR pixels; frame 0 is reference (0, 0)
        if i == 0:
            dx, dy = 0.0, 0.0
        else:
            dx = float(rng.uniform(-cfg.max_dither_lr_px, cfg.max_dither_lr_px))
            dy = float(rng.uniform(-cfg.max_dither_lr_px, cfg.max_dither_lr_px))
        shift_list.append((dx, dy))

        # PSF kernel at LR scale (for ground truth comparison)
        psf_lr = _moffat_psf(size=31, fwhm_px=fwhm_px)
        psf_list.append(psf_lr)

    # ── 6. Generate light FITS frames ────────────────────────────────────
    logger.info("Generating %d light frames …", cfg.n_frames)
    lights_dir = out_root / "lights"
    lights_dir.mkdir(parents=True, exist_ok=True)
    light_paths: List[Path] = []

    for i, (dx, dy) in enumerate(shift_list):
        fwhm_px   = fwhm_list[i]
        t_i       = t_list[i]
        fwhm_hr   = fwhm_px * S

        # Per-frame PSF at HR scale
        psf_hr_i  = _moffat_psf(
            size   = max(31, int(fwhm_hr * 6) | 1),
            fwhm_px= fwhm_hr,
        )

        # a. Apply dither shift in HR Fourier space
        shifted = _phase_shift_hr(scene_hr, dx, dy, S)

        # b. Convolve with per-frame PSF at HR scale
        convolved = _convolve_psf(shifted, psf_hr_i)

        # c. Downsample to LR [ADU/s per LR pixel]
        lr_signal = _downsample(convolved.astype(np.float64), S)

        # d. Add sky background
        sky = np.full((H, W), cfg.sky_bg_adu_s * cfg.exposure_s, dtype=np.float64)
        lr_total = lr_signal * t_i + sky / cfg.exposure_s

        # e. Generate raw frame using instrument forward model
        # scale by exposure_s to get ADU counts
        sky_signal_for_gen = lr_total  # ADU/s; generator multiplies by exposure_s

        raw = gen.generate_raw_light_frame(
            sky_signal = sky_signal_for_gen,
            exposure_s = cfg.exposure_s,
        )

        # f. Write FITS with WCS header encoding the shift
        hdr = _make_wcs_header(dx, dy, (H, W))
        hdr["EXPTIME"]  = cfg.exposure_s
        hdr["FRAME"]    = i
        hdr["TRUE_DX"]  = dx
        hdr["TRUE_DY"]  = dy
        hdr["TRUE_T"]   = t_i
        hdr["TRUE_FWHM"]= fwhm_px
        hdr["CCD-TEMP"] = -10.0
        hdr["GAIN"]     = 100.0

        path = lights_dir / f"light_{i:04d}.fits"
        fits.writeto(str(path), raw.astype(np.float32), hdr, overwrite=True)
        light_paths.append(path)

        logger.info(
            "  Frame %3d/%d  dx=%+.2f  dy=%+.2f  FWHM=%.2f px  t=%.3f",
            i+1, cfg.n_frames, dx, dy, fwhm_px, t_i,
        )

    logger.info("All %d light frames written to %s", cfg.n_frames, lights_dir)

    truth = StarfieldGroundTruth(
        config              = cfg,
        true_scene_hr       = scene_hr,
        true_scene_lr       = scene_lr,
        star_positions_hr   = pos_hr,
        star_fluxes_hr      = fluxes_hr,
        star_positions_lr   = pos_lr,
        nebula_positions_lr = neb_pos_lr,
        true_shifts         = shift_list,
        true_fwhm_lr_px     = fwhm_list,
        true_psf_kernels    = psf_list,
        true_transparency   = t_list,
        true_instrument     = instr,
        light_fits_paths    = light_paths,
        calib_fits_paths    = calib_paths,
        model_path          = model_path,
        output_dir          = out_root,
        _tmpdir             = _tmpdir,
    )
    logger.info(truth.summary())
    return truth
