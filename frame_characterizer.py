"""
frame_characterizer.py
======================
Per-frame characterisation: sub-pixel shift, seeing PSF, transparency,
and sky background.

For each calibrated light frame this module produces a FrameMetadata object
containing everything the Phase 4 MAP stacker needs to model that frame:

    FrameMetadata
      shift        : FrameShift | None   sub-pixel (dx, dy, rotation, scale)
      psf_total    : ndarray [K,K]       empirical PSF  H_total,i  (sum=1)
      psf_seeing   : ndarray [K,K]       atmospheric component H_seeing,i
      transparency : float               t_i in (0,1]  throughput multiplier
      sky_bg       : ndarray [H,W]       smooth 2-D sky background [ADU]
      fwhm_arcsec  : float               seeing FWHM in arcsec
      fwhm_pixels  : float               seeing FWHM in pixels
      n_stars_used : int
      solve_status : str                 'wcs' | 'failed'

Pipeline per frame
------------------
  1. Load raw FITS → calibrate with InstrumentModel
  2. Estimate 2-D sky background  (sigma-clip + 2-D Legendre polynomial fit)
  3. Extract isolated, unsaturated star positions and K×K stamp cutouts
  4. Median-stack normalised stamps → empirical H_total kernel
  5. Fit 2-D Moffat profile → FWHM, β (sub-pixel centroid only for stamp alignment)
  6. Wiener-deconvolve H_instrument from H_total → H_seeing
  7. Aperture photometry on detected stars → transparency vs reference frame
  8. Plate solve FITS header (or call ASTAP) → WCSGeometry → FrameShift

Every step degrades gracefully:
  - Plate solve failure  → shift = None (stacker uses (0,0))
  - < min_stars          → PSF falls back to previous-frame estimate, then Gaussian
  - < 2 valid fluxes     → transparency = 1.0

Physical conventions
--------------------
  PSF kernels  float32, odd square, sum = 1.0
  sky_bg       [ADU] — smooth continuum; subtract before accumulation
  transparency t_i ∈ (0, 1] — ratio of stellar flux to reference frame

Dependencies
------------
  numpy  scipy  astropy  optics (project module)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import curve_fit
from astropy.io import fits

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional project import — optics.py
# ---------------------------------------------------------------------------
try:
    from optics import (
        ScopeGeometry,
        WCSGeometry,
        FrameShift,
        extract_wcs_geometry,
        compute_frame_shift,
        get_instrument_psf,
        ASTAPSolver,
    )
    _OPTICS_OK = True
except ImportError:                          # allow isolated unit tests
    _OPTICS_OK = False
    ScopeGeometry = None                     # type: ignore
    WCSGeometry   = None                     # type: ignore
    FrameShift    = None                     # type: ignore
    ASTAPSolver   = None                     # type: ignore
    logger.warning("optics.py not importable — WCS / PSF features disabled")


# ============================================================================
# FrameMetadata
# ============================================================================

@dataclass
class FrameMetadata:
    """
    All per-frame quantities needed by the MAP stacker.

    shift
        Sub-pixel translation + rotation relative to the reference frame.
        None when plate solving failed; the stacker treats that as (0,0).
    psf_total
        Empirical full-system PSF H_total,i estimated from star stamps.
        Sum = 1.0.  Use directly in the stacker forward model.
    psf_seeing
        Atmospheric component H_seeing,i obtained by Wiener-deconvolving
        H_instrument from psf_total.  Sum = 1.0.
    transparency
        Per-frame throughput t_i.  1.0 on a photometric night.  Used as
        the Poisson rate multiplier λ = t_i · g · sky in the stacker.
    sky_bg
        Smooth 2-D sky background estimate [ADU, H×W float32].
        Subtract from calibrated frame before feeding the stacker.
    fwhm_arcsec / fwhm_pixels
        Seeing FWHM from the Moffat fit to psf_total.
    n_stars_used
        Number of stars that contributed to the PSF stack.
    solve_status
        'wcs'    — shift derived from WCS plate solution
        'failed' — no WCS available; shift is None
    frame_path
        Source FITS path (for logging only).
    """
    shift:        Optional[object]    # FrameShift | None
    psf_total:    np.ndarray
    psf_seeing:   np.ndarray
    transparency: float
    sky_bg:       np.ndarray
    fwhm_arcsec:  float
    fwhm_pixels:  float
    n_stars_used: int
    solve_status: str
    frame_path:   Optional[Path] = None

    def summary(self) -> str:
        if self.shift is not None:
            sh = (f"dx={self.shift.dx_px:+.3f}px  dy={self.shift.dy_px:+.3f}px  "
                  f"rot={self.shift.rotation_deg:+.4f}°")
        else:
            sh = "None (solve failed)"
        return "\n".join([
            "FrameMetadata",
            f"  shift        : {sh}",
            f"  transparency : {self.transparency:.4f}",
            f"  FWHM         : {self.fwhm_arcsec:.2f}\"  /  {self.fwhm_pixels:.2f} px",
            f"  n_stars      : {self.n_stars_used}",
            f"  sky_bg       : median = {float(np.median(self.sky_bg)):.1f} ADU",
            f"  solve_status : {self.solve_status}",
        ])


# ============================================================================
# Sky background  (sigma-clip + 2-D Legendre polynomial)
# ============================================================================

def estimate_sky_background(
    frame:       np.ndarray,
    poly_degree: int   = 2,
    sigma_clip:  float = 3.0,
    n_iter:      int   = 3,
    box_size:    int   = 64,
) -> np.ndarray:
    """
    Estimate a smooth 2-D sky background.

    Algorithm
    ---------
    1. Partition the frame into box_size × box_size tiles.
    2. Compute per-tile median and MAD.
    3. Reject pixels more than sigma_clip × MAD from their tile median
       (iterating n_iter times).
    4. Fit a 2-D Legendre polynomial of degree poly_degree to the surviving
       (background) pixels using least squares.
    5. Evaluate the polynomial on the full pixel grid.

    Returns
    -------
    sky : [H, W] float32
    """
    H, W = frame.shape
    f    = frame.astype(np.float64)

    # Build source mask (True = background pixel, keep for fit)
    mask = np.ones((H, W), dtype=bool)
    for _ in range(n_iter):
        n_rows = max(1, H // box_size)
        n_cols = max(1, W // box_size)
        for tr in range(n_rows):
            r0, r1 = tr * box_size, min((tr + 1) * box_size, H)
            for tc in range(n_cols):
                c0, c1 = tc * box_size, min((tc + 1) * box_size, W)
                vals = f[r0:r1, c0:c1][mask[r0:r1, c0:c1]]
                if vals.size < 4:
                    continue
                med = float(np.median(vals))
                mad = float(np.median(np.abs(vals - med))) * 1.4826
                if mad < 1e-6:
                    continue
                lo = med - sigma_clip * mad
                hi = med + sigma_clip * mad
                mask[r0:r1, c0:c1] &= (f[r0:r1, c0:c1] >= lo) & \
                                       (f[r0:r1, c0:c1] <= hi)

    # Normalised coordinates [-1, 1]
    Y_g, X_g = np.mgrid[0:H, 0:W]
    xn = (X_g.astype(np.float64) / max(W - 1, 1)) * 2.0 - 1.0
    yn = (Y_g.astype(np.float64) / max(H - 1, 1)) * 2.0 - 1.0

    # Legendre polynomial terms  (degree px in x, py in y,  px+py <= poly_degree)
    terms = []
    for px in range(poly_degree + 1):
        for py in range(poly_degree + 1 - px):
            Lx = np.polynomial.legendre.legval(xn, [0] * px + [1])
            Ly = np.polynomial.legendre.legval(yn, [0] * py + [1])
            terms.append(Lx * Ly)

    bg_vals = f[mask]
    A       = np.column_stack([t[mask] for t in terms])   # [N_bg, N_terms]

    if bg_vals.size <= len(terms):
        logger.warning("Too few background pixels — returning global median")
        return np.full((H, W), float(np.median(f)), dtype=np.float32)

    try:
        coeffs, *_ = np.linalg.lstsq(A, bg_vals, rcond=None)
    except np.linalg.LinAlgError:
        logger.warning("Sky fit failed — returning global median")
        return np.full((H, W), float(np.median(bg_vals)), dtype=np.float32)

    sky = sum(c * t for c, t in zip(coeffs, terms))
    return sky.astype(np.float32)


# ============================================================================
# Star extraction
# ============================================================================

def extract_stars(
    frame:          np.ndarray,
    sky_bg:         np.ndarray,
    snr_threshold:  float = 20.0,
    saturation_adu: float = 60_000.0,
    min_sep_px:     float = 20.0,
    stamp_size:     int   = 31,
    max_stars:      int   = 50,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Detect isolated, unsaturated stars and return positions + stamp cutouts.

    Parameters
    ----------
    frame          : [H, W] float32   calibrated frame (not sky-subtracted)
    sky_bg         : [H, W] float32   sky background (subtracted internally)
    snr_threshold  : peak SNR above sky RMS required for detection
    saturation_adu : reject stars with any pixel above this level
    min_sep_px     : minimum centre-to-centre separation between kept stars
    stamp_size     : cutout side length in pixels (forced odd)
    max_stars      : maximum number of stars to return (brightest first)

    Returns
    -------
    positions : [N, 2] float32   sub-pixel (y, x) centroids
    stamps    : list of N float32 [stamp_size, stamp_size] sky-subtracted cutouts
    """
    if stamp_size % 2 == 0:
        stamp_size += 1
    half = stamp_size // 2
    H, W = frame.shape
    sub  = frame.astype(np.float64) - sky_bg.astype(np.float64)

    # Sky noise estimate: RMS of background pixels (below sky median)
    bg_mask  = sub < float(np.percentile(sub, 60))
    sky_rms  = float(np.std(sub[bg_mask])) if bg_mask.sum() > 100 else 1.0
    sky_rms  = max(sky_rms, 1.0)

    # Smooth slightly → robust peak detection
    smoothed = gaussian_filter(sub, sigma=1.0)

    # Local maxima: pixel == neighbourhood max  AND  above SNR floor
    neighborhood = maximum_filter(smoothed, size=max(3, int(min_sep_px // 2)))
    local_max    = (smoothed == neighborhood) & \
                   (smoothed > snr_threshold * sky_rms)

    # Exclude border (need full stamp)
    border = half + 2
    local_max[:border,  :] = False
    local_max[-border:, :] = False
    local_max[:,  :border] = False
    local_max[:, -border:] = False

    ys, xs = np.where(local_max)
    if len(ys) == 0:
        logger.warning("extract_stars: no peaks above SNR %.0f", snr_threshold)
        return np.zeros((0, 2), dtype=np.float32), []

    # Sort brightest first
    order  = np.argsort(-smoothed[ys, xs])
    ys, xs = ys[order], xs[order]

    kept_y: List[int] = []
    kept_x: List[int] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        y, x = int(y), int(x)
        # Saturation check
        r0, r1 = max(0, y - half), min(H, y + half + 1)
        c0, c1 = max(0, x - half), min(W, x + half + 1)
        if float(frame[r0:r1, c0:c1].max()) > saturation_adu:
            continue
        # Isolation check
        if kept_y:
            dy = np.array(kept_y, dtype=np.float64) - y
            dx = np.array(kept_x, dtype=np.float64) - x
            if float(np.min(np.sqrt(dy**2 + dx**2))) < min_sep_px:
                continue
        kept_y.append(y)
        kept_x.append(x)
        if len(kept_y) >= max_stars:
            break

    if not kept_y:
        logger.warning("extract_stars: no isolated unsaturated stars found")
        return np.zeros((0, 2), dtype=np.float32), []

    # Sub-pixel centroids (flux-weighted 5×5 window on sky-subtracted frame)
    positions: List[List[float]] = []
    stamps:    List[np.ndarray]  = []
    for y, x in zip(kept_y, kept_x):
        wy0, wy1 = max(0, y - 2), min(H, y + 3)
        wx0, wx1 = max(0, x - 2), min(W, x + 3)
        win  = sub[wy0:wy1, wx0:wx1]
        norm = float(win.clip(min=0).sum())
        if norm > 0:
            yy, xx = np.mgrid[wy0:wy1, wx0:wx1].astype(np.float64)
            cy = float((yy * win.clip(min=0)).sum() / norm)
            cx = float((xx * win.clip(min=0)).sum() / norm)
        else:
            cy, cx = float(y), float(x)

        r0, r1 = y - half, y - half + stamp_size
        c0, c1 = x - half, x - half + stamp_size
        if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
            continue

        positions.append([cy, cx])
        stamps.append(sub[r0:r1, c0:c1].astype(np.float32))

    logger.info(
        "extract_stars: %d stars  (SNR > %.0f, sep > %.0f px)",
        len(positions), snr_threshold, min_sep_px,
    )
    if not positions:
        return np.zeros((0, 2), dtype=np.float32), []
    return np.array(positions, dtype=np.float32), stamps


# ============================================================================
# PSF helpers
# ============================================================================

def _gaussian_psf(size: int, fwhm_px: float) -> np.ndarray:
    """Analytic 2-D Gaussian PSF kernel, sum=1, float32."""
    half  = size // 2
    sigma = fwhm_px / 2.3548
    y, x  = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    g     = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    g    /= g.sum()
    return g.astype(np.float32)


def _fourier_shift(arr: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Sub-pixel shift via Fourier phase-shift theorem (no interpolation)."""
    from numpy.fft import fft2, ifft2, fftfreq
    H, W  = arr.shape
    fy    = fftfreq(H).reshape(-1, 1)
    fx    = fftfreq(W).reshape(1, -1)
    phase = np.exp(-2j * np.pi * (fy * dy + fx * dx))
    return np.real(ifft2(fft2(arr) * phase))


def _moffat2d_flat(xy, amp, x0, y0, fwhm, beta, bg):
    """2-D Moffat profile for scipy.optimize.curve_fit (returns flattened array)."""
    x, y   = xy
    beta   = max(beta, 0.5)
    alpha  = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    alpha  = max(alpha, 0.1)
    r2     = (x - x0) ** 2 + (y - y0) ** 2
    model  = amp * (1.0 + r2 / alpha ** 2) ** (-beta) + bg
    return model.ravel()


def fit_moffat(stamp: np.ndarray) -> Tuple[float, float]:
    """
    Fit a 2-D Moffat profile to a background-subtracted star stamp.

    Returns
    -------
    fwhm_px : float   FWHM in pixels
    beta    : float   Moffat power-law index (typical: 2–5 for atmospheric seeing)
    """
    K     = stamp.shape[0]
    half  = K / 2.0
    peak  = float(stamp.max())
    if peak <= 0:
        return K / 4.0, 2.5

    y_g, x_g = np.mgrid[0:K, 0:K].astype(np.float64)
    p0        = [peak, half, half, K / 4.0, 2.5, 0.0]
    lo        = [0,    0,    0,    0.5,     0.5, -peak * 0.2]
    hi        = [peak * 5, K, K,  float(K), 10.0, peak * 0.2]

    try:
        popt, _ = curve_fit(
            _moffat2d_flat,
            (x_g.ravel(), y_g.ravel()),
            stamp.ravel().astype(np.float64),
            p0     = p0,
            bounds = (lo, hi),
            maxfev = 3000,
        )
        fwhm = float(np.clip(popt[3], 0.5, K))
        beta = float(np.clip(popt[4], 0.5, 10.0))
        return fwhm, beta
    except Exception:
        pass

    # Fallback: second-moment radius
    s    = stamp.astype(np.float64)
    s    = np.maximum(s, 0.0)
    norm = s.sum()
    if norm <= 0:
        return K / 4.0, 2.5
    yc   = float((y_g * s).sum() / norm)
    xc   = float((x_g * s).sum() / norm)
    r2   = (x_g - xc) ** 2 + (y_g - yc) ** 2
    sigma2 = float((r2 * s).sum() / norm)
    return 2.3548 * float(np.sqrt(max(sigma2, 0.1))), 2.5


# ============================================================================
# PSF estimation from stamps
# ============================================================================

def estimate_psf_from_stamps(
    stamps:   List[np.ndarray],
    psf_size: int = 31,
) -> Tuple[np.ndarray, float, float]:
    """
    Build an empirical PSF kernel H_total by median-stacking star stamps.

    Each stamp is normalised to unit sum, then Fourier-shifted to align its
    centroid to the stamp centre before stacking.  The median stack
    suppresses cosmic rays, bleed trails, and bright neighbours.

    Returns
    -------
    psf_total : [psf_size, psf_size] float32   sum = 1.0
    fwhm_px   : float
    beta      : float
    """
    if psf_size % 2 == 0:
        psf_size += 1

    if not stamps:
        return _gaussian_psf(psf_size, 3.0), 3.0, 2.5

    aligned = []
    for stamp in stamps:
        s     = stamp.astype(np.float64)
        total = float(s.sum())
        if total <= 0:
            continue
        s /= total

        K     = s.shape[0]
        cy    = K / 2.0
        yy, xx = np.mgrid[0:K, 0:K].astype(np.float64)
        s_pos  = np.maximum(s, 0.0)
        n      = float(s_pos.sum())
        if n > 0:
            yc = float((yy * s_pos).sum() / n)
            xc = float((xx * s_pos).sum() / n)
        else:
            yc, xc = cy, cy

        dy, dx = cy - yc, cy - xc
        if abs(dy) > 4 or abs(dx) > 4:
            continue    # centroid too far off — badly blended star

        s_shift = _fourier_shift(s, dy, dx)

        # Crop or pad to psf_size
        if K == psf_size:
            s_out = s_shift
        elif K > psf_size:
            m     = (K - psf_size) // 2
            s_out = s_shift[m:m + psf_size, m:m + psf_size]
        else:
            pad   = psf_size - K
            lo, hi = pad // 2, pad - pad // 2
            s_out = np.pad(s_shift, ((lo, hi), (lo, hi)))

        aligned.append(s_out)

    if not aligned:
        return _gaussian_psf(psf_size, 3.0), 3.0, 2.5

    stack    = np.stack(aligned, axis=0)
    psf_raw  = np.median(stack, axis=0)
    psf_raw  = np.maximum(psf_raw, 0.0)
    total    = float(psf_raw.sum())
    if total <= 0:
        return _gaussian_psf(psf_size, 3.0), 3.0, 2.5

    psf_norm = (psf_raw / total).astype(np.float32)
    fwhm_px, beta = fit_moffat(psf_norm)
    logger.info(
        "estimate_psf_from_stamps: %d/%d stamps used  FWHM=%.2fpx  beta=%.2f",
        len(aligned), len(stamps), fwhm_px, beta,
    )
    return psf_norm, fwhm_px, beta


# ============================================================================
# Instrument PSF deconvolution → H_seeing
# ============================================================================

def deconvolve_instrument_psf(
    psf_total:      np.ndarray,
    psf_instrument: np.ndarray,
    wiener_snr:     float = 10.0,
) -> np.ndarray:
    """
    Recover the atmospheric PSF H_seeing by Wiener deconvolution.

    Model:  H_total = H_instrument ⊛ H_seeing
    Solution (Wiener):
        H_seeing(f) = H_total(f) / H_instrument(f)  ×  W(f)
        W(f) = |H_instr(f)|² / (|H_instr(f)|² + 1/SNR²)

    W(f) tapers the division at frequencies where the instrument PSF has
    near-zero power, preventing amplification of noise.

    Parameters
    ----------
    wiener_snr : float
        Controls regularisation strength.  Higher → sharper deconvolution,
        but more noise amplification.  10–20 is typical.

    Returns
    -------
    psf_seeing : [K, K] float32  sum = 1.0
    """
    from numpy.fft import fft2, ifft2

    K = max(psf_total.shape[0], psf_instrument.shape[0])
    if K % 2 == 0:
        K += 1

    def _centre_pad(p: np.ndarray) -> np.ndarray:
        h, w = p.shape
        ph, pw = (K - h) // 2, (K - w) // 2
        return np.pad(p.astype(np.float64),
                      ((ph, K - h - ph), (pw, K - w - pw)))

    F_tot  = fft2(_centre_pad(psf_total))
    F_ins  = fft2(_centre_pad(psf_instrument))

    abs2   = np.abs(F_ins) ** 2
    wiener = abs2 / (abs2 + 1.0 / max(wiener_snr, 0.1) ** 2)

    F_see  = F_tot * wiener / (F_ins + 1e-14)
    raw    = np.real(ifft2(F_see))

    # FFT puts the centre at (0,0); roll to stamp centre
    raw = np.roll(np.roll(raw, K // 2, axis=0), K // 2, axis=1)
    raw = np.maximum(raw, 0.0)

    s = float(raw.sum())
    if s <= 0:
        return _gaussian_psf(psf_total.shape[0], fwhm_px=2.0)
    raw /= s

    # Crop to original size
    out = psf_total.shape[0]
    m   = (K - out) // 2
    crop = raw[m:m + out, m:m + out].astype(np.float32)
    cs   = float(crop.sum())
    if cs <= 0:
        return _gaussian_psf(out, fwhm_px=2.0)
    return (crop / cs).astype(np.float32)


# ============================================================================
# Aperture photometry helpers
# ============================================================================

def _aperture_fluxes(
    sub:          np.ndarray,     # [H, W] sky-subtracted float64
    positions:    np.ndarray,     # [N, 2] (y, x)
    aperture_r:   float,
) -> np.ndarray:
    """Return aperture flux for each star position (sky-annulus corrected)."""
    H, W  = sub.shape
    Y, X  = np.mgrid[0:H, 0:W].astype(np.float64)
    r_in  = aperture_r
    r_out = aperture_r * 2.5

    fluxes = np.zeros(len(positions), dtype=np.float64)
    for i, (y, x) in enumerate(positions):
        dist    = np.sqrt((Y - y) ** 2 + (X - x) ** 2)
        ap      = dist <= r_in
        sky_ann = (dist > r_in * 1.5) & (dist <= r_out)
        n_ap    = int(ap.sum())
        if n_ap == 0:
            continue
        sky_pp  = float(np.median(sub[sky_ann])) if sky_ann.sum() > 0 else 0.0
        fluxes[i] = float(sub[ap].sum()) - sky_pp * n_ap
    return fluxes


def estimate_transparency(
    positions:    np.ndarray,          # [N, 2]
    calibrated:   np.ndarray,          # [H, W]
    sky_bg:       np.ndarray,          # [H, W]
    ref_fluxes:   Optional[np.ndarray],
    aperture_r:   float = 8.0,
) -> float:
    """
    Estimate frame transparency from stellar aperture photometry.

    Compares aperture fluxes of detected stars to the same stars in the
    reference frame.  Returns median(flux_current / flux_reference).

    Returns 1.0 when ref_fluxes is None (first / reference frame) or when
    fewer than 2 valid star pairs are available.
    """
    if ref_fluxes is None or len(positions) == 0:
        return 1.0

    sub    = calibrated.astype(np.float64) - sky_bg.astype(np.float64)
    fluxes = _aperture_fluxes(sub, positions, aperture_r)

    n      = min(len(fluxes), len(ref_fluxes))
    cur    = fluxes[:n]
    ref    = ref_fluxes[:n]

    valid  = (cur > 0) & (ref > 0)
    if valid.sum() < 2:
        return 1.0

    ratios = cur[valid] / ref[valid]
    t      = float(np.clip(np.median(ratios), 0.01, 1.5))
    logger.debug("transparency: %.4f  from %d stars", t, int(valid.sum()))
    return t


# ============================================================================
# FrameCharacterizer
# ============================================================================

class FrameCharacterizer:
    """
    Characterise each calibrated light frame end-to-end.

    Instantiate once per imaging session, then call `characterize()` or
    `characterize_calibrated()` for each frame.  The first call should use
    ``is_reference=True`` to establish the WCS reference and reference
    stellar fluxes.

    Parameters
    ----------
    scope_geometry    : ScopeGeometry   telescope + camera parameters
    astap_solver      : ASTAPSolver     plate solver (None disables WCS shift)
    psf_size          : int             output PSF kernel side length (odd)
    stamp_size        : int             star cutout side length (>= psf_size, odd)
    snr_threshold     : float           minimum star peak SNR
    saturation_adu    : float           pixel level above which a star is saturated
    poly_degree       : int             sky background polynomial degree
    min_stars_for_psf : int             below this count, fall back to prior PSF
    wiener_snr        : float           Wiener deconvolution regularisation SNR
    aperture_r_px     : float           photometry aperture radius [pixels]
    """

    def __init__(
        self,
        scope_geometry:    "ScopeGeometry",
        astap_solver:      Optional["ASTAPSolver"] = None,
        psf_size:          int   = 31,
        stamp_size:        int   = 41,
        snr_threshold:     float = 20.0,
        saturation_adu:    float = 60_000.0,
        poly_degree:       int   = 2,
        min_stars_for_psf: int   = 5,
        wiener_snr:        float = 10.0,
        aperture_r_px:     float = 8.0,
        min_sep_px:        float = 20.0,
    ) -> None:
        # Force odd kernel sizes
        self.psf_size          = psf_size   | 1
        self.stamp_size        = stamp_size | 1
        self.snr_threshold     = snr_threshold
        self.saturation_adu    = saturation_adu
        self.poly_degree       = poly_degree
        self.min_stars_for_psf = min_stars_for_psf
        self.wiener_snr        = wiener_snr
        self.aperture_r_px     = aperture_r_px
        self.min_sep_px        = min_sep_px
        self.scope             = scope_geometry
        self.solver            = astap_solver

        # Cache instrument PSF (fixed for the session)
        if _OPTICS_OK:
            self._psf_instrument: np.ndarray = get_instrument_psf(
                scope_geometry, kernel_size=self.psf_size
            ).astype(np.float32)
        else:
            self._psf_instrument = _gaussian_psf(self.psf_size, fwhm_px=1.0)

        # Session state
        self._ref_wcs:       Optional[object]       = None  # WCSGeometry
        self._ref_fluxes:    Optional[np.ndarray]   = None
        self._ref_positions: Optional[np.ndarray]   = None
        self._prior_psf:     Optional[np.ndarray]   = None
        self._prior_fwhm_px: float                  = 3.0

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def characterize(
        self,
        fits_path:     str | Path,
        model,                          # InstrumentModel
        exposure_s:    float,
        is_reference:  bool = False,
    ) -> FrameMetadata:
        """
        Load a raw FITS light frame, calibrate it, and characterise it.

        Parameters
        ----------
        fits_path    : path to raw (uncalibrated) FITS file
        model        : fitted InstrumentModel from Phase 0
        exposure_s   : exposure time in seconds
        is_reference : set True for the session's first / best-seeing frame
        """
        fits_path = Path(fits_path)
        raw, header = self._load(fits_path)
        cal         = model.calibrate_frame(raw, exposure_s)
        return self.characterize_calibrated(
            cal, header, exposure_s,
            is_reference=is_reference,
            frame_path=fits_path,
        )

    def characterize_calibrated(
        self,
        calibrated:   np.ndarray,
        header:       fits.Header,
        exposure_s:   float,
        is_reference: bool          = False,
        frame_path:   Optional[Path] = None,
    ) -> FrameMetadata:
        """
        Characterise a frame that has already been calibrated.

        Use this variant when the calling code (e.g. sufficient_statistics.py)
        has already run ``model.calibrate_frame()``.

        Parameters
        ----------
        calibrated  : [H, W] float32 calibrated frame
        header      : original FITS header (for WCS extraction)
        exposure_s  : exposure time in seconds
        is_reference: True for the session reference frame
        frame_path  : optional path, used only for log messages
        """
        name = frame_path.name if frame_path else "<array>"
        logger.info("Characterizing %s  (exp=%.0f s)", name, exposure_s)

        # 1. Sky background
        sky_bg = estimate_sky_background(calibrated, poly_degree=self.poly_degree)

        # 2. Star extraction
        positions, stamps = extract_stars(
            calibrated, sky_bg,
            snr_threshold  = self.snr_threshold,
            saturation_adu = self.saturation_adu,
            stamp_size     = self.stamp_size,
            min_sep_px     = self.min_sep_px,
        )

        # 3. PSF estimation
        psf_total, fwhm_px, beta = self._build_psf(stamps)

        # 4. H_seeing by Wiener deconvolution
        psf_seeing = deconvolve_instrument_psf(
            psf_total, self._psf_instrument, wiener_snr=self.wiener_snr
        )

        fwhm_arcsec = fwhm_px * (
            self.scope.plate_scale_arcsec_per_px if _OPTICS_OK else 1.41
        )

        # 5. Transparency
        if is_reference:
            sub = calibrated.astype(np.float64) - sky_bg.astype(np.float64)
            self._ref_fluxes    = _aperture_fluxes(sub, positions, self.aperture_r_px)
            self._ref_positions = positions
            transparency        = 1.0
        else:
            transparency = estimate_transparency(
                positions, calibrated, sky_bg,
                ref_fluxes  = self._ref_fluxes,
                aperture_r  = self.aperture_r_px,
            )

        # 6. WCS shift
        wcs_geom, solve_status = self._extract_wcs(header)
        shift = self._compute_shift(wcs_geom, calibrated.shape, is_reference)

        # Update session state
        self._prior_psf     = psf_total
        self._prior_fwhm_px = fwhm_px
        if is_reference and wcs_geom is not None:
            self._ref_wcs = wcs_geom

        meta = FrameMetadata(
            shift        = shift,
            psf_total    = psf_total,
            psf_seeing   = psf_seeing,
            transparency = transparency,
            sky_bg       = sky_bg,
            fwhm_arcsec  = fwhm_arcsec,
            fwhm_pixels  = fwhm_px,
            n_stars_used = len(stamps),
            solve_status = solve_status,
            frame_path   = frame_path,
        )
        logger.info(
            "%s → t=%.3f  FWHM=%.2f\"  n_stars=%d  solve=%s",
            name, transparency, fwhm_arcsec, len(stamps), solve_status,
        )
        return meta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> Tuple[np.ndarray, fits.Header]:
        with fits.open(path, memmap=False) as hdul:
            data   = hdul[0].data.astype(np.float32).squeeze()
            header = hdul[0].header
        return data, header

    def _extract_wcs(
        self, header: fits.Header
    ) -> Tuple[Optional[object], str]:
        """Try to extract WCSGeometry from header; return (geom, status)."""
        if not _OPTICS_OK:
            return None, "failed"
        try:
            geom = extract_wcs_geometry(header)
            return geom, "wcs"
        except Exception:
            pass
        if self.solver is not None:
            # solver.solve() needs a file path — not available in array mode
            pass
        return None, "failed"

    def _compute_shift(
        self,
        wcs_geom:     Optional[object],
        frame_shape:  Tuple[int, int],
        is_reference: bool,
    ) -> Optional[object]:
        if is_reference:
            if wcs_geom is not None:
                self._ref_wcs = wcs_geom
            return None   # reference frame defines zero shift
        if wcs_geom is None or self._ref_wcs is None:
            return None
        if not _OPTICS_OK:
            return None
        try:
            return compute_frame_shift(wcs_geom, self._ref_wcs, frame_shape)
        except Exception as exc:
            logger.warning("Shift computation failed: %s", exc)
            return None

    def _build_psf(
        self, stamps: List[np.ndarray]
    ) -> Tuple[np.ndarray, float, float]:
        """Estimate PSF, falling back to prior or Gaussian when stamps are scarce."""
        if len(stamps) >= self.min_stars_for_psf:
            return estimate_psf_from_stamps(stamps, psf_size=self.psf_size)

        src = "prior" if self._prior_psf is not None else "Gaussian fallback"
        logger.warning(
            "Only %d stars (need %d) — using %s PSF",
            len(stamps), self.min_stars_for_psf, src,
        )
        if self._prior_psf is not None:
            return self._prior_psf, self._prior_fwhm_px, 2.5
        psf = _gaussian_psf(self.psf_size, fwhm_px=3.0)
        return psf, 3.0, 2.5
