"""
stretch.py
==========
Astrophotography display stretch functions.

Converts raw calibrated or stacked float32 arrays (linear, un-clipped,
high dynamic range) into [0, 1] display images suitable for PNG export
or browser rendering.

Key concepts
------------
Raw astro data has an extreme dynamic range: stars are often 1 000–100 000×
brighter than the sky background.  Linear normalisation makes the background
invisible and saturates bright stars.  All useful stretches are therefore
non-linear, applied after subtracting the sky background (black point).

The functions here follow the conventions of PixInsight's ScreenTransferFunction
(STF) and DeepSkyStacker's stretch algorithm, adapted for programmable use.

Stretch modes
-------------
  ``asinh``    Inverse-hyperbolic-sine stretch.  The standard for astro display.
               Compresses highlights smoothly, lifts faint detail.  Controlled by
               ``asinh_strength`` (higher = more aggressive stretch) and
               ``target_bg`` (desired mean output level for the background).

  ``linear``   Simple percentile clip + normalise.  Useful as a sanity check or
               for very bright / narrow dynamic range scenes.

  ``log``      Logarithmic stretch.  Stronger highlight compression than asinh but
               harsh on faint detail.  Good for emission nebulae with bright cores.

  ``midtone``  Midtone Transfer Function (MTF).  Pixar-style S-curve parameterised
               by shadows / midtones / highlights.  Most flexible manual control.

Linked vs unlinked colour
--------------------------
  **Linked** (``linked=True``, default):  One set of stretch parameters derived
  from the luminance channel (0.299R + 0.587G + 0.114B) and applied identically
  to all three channels.  Preserves colour balance; star colours are correct.

  **Unlinked** (``linked=False``):  Independent auto-stretch per channel.  Each
  channel fills the full [0, 1] range.  Useful for narrowband (Hα, OIII, SII)
  mapped to RGB, or to reveal faint colour gradients sacrificed by linked stretch.

Public API
----------
  StretchParams        — stretch configuration for one channel (or linked)
  auto_params()        — estimate StretchParams from data (all modes)
  stretch_mono()       — stretch a [H, W] array → [H, W] float32 in [0, 1]
  stretch_rgb()        — stretch a [H, W, 3] array (linked or unlinked)
  stretch_fits()       — load a FITS file and return a display-ready image
  debayer_preview()    — bilinear debayer [H, W] Bayer → [H, W, 3] for display
  to_uint8()           — convert [0, 1] float32 → uint8 for PNG export
  to_png_bytes()       — complete pipeline → PNG bytes (for Flask/HTTP response)

Dependencies
------------
  numpy   astropy (optional, for FITS loading in stretch_fits)
  PIL / Pillow  (for to_png_bytes)

Usage
-----
    from stretch import stretch_fits, StretchParams, auto_params

    # Fully automatic — asinh, linked colour
    img = stretch_fits("lambda_hr.fits")

    # Unlinked, stronger stretch
    from stretch import stretch_rgb, auto_params
    import numpy as np
    rgb  = np.stack([r, g, b], axis=-1)   # [H, W, 3] float32
    img  = stretch_rgb(rgb, linked=False, mode='asinh', asinh_strength=500)

    # Manual control
    params = StretchParams(black_point=120.0, white_point=8000.0, mode='midtone',
                           midtone=0.25)
    from stretch import stretch_mono
    disp = stretch_mono(data, params)

    # PNG for HTTP
    png = to_png_bytes(img)   # bytes ready for Flask send_file / Response
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------
try:
    from astropy.io import fits as _fits
    _ASTROPY_OK = True
except ImportError:
    _ASTROPY_OK = False

try:
    from PIL import Image as _PIL_Image
    _PIL_OK = True
except ImportError:
    _PIL_OK = False
    logger.warning("Pillow not installed — to_png_bytes() will be unavailable")


# ============================================================================
# Types
# ============================================================================

StretchMode = Literal["asinh", "linear", "log", "midtone"]
ArrayLike   = Union[np.ndarray]   # [H,W] or [H,W,3] float32


# ============================================================================
# StretchParams
# ============================================================================

@dataclass
class StretchParams:
    """
    Stretch configuration for one channel (or for linked colour).

    Attributes
    ----------
    black_point : float
        Input value mapped to display 0.  Pixels below this are clipped to 0.
        Usually estimated as ``sky_median + k * sky_MAD``.
    white_point : float
        Input value mapped to display 1 before the non-linear curve is applied.
        For ``linear`` mode this is the hard clip; for other modes it is used
        only to set the normalisation scale.
    mode : StretchMode
        Stretch algorithm.  One of ``'asinh'``, ``'linear'``, ``'log'``,
        ``'midtone'``.
    asinh_strength : float
        Asinh stretch factor β.  Larger = more aggressive highlight compression.
        Typical range 50–2000.  Default 500 matches PixInsight auto-STF.
    target_bg : float
        Desired mean display value for the sky background in [0, 1].
        Used by ``auto_params(mode='asinh')`` to find ``asinh_strength``
        automatically.  Default 0.18 (18% grey).
    midtone : float
        Midtone Transfer Function parameter m ∈ (0, 1).  Controls where the
        midpoint of the MTF curve sits in display space.  0.5 = linear;
        < 0.5 = boost shadows; > 0.5 = boost highlights.  Default 0.20.
    log_pedestal : float
        Small positive offset added before taking log to avoid log(0).
        Default 1.0 (in normalised [0, white-black] units).

    Notes
    -----
    All input values are in the raw ADU units of the input array, BEFORE
    black-point subtraction, UNLESS the array has already been sky-subtracted.
    ``black_point`` should then be set to 0.0 or a small positive value.
    """
    black_point:    float      = 0.0
    white_point:    float      = 65535.0
    mode:           StretchMode = "asinh"
    asinh_strength: float      = 500.0
    target_bg:      float      = 0.18
    midtone:        float      = 0.20
    log_pedestal:   float      = 1.0

    def copy(self) -> "StretchParams":
        import copy
        return copy.copy(self)

    def __repr__(self) -> str:  # noqa: D105
        return (f"StretchParams(mode={self.mode!r} "
                f"bp={self.black_point:.2f} wp={self.white_point:.2f} "
                f"asinh_strength={self.asinh_strength:.1f} "
                f"midtone={self.midtone:.3f})")


# ============================================================================
# Auto-parameter estimation
# ============================================================================

def _sky_stats(arr: np.ndarray, clip_sigma: float = 3.0, n_iter: int = 3
               ) -> Tuple[float, float]:
    """
    Robust sky background statistics via iterative sigma-clipping.

    Returns
    -------
    (median, MAD_sigma)  where MAD_sigma = 1.4826 * median(|x - median|)
    """
    data = arr.ravel().astype(np.float64)
    for _ in range(n_iter):
        med = float(np.median(data))
        mad = float(np.median(np.abs(data - med))) * 1.4826
        if mad < 1e-9:
            break
        data = data[np.abs(data - med) < clip_sigma * mad]
    med = float(np.median(data))
    mad = float(np.median(np.abs(data - med))) * 1.4826
    return med, max(mad, 1e-6)


def _find_asinh_strength(
    norm_bg:    float,   # normalised background level in [0, 1] after bp/wp
    target_bg:  float,   # desired display value for background
    beta_lo:    float = 1.0,
    beta_hi:    float = 1e6,
    tol:        float = 1e-4,
    max_iter:   int   = 60,
) -> float:
    """
    Binary-search for the asinh stretch factor β such that
    ``asinh(β * norm_bg) / asinh(β) == target_bg``.

    This is the PixInsight STF auto-stretch formula solved for β.
    """
    for _ in range(max_iter):
        beta = 0.5 * (beta_lo + beta_hi)
        denom = float(np.arcsinh(beta))
        if denom < 1e-12:
            break
        val = float(np.arcsinh(beta * norm_bg)) / denom
        if abs(val - target_bg) < tol:
            return beta
        if val < target_bg:   # f is increasing in β → need bigger β
            beta_lo = beta
        else:
            beta_hi = beta
    return 0.5 * (beta_lo + beta_hi)


def auto_params(
    arr:           np.ndarray,
    mode:          StretchMode = "asinh",
    target_bg:     float       = 0.18,
    asinh_strength: Optional[float] = None,
    bp_sigma:      float       = -2.8,
    wp_percentile: float       = 99.9,
    midtone:       Optional[float] = None,
    clip_sigma:    float       = 3.0,
) -> StretchParams:
    """
    Estimate ``StretchParams`` automatically from image data.

    Parameters
    ----------
    arr : [H, W] float32
        Single-channel input.  For colour images call once per channel when
        using unlinked mode, or on the luminance channel for linked mode.
    mode : StretchMode
        Which stretch algorithm to parameterise.
    target_bg : float
        Desired mean display value for the sky background ∈ (0, 0.5].
        Only used for ``'asinh'`` and ``'midtone'`` modes.
    asinh_strength : float | None
        Override automatic asinh factor search.  If None, computed from
        ``target_bg``.
    bp_sigma : float
        Black point = sky_median + bp_sigma * sky_MAD.  Default -2.8 places
        the black point 2.8σ BELOW the sky median, matching PixInsight STF.
        This means sky background pixels appear above zero in the display,
        which is required for the asinh auto-strength calculation to work.
        Use 0.0 to map the sky median exactly to zero (clip background to black).
        Positive values clip more aggressively — background disappears.
    wp_percentile : float
        White point set to this percentile of the input.  Default 99.9
        clips the very brightest saturated stars.
    midtone : float | None
        Override automatic MTF midtone.  If None, ``target_bg`` is used.
    clip_sigma : float
        Sigma-clipping threshold for sky_stats.

    Returns
    -------
    StretchParams
    """
    sky_med, sky_mad = _sky_stats(arr, clip_sigma=clip_sigma)

    # Black point: set BELOW the sky median so background pixels are visible.
    # bp_sigma should be negative (default -2.8 matches PixInsight STF default).
    # With bp below sky: normalised sky ≈ |bp_sigma|*mad / (wp-bp) — a small
    # positive number that the asinh binary search can target.
    black_point = sky_med + bp_sigma * sky_mad
    white_point = float(np.percentile(arr, wp_percentile))
    white_point = max(white_point, black_point + 1.0)
    scale       = white_point - black_point

    params = StretchParams(
        black_point    = black_point,
        white_point    = white_point,
        mode           = mode,
        target_bg      = target_bg,
        midtone        = midtone if midtone is not None else target_bg,
    )

    if mode == "asinh":
        if asinh_strength is not None:
            params.asinh_strength = asinh_strength
        else:
            # norm_bg: where does the sky MEDIAN fall in normalised [0,1] space?
            # = (sky_med - black_point) / (white_point - black_point)
            # With bp_sigma<0 this equals |bp_sigma|*mad/scale — a small positive.
            norm_bg = (sky_med - black_point) / scale
            norm_bg = float(np.clip(norm_bg, 1e-6, 0.5))
            params.asinh_strength = _find_asinh_strength(norm_bg, target_bg)
            logger.debug(
                "auto_params(asinh): bp=%.2f  wp=%.2f  β=%.1f  norm_bg=%.5f",
                black_point, white_point, params.asinh_strength, norm_bg,
            )

    elif mode == "midtone":
        params.midtone = target_bg if midtone is None else midtone

    return params


# ============================================================================
# Core stretch functions (single channel)
# ============================================================================

def _normalise(arr: np.ndarray, bp: float, wp: float) -> np.ndarray:
    """
    Clip to [bp, wp], subtract black point, normalise to [0, 1].
    Returns float64 array.
    """
    x = arr.astype(np.float64)
    x = np.clip(x, bp, wp)
    scale = wp - bp
    if scale < 1e-12:
        return np.zeros_like(x)
    return (x - bp) / scale


def _apply_asinh(x_norm: np.ndarray, beta: float) -> np.ndarray:
    """
    Asinh stretch: ``asinh(β·x) / asinh(β)``.

    Maps [0, 1] → [0, 1].  For β >> 1 this is nearly logarithmic near 0
    and linear near 1, giving the characteristic astro-stretch shape.
    """
    denom = float(np.arcsinh(beta))
    if denom < 1e-12:
        return x_norm.copy()
    return np.arcsinh(beta * x_norm) / denom


def _apply_linear(x_norm: np.ndarray) -> np.ndarray:
    """Identity on already-normalised data."""
    return x_norm.copy()


def _apply_log(x_norm: np.ndarray, pedestal: float = 1.0) -> np.ndarray:
    """
    Log stretch: ``log(1 + pedestal·x) / log(1 + pedestal)``.
    Maps [0, 1] → [0, 1].
    """
    denom = float(np.log1p(pedestal))
    if denom < 1e-12:
        return x_norm.copy()
    return np.log1p(pedestal * x_norm) / denom


def _apply_midtone(x_norm: np.ndarray, m: float) -> np.ndarray:
    """
    Midtone Transfer Function (MTF) as used in PixInsight / CCD tools.

    MTF(x; m) = (m - 1)·x / ((2m - 1)·x - m)

    Properties:
      MTF(0, m) = 0   MTF(1, m) = 1   MTF(m, m) = 0.5  (midtone maps to 50%)

    m < 0.5 → lift shadows   m > 0.5 → push shadows down   m = 0.5 → linear.

    Parameters
    ----------
    x_norm : already normalised to [0, 1]
    m      : midtone ∈ (0, 1), not 0 or 1 exactly
    """
    m = float(np.clip(m, 1e-6, 1.0 - 1e-6))
    denom = (2.0 * m - 1.0) * x_norm - m
    # Avoid division by zero (denominator = 0 when x = m/(2m-1), outside [0,1] for m<0.5)
    safe  = np.abs(denom) > 1e-12
    out   = np.where(safe, (m - 1.0) * x_norm / denom, 0.0)
    return np.clip(out, 0.0, 1.0)


def stretch_mono(arr: np.ndarray, params: StretchParams) -> np.ndarray:
    """
    Stretch a single-channel array to [0, 1] for display.

    Parameters
    ----------
    arr    : [H, W] float32 (or float64)  — raw or calibrated pixel values
    params : StretchParams

    Returns
    -------
    [H, W] float32 in [0, 1]
    """
    x = _normalise(arr, params.black_point, params.white_point)

    if params.mode == "asinh":
        x = _apply_asinh(x, params.asinh_strength)
    elif params.mode == "linear":
        x = _apply_linear(x)
    elif params.mode == "log":
        x = _apply_log(x, params.log_pedestal)
    elif params.mode == "midtone":
        x = _apply_midtone(x, params.midtone)
    else:
        raise ValueError(f"Unknown stretch mode: {params.mode!r}")

    return np.clip(x, 0.0, 1.0).astype(np.float32)


# ============================================================================
# RGB stretch  (linked and unlinked)
# ============================================================================

def stretch_rgb(
    arr:           np.ndarray,
    linked:        bool               = True,
    mode:          StretchMode        = "asinh",
    params_r:      Optional[StretchParams] = None,
    params_g:      Optional[StretchParams] = None,
    params_b:      Optional[StretchParams] = None,
    params_linked: Optional[StretchParams] = None,
    # Auto-stretch kwargs forwarded to auto_params()
    target_bg:     float = 0.18,
    asinh_strength: Optional[float] = None,
    bp_sigma:      float = -2.8,
    wp_percentile: float = 99.9,
    midtone:       Optional[float] = None,
) -> np.ndarray:
    """
    Stretch an RGB image to [0, 1].

    Parameters
    ----------
    arr : [H, W, 3] float32
        RGB image, channels in order R / G / B.
    linked : bool
        If True, one stretch is estimated from the luminance channel and
        applied identically to R, G, B.  Preserves colour balance.
        If False, each channel is stretched independently to fill [0, 1].
    mode : StretchMode
        Algorithm used when auto-estimating params.  Ignored if all explicit
        params are provided.
    params_r / params_g / params_b : StretchParams | None
        Explicit per-channel params for **unlinked** mode.  If None, estimated
        automatically from each channel.
    params_linked : StretchParams | None
        Explicit params for **linked** mode.  If None, estimated from luminance.
    target_bg, asinh_strength, bp_sigma, wp_percentile, midtone :
        Forwarded to ``auto_params()`` when params are not provided explicitly.

    Returns
    -------
    [H, W, 3] float32 in [0, 1]
    """
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected [H, W, 3] array, got shape {arr.shape}")

    auto_kw = dict(
        mode           = mode,
        target_bg      = target_bg,
        asinh_strength = asinh_strength,
        bp_sigma       = bp_sigma,
        wp_percentile  = wp_percentile,
        midtone        = midtone,
    )

    if linked:
        # Luminance = ITU-R BT.601 coefficients
        lum = (0.299 * arr[:, :, 0].astype(np.float64)
             + 0.587 * arr[:, :, 1].astype(np.float64)
             + 0.114 * arr[:, :, 2].astype(np.float64))
        p = params_linked if params_linked is not None else auto_params(lum, **auto_kw)
        out = np.stack([
            stretch_mono(arr[:, :, c], p) for c in range(3)
        ], axis=-1)

    else:
        channel_params = []
        for c, explicit in enumerate([params_r, params_g, params_b]):
            if explicit is not None:
                channel_params.append(explicit)
            else:
                channel_params.append(auto_params(arr[:, :, c], **auto_kw))
        out = np.stack([
            stretch_mono(arr[:, :, c], channel_params[c]) for c in range(3)
        ], axis=-1)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def get_channel_params(
    arr:          np.ndarray,
    linked:       bool        = True,
    mode:         StretchMode = "asinh",
    target_bg:    float       = 0.18,
    asinh_strength: Optional[float] = None,
    bp_sigma:     float       = -2.8,
    wp_percentile: float      = 99.9,
    midtone:      Optional[float] = None,
) -> dict:
    """
    Return the auto-estimated stretch parameters without applying the stretch.

    Useful for displaying current parameters in a UI before committing.

    Returns
    -------
    dict with keys ``'linked'`` (bool), ``'params'`` (StretchParams for linked),
    or ``'r'``, ``'g'``, ``'b'`` (StretchParams per channel for unlinked).
    """
    auto_kw = dict(
        mode=mode, target_bg=target_bg, asinh_strength=asinh_strength,
        bp_sigma=bp_sigma, wp_percentile=wp_percentile, midtone=midtone,
    )
    if arr.ndim == 2:
        return {"linked": True, "params": auto_params(arr, **auto_kw)}

    if linked:
        lum = (0.299 * arr[:, :, 0].astype(np.float64)
             + 0.587 * arr[:, :, 1].astype(np.float64)
             + 0.114 * arr[:, :, 2].astype(np.float64))
        return {"linked": True, "params": auto_params(lum, **auto_kw)}
    else:
        return {
            "linked": False,
            "r": auto_params(arr[:, :, 0], **auto_kw),
            "g": auto_params(arr[:, :, 1], **auto_kw),
            "b": auto_params(arr[:, :, 2], **auto_kw),
        }


# ============================================================================
# Debayer (OSC Bayer → RGB)
# ============================================================================

# Bayer pattern offset tables: maps pattern string → (R, G1, G2, B) pixel offsets
# Each offset is (row, col) of that colour in the 2×2 super-pixel.
_BAYER_OFFSETS = {
    "RGGB": {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)},
    "BGGR": {"R": (1, 1), "G1": (0, 1), "G2": (1, 0), "B": (0, 0)},
    "GRBG": {"R": (0, 1), "G1": (0, 0), "G2": (1, 1), "B": (1, 0)},
    "GBRG": {"R": (1, 0), "G1": (0, 0), "G2": (1, 1), "B": (0, 1)},
}


def debayer_preview(
    arr:     np.ndarray,
    pattern: str = "RGGB",
) -> np.ndarray:
    """
    Bilinear debayer a raw Bayer-pattern frame for preview display only.

    This is NOT a science-quality demosaic.  It produces a colour preview
    adequate for judging composition, focus, and stretch.  Use dedicated
    demosaic software (rawpy, LibRaw) for science processing.

    Algorithm
    ---------
    1. Extract each colour plane (R, G = mean(G1, G2), B) at half resolution
       by sampling the 2×2 super-pixel grid.
    2. Bilinear-upsample each plane back to full resolution.
    3. Stack → [H, W, 3] float32.

    Parameters
    ----------
    arr : [H, W] float32
        Raw sensor frame with Bayer pattern.  H and W must be even.
    pattern : str
        Bayer CFA pattern.  One of ``'RGGB'``, ``'BGGR'``, ``'GRBG'``, ``'GBRG'``.

    Returns
    -------
    [H, W, 3] float32  (R, G, B channels)
    """
    pattern = pattern.upper()
    if pattern not in _BAYER_OFFSETS:
        raise ValueError(f"Unknown Bayer pattern {pattern!r}.  "
                         f"Use one of {list(_BAYER_OFFSETS)}")
    H, W = arr.shape
    if H % 2 != 0 or W % 2 != 0:
        # Crop one row/col if odd
        arr = arr[:H - H%2, :W - W%2]
        H, W = arr.shape

    off  = _BAYER_OFFSETS[pattern]
    data = arr.astype(np.float32)

    # Sub-sample colour planes at half resolution
    r_half  = data[off["R" ][0]::2, off["R" ][1]::2]
    g1_half = data[off["G1"][0]::2, off["G1"][1]::2]
    g2_half = data[off["G2"][0]::2, off["G2"][1]::2]
    b_half  = data[off["B" ][0]::2, off["B" ][1]::2]
    g_half  = 0.5 * (g1_half + g2_half)

    # Bilinear upsample back to full resolution using numpy repeat + average
    def _upsample(plane: np.ndarray) -> np.ndarray:
        # Nearest-neighbour upsample × 2, then box-filter with 2×2 kernel
        u = np.repeat(np.repeat(plane, 2, axis=0), 2, axis=1).astype(np.float32)
        # Simple 2×2 box average for smoothing
        out = u.copy()
        out[:-1, :-1] = (u[:-1, :-1] + u[1:, :-1] + u[:-1, 1:] + u[1:, 1:]) * 0.25
        return out[:H, :W]

    r_full = _upsample(r_half)
    g_full = _upsample(g_half)
    b_full = _upsample(b_half)

    return np.stack([r_full, g_full, b_full], axis=-1)


# ============================================================================
# FITS → display image
# ============================================================================

def stretch_fits(
    path:          str,
    linked:        bool        = True,
    mode:          StretchMode = "asinh",
    debayer:       bool        = False,
    bayer_pattern: str         = "RGGB",
    hdu_index:     int         = 0,
    target_bg:     float       = 0.18,
    asinh_strength: Optional[float] = None,
    bp_sigma:      float       = -2.8,
    wp_percentile: float       = 99.9,
    midtone:       Optional[float] = None,
    params_r:      Optional[StretchParams] = None,
    params_g:      Optional[StretchParams] = None,
    params_b:      Optional[StretchParams] = None,
    params_linked: Optional[StretchParams] = None,
    max_size:      Optional[int] = None,
) -> np.ndarray:
    """
    Load a FITS file and return a display-ready [H, W] or [H, W, 3] float32
    array in [0, 1].

    Parameters
    ----------
    path : str
        Path to a FITS file.  The primary HDU (or ``hdu_index``) must contain
        a 2-D image.
    linked : bool
        Linked colour stretch (only used when ``debayer=True``).
    mode : StretchMode
        Stretch algorithm.
    debayer : bool
        If True, apply ``debayer_preview()`` before stretching.  Set for
        raw OSC frames.  False for calibrated mono / already-demosaiced.
    bayer_pattern : str
        Passed to ``debayer_preview()`` when ``debayer=True``.
    hdu_index : int
        Which HDU to read (default 0 = PRIMARY).
    target_bg, asinh_strength, bp_sigma, wp_percentile, midtone :
        Forwarded to ``auto_params()``.
    params_r, params_g, params_b, params_linked :
        Explicit stretch params; override auto-estimation.
    max_size : int | None
        If set, downsample the longest axis to this many pixels before
        stretching (for fast preview generation).

    Returns
    -------
    [H, W] float32  (mono) or [H, W, 3] float32  (colour)
    """
    if not _ASTROPY_OK:
        raise ImportError("astropy is required for stretch_fits()")

    with _fits.open(path, memmap=False) as hdul:
        data   = hdul[hdu_index].data.astype(np.float32).squeeze()
        header = hdul[hdu_index].header
        # Try to detect Bayer pattern from header if not forced
        if debayer and bayer_pattern == "RGGB":
            bp_hdr = header.get("BAYERPAT", header.get("COLORTYP", "RGGB"))
            bayer_pattern = str(bp_hdr).strip().upper()

    if data.ndim != 2:
        raise ValueError(f"FITS HDU {hdu_index} is {data.ndim}D; expected 2D image")

    # Optional downscale for preview speed
    if max_size is not None:
        H, W  = data.shape
        scale = max_size / max(H, W)
        if scale < 1.0:
            from scipy.ndimage import zoom
            data = zoom(data, scale, order=1).astype(np.float32)

    auto_kw = dict(
        mode=mode, target_bg=target_bg, asinh_strength=asinh_strength,
        bp_sigma=bp_sigma, wp_percentile=wp_percentile, midtone=midtone,
    )

    if debayer:
        rgb = debayer_preview(data, pattern=bayer_pattern)
        return stretch_rgb(rgb, linked=linked,
                           params_r=params_r, params_g=params_g, params_b=params_b,
                           params_linked=params_linked, **auto_kw)
    else:
        if params_linked is not None:
            return stretch_mono(data, params_linked)
        p = auto_params(data, **auto_kw)
        return stretch_mono(data, p)


# ============================================================================
# Output helpers
# ============================================================================

def to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Convert a [0, 1] float32 display image to uint8 [0, 255].

    Parameters
    ----------
    arr : [H, W] or [H, W, 3] float32 in [0, 1]

    Returns
    -------
    [H, W] or [H, W, 3] uint8
    """
    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_png_bytes(
    arr:     np.ndarray,
    quality: int = 95,
) -> bytes:
    """
    Convert a display image to PNG bytes suitable for HTTP response or file write.

    Parameters
    ----------
    arr : [H, W] or [H, W, 3] float32 in [0, 1]
        Display-ready image from ``stretch_mono`` / ``stretch_rgb`` / ``stretch_fits``.
    quality : int
        Ignored for PNG (lossless); kept for API consistency if JPEG is added.

    Returns
    -------
    bytes — PNG-encoded image
    """
    if not _PIL_OK:
        raise ImportError("Pillow is required for to_png_bytes().  "
                          "Install with: pip install Pillow")
    u8  = to_uint8(arr)
    if u8.ndim == 2:
        img = _PIL_Image.fromarray(u8, mode="L")
    elif u8.ndim == 3:
        img = _PIL_Image.fromarray(u8, mode="RGB")
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=3)
    return buf.getvalue()


def save_png(arr: np.ndarray, path: str) -> None:
    """
    Save a display image directly to a PNG file.

    Parameters
    ----------
    arr  : [H, W] or [H, W, 3] float32 in [0, 1]
    path : output file path (should end in .png)
    """
    if not _PIL_OK:
        raise ImportError("Pillow is required for save_png()")
    u8  = to_uint8(arr)
    if u8.ndim == 2:
        img = _PIL_Image.fromarray(u8, mode="L")
    else:
        img = _PIL_Image.fromarray(u8, mode="RGB")
    img.save(path, format="PNG", compress_level=3)
    logger.info("Saved PNG → %s", path)


# ============================================================================
# Histogram helper (useful for UI display)
# ============================================================================

def histogram(
    arr:    np.ndarray,
    n_bins: int   = 256,
    log:    bool  = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a display histogram of a [0, 1] stretched image.

    Parameters
    ----------
    arr    : [H, W] or [H, W, 3] float32 in [0, 1]
    n_bins : number of histogram bins
    log    : if True, take log10 of counts (for display)

    Returns
    -------
    (bin_centres, counts)  both float32 [n_bins]
    For RGB input, counts are summed across all channels.
    """
    data   = arr.ravel().astype(np.float32)
    counts, edges = np.histogram(data, bins=n_bins, range=(0.0, 1.0))
    centres = 0.5 * (edges[:-1] + edges[1:])
    if log:
        counts = np.log10(counts.astype(np.float64) + 1.0).astype(np.float32)
    return centres.astype(np.float32), counts.astype(np.float32)


def rgb_histograms(
    arr:    np.ndarray,
    n_bins: int  = 256,
    log:    bool = False,
) -> dict:
    """
    Compute per-channel histograms for an RGB [H,W,3] array.

    Returns
    -------
    dict with keys ``'r'``, ``'g'``, ``'b'``, ``'lum'``, each a
    tuple of (bin_centres [n_bins], counts [n_bins]).
    """
    if arr.ndim == 2:
        centres, counts = histogram(arr, n_bins=n_bins, log=log)
        return {"lum": (centres, counts)}

    lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
    return {
        "r":   histogram(arr[:, :, 0], n_bins, log),
        "g":   histogram(arr[:, :, 1], n_bins, log),
        "b":   histogram(arr[:, :, 2], n_bins, log),
        "lum": histogram(lum,           n_bins, log),
    }
