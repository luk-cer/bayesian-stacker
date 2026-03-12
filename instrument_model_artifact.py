"""
instrument_model.py
===================
Fits and serializes per-pixel statistical models from astrophotography
calibration frames (bias, dark, flat, dark_flat).

Design goals
------------
- All fitting done in a single streaming pass:  O(H×W) memory regardless
  of frame count — frames are never all held in memory simultaneously.
- Numerically stable accumulators:
    WelfordAccumulator          — online mean + variance (Welford 1962)
    LinearRegressionAccumulator — online per-pixel linear regression using
                                  the Welford-generalised centred formulation
                                  to eliminate catastrophic cancellation.
- FITS I/O via astropy; frame-type discovery by subfolder name or IMAGETYP
  header keyword.
- HDF5 serialization (h5py) — single file, named arrays, human-inspectable.
- OSC (Bayer) aware flat normalisation — per-channel median to avoid
  cross-channel bias from the global median.

Calibration pipeline order (bias must be fitted first)
-------------------------------------------------------
    fit_bias  →  fit_dark  →  fit_flat  →  fit_dark_flat

Or simply call  InstrumentModel.fit_all(folder).

Bayesian cross-session usage
-----------------------------
Each fit_* method accepts an optional `bayes_state` (BayesCalibrationState).
When supplied the method uses conjugate-prior accumulators instead of Welford,
incorporating knowledge from all previous sessions.  The point-estimate output
arrays (bias_mean, read_noise, dark_rate, flat_gain) are format-identical
whether or not the Bayesian path is taken — the rest of the pipeline is
unchanged.

    # Session 1 — cold start from sensor spec priors
    from bayes_calibration import BayesCalibrationState, SensorPriors
    priors = SensorPriors.for_asi533_gain100()
    state  = BayesCalibrationState.from_priors(priors, shape=(3008, 3008))
    model  = InstrumentModel.fit_all("session_1/calibration/",
                                     bayes_state=state)
    model.save("instrument.h5")        # point estimates for calibrate_frame
    state.save("instrument.h5")        # full posteriors under /bayes/

    # Session 2 — posterior from session 1 becomes prior for session 2
    state2 = BayesCalibrationState.load("instrument.h5",
                                        new_session=True,
                                        new_temp_c=-11.0)   # Arrhenius-correct dark
    model2 = InstrumentModel.fit_all("session_2/calibration/",
                                     bayes_state=state2)
    model2.save("instrument.h5")
    state2.save("instrument.h5")

Without Bayesian state (original behaviour, no dependency on bayes_calibration):
    model = InstrumentModel.fit_all("session_1/calibration/")
    model.save("instrument.h5")

Dependencies
------------
    astropy  h5py  numpy  psutil
    bayes_calibration  (optional — only needed for Bayesian cross-session mode)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
import psutil
from astropy.io import fits

# Optional Bayesian prior integration — imported lazily to keep this module
# usable without bayes_calibration installed.
try:
    from bayes_calibration import (
        BayesCalibrationState,
        SensorPriors,
    )
    _BAYES_AVAILABLE = True
except ImportError:
    _BAYES_AVAILABLE = False
    BayesCalibrationState = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ============================================================================
# Streaming accumulators
# ============================================================================

class WelfordAccumulator:
    """
    Online mean and variance using Welford's algorithm (1962).

    Avoids the catastrophic cancellation that plagues the naive
    "accumulate Σx and Σx² then subtract" approach.  M2 accumulates
    products of two small deviations and therefore never grows large.

    Memory: 3 × [H, W] float64 arrays, independent of frame count.
    """

    def __init__(self) -> None:
        self.n: int = 0
        self._mean: Optional[np.ndarray] = None
        self._M2:   Optional[np.ndarray] = None   # Σ (xᵢ - x̄)²

    def update(self, frame: np.ndarray) -> None:
        frame = frame.astype(np.float64)
        if self._mean is None:
            self._mean = np.zeros_like(frame)
            self._M2   = np.zeros_like(frame)
        self.n += 1
        delta        = frame - self._mean          # deviation before update
        self._mean  += delta / self.n
        delta2       = frame - self._mean          # deviation after update
        self._M2    += delta * delta2              # always small × small

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, sample_std).  Requires n ≥ 2."""
        if self.n < 2:
            raise RuntimeError("WelfordAccumulator needs at least 2 frames.")
        std = np.sqrt(self._M2 / (self.n - 1))
        return self._mean.copy(), std

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()


class LinearRegressionAccumulator:
    """
    Online per-pixel linear regression:  counts = slope · t + intercept.

    Uses the Welford-generalised centred formulation to eliminate
    catastrophic cancellation.

    The naive formulation accumulates Σt², Σx², Σtx as raw sums and
    then computes:

        slope = (n·Σtx − Σt·Σx) / (n·Σt² − (Σt)²)

    When all exposure times are similar and pixel values are large, both
    numerator and denominator involve subtracting two nearly-equal large
    numbers, destroying most significant digits.

    The centred formulation tracks instead:

        t_mean  — running mean of t                       (scalar)
        x_mean  — running per-pixel mean of x             [H,W]
        Stt     — Σ (tᵢ − t̄)²                           (scalar)
        Stx     — Σ (tᵢ − t̄)(xᵢ − x̄)                  [H,W]
        Sxx     — Σ (xᵢ − x̄)²  (Welford M2 for x)       [H,W]

    Every accumulated value is a product of two *small* deviations —
    no large-number cancellation is possible.

    Welford bivariate update (n-th sample):
        dt_old  = t  − t_mean_old
        t_mean += dt_old / n
        dt_new  = t  − t_mean              ← after mean update
        dx_old  = x  − x_mean_old
        x_mean += dx_old / n
        dx_new  = x  − x_mean              ← after mean update
        Stt    += dt_old · dt_new
        Stx    += dt_old · dx_new          ← asymmetry intentional (Knuth §4.2.2)
        Sxx    += dx_old · dx_new

    Final estimates:
        slope        = Stx / Stt
        intercept    = x_mean − slope · t_mean
        ss_res       = Sxx − Stx² / Stt    ← numerically safe, both centred
        residual_std = √(ss_res / (n−2))

    Memory: 4 × [H,W] + 2 scalars.
    """

    def __init__(self) -> None:
        self.n: int        = 0
        self._t_mean: float = 0.0
        self._x_mean: Optional[np.ndarray] = None
        self._Stt: float   = 0.0
        self._Stx: Optional[np.ndarray] = None
        self._Sxx: Optional[np.ndarray] = None
        self._shape: Optional[Tuple]    = None

    def update(self, frame: np.ndarray, exposure_time: float) -> None:
        frame = frame.astype(np.float64)
        t     = float(exposure_time)

        if self._x_mean is None:
            self._shape  = frame.shape
            self._x_mean = np.zeros(self._shape)
            self._Stx    = np.zeros(self._shape)
            self._Sxx    = np.zeros(self._shape)

        self.n += 1

        # scalar Welford for t
        dt_old       = t - self._t_mean
        self._t_mean += dt_old / self.n
        dt_new       = t - self._t_mean

        # vector Welford for x
        dx_old       = frame - self._x_mean
        self._x_mean += dx_old / self.n
        dx_new       = frame - self._x_mean

        # centred cross / auto products
        self._Stt += dt_old * dt_new           # scalar
        self._Stx += dt_old * dx_new           # [H,W]
        self._Sxx += dx_old * dx_new           # [H,W]

    def finalize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (slope [ADU/s], intercept [ADU], residual_std [ADU]).

        Degenerate case (all frames at the same exposure time → Stt ≈ 0):
        returns slope = 0, intercept = x_mean, residual_std from Sxx.
        This is the correct answer — the data simply cannot constrain the
        slope when there is no variation in t.
        """
        if self.n < 2:
            raise RuntimeError(
                "LinearRegressionAccumulator needs at least 2 frames.")

        if abs(self._Stt) < 1e-12:
            # Degenerate: all exposures identical
            slope        = np.zeros(self._shape)
            intercept    = self._x_mean.copy()
            dof          = max(self.n - 1, 1)
            residual_std = np.sqrt(np.maximum(self._Sxx, 0.0) / dof)
            return slope, intercept, residual_std

        slope        = self._Stx / self._Stt
        intercept    = self._x_mean - slope * self._t_mean
        ss_res       = self._Sxx - (self._Stx ** 2) / self._Stt
        ss_res       = np.maximum(ss_res, 0.0)          # clamp underflow
        dof          = max(self.n - 2, 1)
        residual_std = np.sqrt(ss_res / dof)

        return slope, intercept, residual_std


# ============================================================================
# FITS helpers
# ============================================================================

_FRAME_TYPE_KEYWORDS: Dict[str, set] = {
    "bias":      {"bias", "bias frame"},
    "dark":      {"dark", "dark frame"},
    "flat":      {"flat", "flat field", "flat frame"},
    "dark_flat": {"dark flat", "darkflat"},
}

_SUBFOLDER_MAP: Dict[str, set] = {
    "bias":      {"bias", "biases"},
    "dark":      {"dark", "darks"},
    "flat":      {"flat", "flats"},
    "dark_flat": {"dark_flat", "dark_flats", "darkflat", "darkflats"},
}

# Bayer pattern channel offsets  {pattern_string: {channel: (row_offset, col_offset)}}
_BAYER_OFFSETS: Dict[str, Dict[str, Tuple[int,int]]] = {
    "RGGB": {"R": (0,0), "G0": (0,1), "G1": (1,0), "B": (1,1)},
    "BGGR": {"B": (0,0), "G0": (0,1), "G1": (1,0), "R": (1,1)},
    "GRBG": {"G0": (0,0), "R": (0,1), "B": (1,0), "G1": (1,1)},
    "GBRG": {"G0": (0,0), "B": (0,1), "R": (1,0), "G1": (1,1)},
}


def bayer_split(
    frame:   np.ndarray,
    pattern: str,
) -> "Tuple[np.ndarray, List[str]]":
    """
    Split a [H, W] Bayer mosaic into a [4, H//2, W//2] float32 array.

    Each of the 4 planes corresponds to one Bayer colour channel extracted
    at its native sub-pixel position.  For RGGB the order is R, G0, G1, B;
    other patterns are handled analogously via _BAYER_OFFSETS.

    Parameters
    ----------
    frame   : [H, W] 2-D array (any numeric dtype)
    pattern : Bayer pattern string, e.g. 'RGGB'

    Returns
    -------
    planes : float32 [4, H//2, W//2] — one plane per channel
    names  : list[str] of channel labels in the same order as planes
    """
    offsets = _BAYER_OFFSETS[pattern]
    planes, names = [], []
    for name, (r, c) in offsets.items():
        planes.append(frame[r::2, c::2].astype(np.float32))
        names.append(name)
    return np.stack(planes, axis=0), names


def _detect_frame_type(header: fits.Header) -> Optional[str]:
    raw = str(header.get("IMAGETYP", "")).strip().lower()
    for ftype, kws in _FRAME_TYPE_KEYWORDS.items():
        if raw in kws:
            return ftype
    return None


def _get_exposure_time(header: fits.Header) -> float:
    for kw in ("EXPTIME", "EXPOSURE"):
        if kw in header:
            return float(header[kw])
    raise KeyError("No EXPTIME/EXPOSURE keyword found in FITS header.")


def _get_temperature(header: fits.Header) -> Optional[float]:
    for kw in ("CCD-TEMP", "CCDTEMP", "TEMP"):
        if kw in header:
            return float(header[kw])
    return None


def _get_gain(header: fits.Header) -> Optional[float]:
    for kw in ("GAIN", "EGAIN"):
        if kw in header:
            return float(header[kw])
    return None


def _get_bayer_pattern(header: fits.Header) -> Optional[str]:
    for kw in ("BAYERPAT", "COLORTYP", "BAYOFFX"):
        val = header.get(kw)
        if val is not None:
            s = str(val).strip().upper()
            if s in _BAYER_OFFSETS:
                return s
    return None


def _load_frame(path: Path) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                return hdu.data.astype(np.float64), hdu.header
    raise ValueError(f"No 2-D image data found in {path}")


# ============================================================================
# Memory / chunk helpers
# ============================================================================

def _auto_chunk_size(frame_shape: Tuple[int, int],
                     headroom_bytes: int = 2 * 1024 ** 3) -> int:
    """How many float64 frames fit in available RAM less headroom."""
    available   = psutil.virtual_memory().available - headroom_bytes
    frame_bytes = int(np.prod(frame_shape)) * 8
    return max(1, int(available / frame_bytes))


def _iter_chunks(paths: List[Path],
                 chunk_size: int) -> Iterator[List[Path]]:
    for i in range(0, len(paths), chunk_size):
        yield paths[i : i + chunk_size]


# ============================================================================
# Main class
# ============================================================================

@dataclass
class InstrumentModel:
    """
    Per-pixel statistical characterisation of an astrophotography camera
    derived from calibration frames.

    All arrays are float32, shaped (H, W) unless noted.

    Quick start
    -----------
    >>> model = InstrumentModel.fit_all("/data/calibration/")
    >>> model.save("instrument.h5")
    >>> model = InstrumentModel.load("instrument.h5")
    >>> calibrated = model.calibrate_frame(raw_adu, exposure_time=120.0)
    """

    # ---- Bias / read noise ------------------------------------------------
    bias_mean:  Optional[np.ndarray] = field(default=None, repr=False)
    """Per-pixel bias pedestal [ADU]."""

    read_noise: Optional[np.ndarray] = field(default=None, repr=False)
    """Per-pixel read noise standard deviation [ADU]."""

    # ---- Dark current -----------------------------------------------------
    dark_rate:      Optional[np.ndarray] = field(default=None, repr=False)
    """Per-pixel dark current rate [ADU/sec], bias-subtracted."""

    hot_pixel_mask: Optional[np.ndarray] = field(default=None, repr=False)
    """Boolean mask — True where dark_rate exceeds threshold."""

    # ---- Flat field -------------------------------------------------------
    flat_gain:        Optional[np.ndarray] = field(default=None, repr=False)
    """Normalised per-pixel throughput (median = 1.0)."""

    flat_uncertainty: Optional[np.ndarray] = field(default=None, repr=False)
    """Per-pixel flat uncertainty above the shot-noise floor."""

    # ---- Dark flat / amp glow --------------------------------------------
    amp_glow_profile: Optional[np.ndarray] = field(default=None, repr=False)
    """Residual spatial structure in dark flats (amp glow).  ~0 for ASI533."""

    # ---- Instrument metadata ---------------------------------------------
    frame_shape:  Optional[Tuple[int, int]] = None
    gain_setting: Optional[float]           = None
    temperature:  Optional[float]           = None
    bayer_pattern: Optional[str]            = None   # None → mono

    # ---- Fit diagnostics -------------------------------------------------
    metadata: dict = field(default_factory=dict)

    # ==========================================================================
    # Frame-type discovery
    # ==========================================================================

    @staticmethod
    def _discover_paths(folder: Path) -> Dict[str, List[Path]]:
        """
        Return dict mapping frame-type → sorted list of FITS paths.

        Strategy 1: look for subfolders named bias/, darks/, flats/, etc.
        Strategy 2 (fallback): scan all FITS files and classify by IMAGETYP.
        """
        folder = Path(folder)
        result: Dict[str, List[Path]] = {
            k: [] for k in _SUBFOLDER_MAP
        }

        for ftype, names in _SUBFOLDER_MAP.items():
            for name in names:
                sub = folder / name
                if sub.is_dir():
                    result[ftype].extend(sorted(
                        p for p in sub.iterdir()
                        if p.suffix.lower() in (".fits", ".fit", ".fts")
                    ))

        if any(result.values()):
            for k, v in result.items():
                logger.info("Subfolder discovery: %d %s frames", len(v), k)
            return result

        logger.info("No subfolders found — classifying by IMAGETYP header")
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in (".fits", ".fit", ".fts"):
                continue
            try:
                with fits.open(path, memmap=False) as hdul:
                    header = hdul[0].header
                ftype = _detect_frame_type(header)
                if ftype:
                    result[ftype].append(path)
            except Exception as exc:
                logger.warning("Skipping %s: %s", path.name, exc)

        for k, v in result.items():
            logger.info("Header discovery: %d %s frames", len(v), k)
        return result

    # ==========================================================================
    # Internal bookkeeping
    # ==========================================================================

    def _check_or_set_shape(self, shape: Tuple[int, int], src: str) -> None:
        if self.frame_shape is None:
            self.frame_shape = shape
        elif self.frame_shape != shape:
            raise ValueError(
                f"Shape mismatch in {src}: expected {self.frame_shape}, got {shape}")

    def _update_meta_from_header(self, header: fits.Header) -> None:
        if self.gain_setting  is None: self.gain_setting  = _get_gain(header)
        if self.temperature   is None: self.temperature   = _get_temperature(header)
        if self.bayer_pattern is None: self.bayer_pattern = _get_bayer_pattern(header)

    def _resolve_chunk_size(self, chunk_size: str | int,
                             paths: List[Path]) -> int:
        if chunk_size != "auto":
            return int(chunk_size)
        if self.frame_shape is not None:
            cs = _auto_chunk_size(self.frame_shape)
        else:
            try:
                frame, _ = _load_frame(paths[0])
                cs = _auto_chunk_size(frame.shape[-2:])
            except Exception:
                cs = 10
        logger.debug("Auto chunk size: %d frames", cs)
        return cs

    # ==========================================================================
    # OSC helpers
    # ==========================================================================

    def _is_osc(self) -> bool:
        return self.bayer_pattern is not None

    def _bayer_channel_medians(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Return median value for each Bayer channel in a 2-D frame.
        Used to normalise flat frames per-channel instead of globally.
        """
        offsets = _BAYER_OFFSETS[self.bayer_pattern]
        medians: Dict[str, float] = {}
        for ch, (r0, c0) in offsets.items():
            medians[ch] = float(np.median(frame[r0::2, c0::2]))
        return medians

    def _bayer_normalise(self, frame: np.ndarray) -> np.ndarray:
        """
        Divide each Bayer channel by its own median so all channels sit
        at 1.0 on average.  Returns a copy.
        """
        out     = frame.copy()
        offsets = _BAYER_OFFSETS[self.bayer_pattern]
        medians = self._bayer_channel_medians(frame)
        for ch, (r0, c0) in offsets.items():
            m = medians[ch]
            if m > 0:
                out[r0::2, c0::2] /= m
        return out

    # ==========================================================================
    # fit_bias
    # ==========================================================================

    def fit_bias(self,
                 paths: List[Path],
                 chunk_size: str | int = "auto",
                 bayes_state: "Optional[BayesCalibrationState]" = None) -> None:
        """
        Estimate bias_mean and read_noise from bias frames.

        Two modes
        ---------
        bayes_state=None (default)
            Welford online mean + variance — the original point-estimate path.
            Produces the same output as before.

        bayes_state=<BayesCalibrationState>
            Uses BiasPriorAccumulator and ReadNoisePriorAccumulator instead.
            The conjugate posterior mean is used as the point estimate, so
            output arrays (bias_mean, read_noise) are identical in format but
            incorporate the Bayesian prior from the previous session.
            The updated accumulators are written back to bayes_state.
        """
        if not paths:
            logger.warning("fit_bias: no paths provided — skipping.")
            return

        logger.info("fit_bias: %d frames  (bayes=%s)", len(paths), bayes_state is not None)
        cs = self._resolve_chunk_size(chunk_size, paths)

        if bayes_state is not None and (bayes_state.bias_acc is not None
                                        or bayes_state.priors is not None):
            # --- Bayesian path ---
            bias_acc = None   # resolved after lazy init on the first frame
            rn_acc   = None
            rn_init  = None

            for chunk in _iter_chunks(paths, cs):
                for path in chunk:
                    try:
                        frame, header = _load_frame(path)
                        self._check_or_set_shape(frame.shape[-2:], "bias")
                        self._update_meta_from_header(header)
                        bayes_state.ensure_initialized(frame.shape[-2:])
                        if bias_acc is None:
                            bias_acc = bayes_state.bias_acc
                            rn_acc   = bayes_state.read_noise_acc
                            # Read noise unknown at session start; use prior mean
                            rn_init = (rn_acc.posterior_mean_std
                                       if rn_acc is not None
                                       else np.full(frame.shape[-2:], 6.0,
                                                    dtype=np.float64))
                        f = frame.squeeze().astype(np.float64)
                        bias_acc.update(f, rn_init)
                        if rn_acc is not None:
                            # Use current posterior mean as the "known" bias
                            rn_acc.update(f, bias_acc.posterior_mean)
                            # Refresh rn_init for next frame
                            rn_init = rn_acc.posterior_mean_std
                    except Exception as exc:
                        logger.warning("Skipping %s: %s", path.name, exc)

            # Extract point estimates from posteriors
            self.bias_mean  = bias_acc.posterior_mean.astype(np.float32)
            self.read_noise = (rn_acc.posterior_mean_std.astype(np.float32)
                               if rn_acc is not None
                               else np.full_like(self.bias_mean, 6.0))
            n = bias_acc.n
        else:
            # --- Original Welford path ---
            acc = WelfordAccumulator()
            for chunk in _iter_chunks(paths, cs):
                for path in chunk:
                    try:
                        frame, header = _load_frame(path)
                        self._check_or_set_shape(frame.shape[-2:], "bias")
                        self._update_meta_from_header(header)
                        acc.update(frame.squeeze())
                    except Exception as exc:
                        logger.warning("Skipping %s: %s", path.name, exc)
            mean, std       = acc.finalize()
            self.bias_mean  = mean.astype(np.float32)
            self.read_noise = std.astype(np.float32)
            n = acc.n

        self.metadata["n_bias_frames"]      = n
        self.metadata["bias_mean_global"]   = float(np.median(self.bias_mean))
        self.metadata["read_noise_median"]  = float(np.median(self.read_noise))
        self.metadata["bias_bayes"]         = bayes_state is not None
        logger.info("fit_bias done. median_bias=%.1f  median_RN=%.2f ADU",
                    self.metadata["bias_mean_global"],
                    self.metadata["read_noise_median"])

    # ==========================================================================
    # fit_dark
    # ==========================================================================

    def fit_dark(self,
                 paths: List[Path],
                 hot_pixel_sigma: float = 5.0,
                 chunk_size: str | int  = "auto",
                 bayes_state: "Optional[BayesCalibrationState]" = None) -> None:
        """
        Estimate dark_rate [ADU/sec] per pixel via online linear regression.

        Requires frames at ≥ 2 different exposure times for a valid slope.
        If all frames share the same exposure time, slope is set to zero
        (degenerate case — see LinearRegressionAccumulator).

        hot_pixel_mask is always derived from dark_rate via MAD threshold
        regardless of mode — it is a hard flag used downstream by validate()
        and calibrate_frame().  The DarkMixtureModel provides the soft
        probabilistic replacement; run fit_dark_mixture() after fit_dark.

        Two modes
        ---------
        bayes_state=None (default)
            LinearRegressionAccumulator slope fit — original path.

        bayes_state=<BayesCalibrationState>
            GammaPoissonAccumulator — incorporates cross-session prior.
            Posterior mean dark rate replaces the regression slope.
            The prior is temperature-corrected before this call via
            BayesCalibrationState.load(new_temp_c=...).
        """
        if not paths:
            logger.warning("fit_dark: no paths provided — skipping.")
            return

        logger.info("fit_dark: %d frames  (bayes=%s)", len(paths), bayes_state is not None)
        cs = self._resolve_chunk_size(chunk_size, paths)
        n  = 0

        if bayes_state is not None and (bayes_state.dark_acc is not None
                                        or bayes_state.priors is not None):
            # --- Bayesian path: Gamma-Poisson conjugate ---
            dk_acc = None   # resolved after lazy init on the first frame
            bias   = self.bias_mean  # may be None — handled inside update()

            for chunk in _iter_chunks(paths, cs):
                for path in chunk:
                    try:
                        frame, header = _load_frame(path)
                        self._check_or_set_shape(frame.shape[-2:], "dark")
                        self._update_meta_from_header(header)
                        bayes_state.ensure_initialized(frame.shape[-2:])
                        if dk_acc is None:
                            dk_acc = bayes_state.dark_acc
                        frame    = frame.squeeze().astype(np.float64)
                        exp_time = _get_exposure_time(header)
                        bias_arr = (bias.astype(np.float64)
                                    if bias is not None
                                    else np.zeros_like(frame))
                        dk_acc.update(frame, exp_time, bias_arr)
                        n += 1
                    except Exception as exc:
                        logger.warning("Skipping %s: %s", path.name, exc)

            dark_mean, _ = dk_acc.finalize()
            self.dark_rate = np.maximum(dark_mean, 0.0).astype(np.float32)
        else:
            # --- Original linear regression path ---
            acc = LinearRegressionAccumulator()
            for chunk in _iter_chunks(paths, cs):
                for path in chunk:
                    try:
                        frame, header = _load_frame(path)
                        self._check_or_set_shape(frame.shape[-2:], "dark")
                        self._update_meta_from_header(header)
                        frame    = frame.squeeze()
                        exp_time = _get_exposure_time(header)
                        if self.bias_mean is not None:
                            frame = frame - self.bias_mean.astype(np.float64)
                        acc.update(frame, exp_time)
                    except Exception as exc:
                        logger.warning("Skipping %s: %s", path.name, exc)
            slope, _intercept, _residual_std = acc.finalize()
            self.dark_rate = np.maximum(slope, 0.0).astype(np.float32)
            n = acc.n

        # Hot pixel mask — always from MAD threshold on dark_rate
        median_rate         = float(np.median(self.dark_rate))
        mad                 = float(np.median(np.abs(self.dark_rate - median_rate)))
        threshold           = median_rate + hot_pixel_sigma * 1.4826 * mad
        self.hot_pixel_mask = self.dark_rate > threshold

        self.metadata["n_dark_frames"]    = n
        self.metadata["dark_rate_median"] = median_rate
        self.metadata["hot_pixel_count"]  = int(self.hot_pixel_mask.sum())
        self.metadata["dark_bayes"]       = bayes_state is not None
        logger.info("fit_dark done. median_rate=%.4f ADU/s  hot_pixels=%d",
                    median_rate, self.metadata["hot_pixel_count"])

    # ==========================================================================
    # fit_flat
    # ==========================================================================

    def fit_flat(self,
                 paths: List[Path],
                 chunk_size: str | int = "auto",
                 bayes_state: "Optional[BayesCalibrationState]" = None) -> None:
        """
        Estimate flat_gain (normalised throughput map) and flat_uncertainty.

        For mono cameras: each frame is normalised by its global median.
        For OSC cameras:  each frame is normalised per Bayer channel to
                          avoid cross-channel bias (green pixels are 2×
                          more numerous than red or blue, so the global
                          median is pulled toward green).

        flat_uncertainty captures per-pixel variance in excess of the
        expected shot noise floor, indicating dust motion, QE instability,
        or illumination non-uniformity between sessions.

        Two modes
        ---------
        bayes_state=None (default)
            Welford streaming fit — original path.

        bayes_state=<BayesCalibrationState>
            FlatGainPriorAccumulator — Gaussian conjugate with Poisson
            shot noise likelihood.  The prior mean is always reset to 1.0
            at the start of each new session (flat gains are not reusable
            across sessions); the prior *variance* carries forward as a
            regulariser encoding the stable pixel-to-pixel gain structure.
            flat_uncertainty is still derived from the Welford residuals,
            since the Bayesian accumulator does not track excess variance.
        """
        if not paths:
            logger.warning("fit_flat: no paths provided — skipping.")
            return

        logger.info("fit_flat: %d frames  (bayes=%s)", len(paths), bayes_state is not None)
        raw_acc = WelfordAccumulator()   # always needed for flat_uncertainty
        cs      = self._resolve_chunk_size(chunk_size, paths)
        n       = 0

        read_noise_arr = (self.read_noise.astype(np.float64)
                          if self.read_noise is not None
                          else None)

        if bayes_state is not None and (bayes_state.flat_acc is not None
                                        or bayes_state.priors is not None):
            # --- Bayesian path ---
            fl_acc    = None   # resolved after lazy init on the first frame
            norm_acc  = WelfordAccumulator()   # track residuals for uncertainty

            for chunk in _iter_chunks(paths, cs):
                for path in chunk:
                    try:
                        frame, header = _load_frame(path)
                        self._check_or_set_shape(frame.shape[-2:], "flat")
                        self._update_meta_from_header(header)
                        bayes_state.ensure_initialized(frame.shape[-2:])
                        if fl_acc is None:
                            fl_acc = bayes_state.flat_acc
                        frame = frame.squeeze()
                        if self.bias_mean is not None:
                            frame = frame - self.bias_mean.astype(np.float64)
                        raw_acc.update(frame)

                        # Per-channel or global normalisation
                        if self._is_osc():
                            normed     = self._bayer_normalise(frame)
                            sky_median = float(np.median(frame))
                        else:
                            sky_median = float(np.median(frame))
                            if sky_median <= 0:
                                logger.warning(
                                    "Flat %s has non-positive median — skipping",
                                    path.name)
                                continue
                            normed = frame / sky_median

                        rn = (read_noise_arr
                              if read_noise_arr is not None
                              else np.full_like(normed, 6.0))
                        fl_acc.update(normed, sky_level=sky_median, read_noise=rn)
                        norm_acc.update(normed)
                        n += 1
                    except Exception as exc:
                        logger.warning("Skipping %s: %s", path.name, exc)

            gain_map    = fl_acc.posterior_mean.astype(np.float64)
            norm_mean   = gain_map
            norm_std    = np.sqrt(fl_acc.posterior_variance).astype(np.float64)
        else:
            # --- Original Welford path ---
            acc      = WelfordAccumulator()
            norm_std = None

            for chunk in _iter_chunks(paths, cs):
                for path in chunk:
                    try:
                        frame, header = _load_frame(path)
                        self._check_or_set_shape(frame.shape[-2:], "flat")
                        self._update_meta_from_header(header)
                        frame = frame.squeeze()
                        if self.bias_mean is not None:
                            frame = frame - self.bias_mean.astype(np.float64)
                        raw_acc.update(frame)
                        if self._is_osc():
                            normed = self._bayer_normalise(frame)
                        else:
                            median_val = float(np.median(frame))
                            if median_val <= 0:
                                logger.warning(
                                    "Flat %s has non-positive median — skipping",
                                    path.name)
                                continue
                            normed = frame / median_val
                        acc.update(normed)
                        n += 1
                    except Exception as exc:
                        logger.warning("Skipping %s: %s", path.name, exc)

            norm_mean, norm_std = acc.finalize()

        # Normalise so median of flat_gain = 1.0
        median_gain    = float(np.median(norm_mean))
        self.flat_gain = (norm_mean / median_gain).astype(np.float32)

        # flat_uncertainty — excess variance above shot noise floor
        if raw_acc.n >= 2:
            raw_mean, _ = raw_acc.finalize()
            shot_noise_norm = np.where(
                raw_mean > 0,
                1.0 / np.sqrt(np.maximum(raw_mean, 1.0) * n),
                norm_std if norm_std is not None else 0.0,
            )
            excess_var = np.maximum(
                (norm_std ** 2 if norm_std is not None else 0.0)
                - shot_noise_norm ** 2,
                0.0,
            )
            self.flat_uncertainty = np.sqrt(excess_var).astype(np.float32)
        else:
            self.flat_uncertainty = np.zeros_like(self.flat_gain)

        self.metadata["n_flat_frames"]           = n
        self.metadata["flat_gain_min"]           = float(self.flat_gain.min())
        self.metadata["flat_gain_max"]           = float(self.flat_gain.max())
        self.metadata["flat_uncertainty_median"] = float(
            np.median(self.flat_uncertainty))
        self.metadata["osc_flat_normalisation"]  = self._is_osc()
        self.metadata["flat_bayes"]              = bayes_state is not None
        logger.info(
            "fit_flat done. gain=[%.3f, %.3f]  uncertainty_median=%.5f  osc=%s",
            self.metadata["flat_gain_min"],
            self.metadata["flat_gain_max"],
            self.metadata["flat_uncertainty_median"],
            self._is_osc(),
        )

    # ==========================================================================
    # fit_dark_flat
    # ==========================================================================

    def fit_dark_flat(self,
                      paths: List[Path],
                      chunk_size: str | int = "auto") -> None:
        """
        Estimate amp_glow_profile from dark flat frames.

        Dark flats are taken with the same exposure as flats but with the
        sensor covered.  After subtracting bias and scaled dark, the residual
        spatial structure is the amp glow profile.

        For the ASI533 this should be essentially zero.
        """
        if not paths:
            logger.warning("fit_dark_flat: no paths provided — skipping.")
            return

        logger.info("fit_dark_flat: %d frames", len(paths))
        acc = WelfordAccumulator()
        cs  = self._resolve_chunk_size(chunk_size, paths)

        for chunk in _iter_chunks(paths, cs):
            for path in chunk:
                try:
                    frame, header = _load_frame(path)
                    self._check_or_set_shape(frame.shape[-2:], "dark_flat")
                    frame    = frame.squeeze()
                    exp_time = _get_exposure_time(header)
                    if self.bias_mean is not None:
                        frame = frame - self.bias_mean.astype(np.float64)
                    if self.dark_rate is not None:
                        frame = frame - self.dark_rate.astype(np.float64) * exp_time
                    acc.update(frame)
                except Exception as exc:
                    logger.warning("Skipping %s: %s", path.name, exc)

        mean, _                = acc.finalize()
        self.amp_glow_profile  = mean.astype(np.float32)
        peak                   = float(np.max(np.abs(self.amp_glow_profile)))

        self.metadata["n_dark_flat_frames"]   = acc.n
        self.metadata["amp_glow_peak_adu"]    = peak
        logger.info("fit_dark_flat done. amp_glow_peak=%.2f ADU", peak)

    # ==========================================================================
    # Top-level entry point
    # ==========================================================================

    @classmethod
    def fit_all(cls,
                folder:          str | Path,
                chunk_size:      str | int  = "auto",
                hot_pixel_sigma: float      = 5.0,
                bayes_state:     "Optional[BayesCalibrationState]" = None,
                ) -> "InstrumentModel":
        """
        Discover and fit all calibration frame types under `folder`.

        Fitting order: bias → dark → flat → dark_flat
        (bias must come first; subsequent steps subtract it).

        Parameters
        ----------
        folder : str | Path
            Root of the calibration tree.  Frames are discovered by subfolder
            name (bias/, dark/, flat/, dark_flat/) or FITS ``IMAGETYP`` header.
        chunk_size : int | "auto"
            Frames per memory chunk.  "auto" sizes from available RAM.
        hot_pixel_sigma : float
            MAD-based sigma threshold for the hard hot_pixel_mask.  Default 5.
        bayes_state : BayesCalibrationState, optional
            When provided, all four fit_* methods use conjugate-prior
            accumulators instead of Welford/regression.  The accumulated
            posteriors are written back into the same object so the caller
            can save them alongside the model::

                state = BayesCalibrationState.from_priors(priors, shape)
                model = InstrumentModel.fit_all(folder, bayes_state=state)
                model.save("instrument.h5")
                state.save("instrument.h5")   # /bayes/ group appended

        Returns
        -------
        InstrumentModel  with all point-estimate arrays populated.
        """
        folder = Path(folder)
        model  = cls()
        model.metadata["fit_date"]      = datetime.utcnow().isoformat()
        model.metadata["source_folder"] = str(folder)
        model.metadata["bayes_mode"]    = bayes_state is not None

        paths = cls._discover_paths(folder)
        model.fit_bias(
            paths["bias"],
            chunk_size=chunk_size,
            bayes_state=bayes_state,
        )
        model.fit_dark(
            paths["dark"],
            chunk_size=chunk_size,
            hot_pixel_sigma=hot_pixel_sigma,
            bayes_state=bayes_state,
        )
        model.fit_flat(
            paths["flat"],
            chunk_size=chunk_size,
            bayes_state=bayes_state,
        )
        model.fit_dark_flat(paths["dark_flat"], chunk_size=chunk_size)
        model.validate()
        return model

    # ==========================================================================
    # Validation
    # ==========================================================================

    def validate(self) -> None:
        """Log warnings for suspicious parameter values."""
        issues = []

        if self.frame_shape is None:
            issues.append("frame_shape not set — no frames loaded successfully")

        if self.read_noise is not None:
            p99 = float(np.percentile(self.read_noise, 99))
            if p99 > 50:
                issues.append(
                    f"99th-percentile read noise {p99:.1f} ADU is suspiciously high")

        if self.dark_rate is not None and self.hot_pixel_mask is not None:
            total = int(np.prod(self.frame_shape)) if self.frame_shape else 1
            frac  = self.hot_pixel_mask.sum() / total
            if frac > 0.01:
                issues.append(
                    f"{frac*100:.2f}% hot pixels — consider lowering sensor temperature")

        if self.flat_gain is not None:
            min_g = float(self.flat_gain.min())
            if min_g < 0.1:
                issues.append(
                    f"Flat gain has values as low as {min_g:.3f} — "
                    "severe vignetting or bad frames")

        if issues:
            for issue in issues:
                logger.warning("Validation: %s", issue)
        else:
            logger.info("Validation passed.")

    # ==========================================================================
    # Calibration application
    # ==========================================================================

    def calibrate_frame(self,
                        raw: np.ndarray,
                        exposure_time: float) -> np.ndarray:
        """
        Apply bias subtraction, dark subtraction, and flat division to one
        raw light frame.

        Parameters
        ----------
        raw           : [H, W] array of raw ADU counts (any numeric dtype)
        exposure_time : exposure duration in seconds

        Returns
        -------
        Calibrated frame as float32.  Dead / very-low-gain pixels are
        protected against division by zero (flat_gain < 0.01 → no division).
        """
        frame = raw.astype(np.float32)
        if self.bias_mean  is not None:
            frame -= self.bias_mean
        if self.dark_rate  is not None:
            frame -= self.dark_rate * float(exposure_time)
        if self.flat_gain  is not None:
            safe_gain = np.where(self.flat_gain > 0.01, self.flat_gain, 1.0)
            frame    /= safe_gain
        return frame

    # ==========================================================================
    # Compatibility check
    # ==========================================================================

    def is_compatible(self,
                      other: "InstrumentModel",
                      temp_tolerance: float = 2.0) -> bool:
        """
        Return True if `other` was acquired under conditions compatible with
        this model (shape, gain, temperature within tolerance).
        """
        ok = True
        if self.frame_shape != other.frame_shape:
            logger.warning("Shape mismatch: %s vs %s",
                           self.frame_shape, other.frame_shape)
            ok = False
        if (self.gain_setting is not None and other.gain_setting is not None
                and self.gain_setting != other.gain_setting):
            logger.warning("Gain mismatch: %s vs %s",
                           self.gain_setting, other.gain_setting)
            ok = False
        if (self.temperature is not None and other.temperature is not None
                and abs(self.temperature - other.temperature) > temp_tolerance):
            logger.warning("Temperature difference %.1f°C > tolerance %.1f°C",
                           abs(self.temperature - other.temperature),
                           temp_tolerance)
            ok = False
        return ok

    # ==========================================================================
    # Serialization
    # ==========================================================================

    def save(self, path: str | Path) -> None:
        """
        Serialize to HDF5.

        Layout
        ------
        /bias/mean              [H,W] float32
        /bias/read_noise        [H,W] float32
        /dark/rate              [H,W] float32
        /dark/hot_pixels        [H,W] bool
        /flat/gain              [H,W] float32
        /flat/uncertainty       [H,W] float32
        /dark_flat/amp_glow     [H,W] float32
        /metadata               (HDF5 group attributes)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {
            "bias/mean":           self.bias_mean,
            "bias/read_noise":     self.read_noise,
            "dark/rate":           self.dark_rate,
            "dark/hot_pixels":     self.hot_pixel_mask,
            "flat/gain":           self.flat_gain,
            "flat/uncertainty":    self.flat_uncertainty,
            "dark_flat/amp_glow":  self.amp_glow_profile,
        }

        with h5py.File(path, "w") as f:
            for key, arr in arrays.items():
                if arr is not None:
                    f.create_dataset(key, data=arr,
                                     compression="gzip", compression_opts=4)
            meta = f.require_group("metadata")
            if self.frame_shape   is not None:
                meta.attrs["frame_shape"]   = list(self.frame_shape)
            if self.gain_setting  is not None:
                meta.attrs["gain_setting"]  = self.gain_setting
            if self.temperature   is not None:
                meta.attrs["temperature"]   = self.temperature
            if self.bayer_pattern is not None:
                meta.attrs["bayer_pattern"] = self.bayer_pattern
            for k, v in self.metadata.items():
                try:
                    meta.attrs[k] = v
                except TypeError:
                    meta.attrs[k] = str(v)

        logger.info("InstrumentModel saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "InstrumentModel":
        """Deserialize from HDF5.  Inverse of save()."""
        path  = Path(path)
        model = cls()

        field_map = {
            "bias/mean":          "bias_mean",
            "bias/read_noise":    "read_noise",
            "dark/rate":          "dark_rate",
            "dark/hot_pixels":    "hot_pixel_mask",
            "flat/gain":          "flat_gain",
            "flat/uncertainty":   "flat_uncertainty",
            "dark_flat/amp_glow": "amp_glow_profile",
        }

        with h5py.File(path, "r") as f:
            for hdf_key, attr in field_map.items():
                if hdf_key in f:
                    setattr(model, attr, f[hdf_key][:])
            if "metadata" in f:
                meta = f["metadata"]
                shape = meta.attrs.get("frame_shape")
                if shape is not None:
                    model.frame_shape = tuple(int(x) for x in shape)
                for scalar_attr in ("gain_setting", "temperature"):
                    v = meta.attrs.get(scalar_attr)
                    if v is not None:
                        setattr(model, scalar_attr, float(v))
                bp = meta.attrs.get("bayer_pattern")
                if bp is not None:
                    model.bayer_pattern = str(bp)
                for k in meta.attrs:
                    if k not in ("frame_shape", "gain_setting",
                                 "temperature", "bayer_pattern"):
                        model.metadata[k] = meta.attrs[k]

        logger.info("InstrumentModel loaded ← %s", path)
        return model

    # ==========================================================================
    # Display helpers
    # ==========================================================================

    def summary(self) -> str:
        lines = ["InstrumentModel Summary", "=" * 42]
        lines.append(f"Frame shape    : {self.frame_shape}")
        lines.append(f"Camera type    : {'OSC (' + self.bayer_pattern + ')' if self._is_osc() else 'Mono'}")
        lines.append(f"Gain setting   : {self.gain_setting}")
        lines.append(f"Temperature    : {self.temperature}°C")
        lines.append(f"Fit date       : {self.metadata.get('fit_date', 'unknown')}")
        lines.append("")
        if self.bias_mean is not None:
            lines.append(f"Bias  ({self.metadata.get('n_bias_frames','?')} frames)")
            lines.append(f"  median bias   : {self.metadata.get('bias_mean_global','?'):.1f} ADU")
            lines.append(f"  median RN     : {self.metadata.get('read_noise_median','?'):.2f} ADU")
        if self.dark_rate is not None:
            lines.append(f"Dark  ({self.metadata.get('n_dark_frames','?')} frames)")
            lines.append(f"  median rate   : {self.metadata.get('dark_rate_median','?'):.4f} ADU/s")
            lines.append(f"  hot pixels    : {self.metadata.get('hot_pixel_count','?')}")
        if self.flat_gain is not None:
            lines.append(f"Flat  ({self.metadata.get('n_flat_frames','?')} frames,"
                         f" {'per-channel' if self.metadata.get('osc_flat_normalisation') else 'global'} norm)")
            lines.append(f"  gain range    : [{self.metadata.get('flat_gain_min','?'):.3f},"
                         f" {self.metadata.get('flat_gain_max','?'):.3f}]")
            lines.append(f"  uncert median : {self.metadata.get('flat_uncertainty_median','?'):.5f}")
        if self.amp_glow_profile is not None:
            lines.append(f"Dark flat ({self.metadata.get('n_dark_flat_frames','?')} frames)")
            lines.append(f"  amp glow peak : {self.metadata.get('amp_glow_peak_adu','?'):.2f} ADU")
        return "\n".join(lines)

    def __repr__(self) -> str:
        fitted = [k for k in ("bias_mean", "dark_rate", "flat_gain",
                               "amp_glow_profile")
                  if getattr(self, k) is not None]
        return (f"InstrumentModel(shape={self.frame_shape}, "
                f"camera={'OSC/' + self.bayer_pattern if self._is_osc() else 'Mono'}, "
                f"fitted={fitted})")
