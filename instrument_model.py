"""
InstrumentModel — fits and serializes per-pixel statistical models from
astrophotography calibration frames (bias, dark, flat, dark_flat).

All fitting is done in a single streaming pass with O(H×W) memory
regardless of the number of input frames.

Dependencies: astropy, numpy, h5py, psutil, scipy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import psutil
from astropy.io import fits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming accumulators
# ---------------------------------------------------------------------------

class WelfordAccumulator:
    """
    Online mean and variance via Welford's algorithm.
    Memory: 3 × [H, W] float64 arrays regardless of frame count.
    """

    def __init__(self) -> None:
        self.n: int = 0
        self._mean: Optional[np.ndarray] = None
        self._M2: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray) -> None:
        frame = frame.astype(np.float64)
        if self._mean is None:
            self._mean = np.zeros_like(frame)
            self._M2 = np.zeros_like(frame)
        self.n += 1
        delta = frame - self._mean
        self._mean += delta / self.n
        delta2 = frame - self._mean
        self._M2 += delta * delta2

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std). std is sample std (ddof=1)."""
        if self.n < 2:
            raise RuntimeError("Need at least 2 frames to compute variance.")
        std = np.sqrt(self._M2 / (self.n - 1))
        return self._mean.copy(), std

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()


class LinearRegressionAccumulator:
    """
    Online per-pixel linear regression: counts = slope * t + intercept.

    Uses the Welford-generalised centred formulation to avoid catastrophic
    cancellation.  The naive approach accumulates Σt², Σx², Σtx as raw sums
    and then subtracts large nearly-equal quantities in the slope formula —
    losing significant digits whenever the dark signal is small relative to
    the pixel values.  Here we instead track running means and *centred*
    cross/auto products so every accumulated quantity stays numerically small.

    State (all scalars or [H,W] float64):
        t_mean   — running mean of exposure times          (scalar)
        x_mean   — running per-pixel mean of counts        [H,W]
        Stt      — Σ (tᵢ - t̄)²                           (scalar)
        Stx      — Σ (tᵢ - t̄)(xᵢ - x̄)                  [H,W]
        Sxx      — Σ (xᵢ - x̄)²  per pixel (Welford M2)   [H,W]

    Memory: 4 × [H,W] + 2 scalars — same order as before.

    Update derivation (Welford step for bivariate data)
    ---------------------------------------------------
    After seeing the n-th sample (t_n, x_n):

        dt_old = t_n - t_mean_old          # deviation before mean update
        t_mean += dt_old / n
        dt_new = t_n - t_mean              # deviation after mean update

        dx_old = x_n - x_mean_old
        x_mean += dx_old / n
        dx_new = x_n - x_mean

        Stt += dt_old * dt_new             # equivalent to Σ(t - t̄)²
        Stx += dt_old * dx_new            # cross term — note asymmetry:
                                           #   dt uses old then new mean,
                                           #   dx uses *new* x_mean
        Sxx += dx_old * dx_new            # Welford M2 for x

    The asymmetry in Stx is intentional and matches the standard Welford
    bivariate proof (Knuth vol.2 §4.2.2).
    """

    def __init__(self) -> None:
        self.n: int = 0
        self._t_mean: float = 0.0          # scalar — mean exposure time
        self._x_mean: Optional[np.ndarray] = None   # [H,W]
        self._Stt: float = 0.0             # scalar — Σ(t - t̄)²
        self._Stx: Optional[np.ndarray] = None      # [H,W] — Σ(t-t̄)(x-x̄)
        self._Sxx: Optional[np.ndarray] = None      # [H,W] — Σ(x-x̄)²
        self._shape: Optional[Tuple] = None

    def update(self, frame: np.ndarray, exposure_time: float) -> None:
        frame = frame.astype(np.float64)
        t = float(exposure_time)

        if self._x_mean is None:
            self._shape = frame.shape
            self._x_mean = np.zeros(self._shape)
            self._Stx    = np.zeros(self._shape)
            self._Sxx    = np.zeros(self._shape)

        self.n += 1

        # --- exposure time update (scalar Welford) ---
        dt_old = t - self._t_mean
        self._t_mean += dt_old / self.n
        dt_new = t - self._t_mean          # deviation w.r.t. updated mean

        # --- pixel value update (vector Welford) ---
        dx_old = frame - self._x_mean
        self._x_mean += dx_old / self.n
        dx_new = frame - self._x_mean      # deviation w.r.t. updated mean

        # --- centred cross/auto products ---
        self._Stt += dt_old * dt_new       # scalar
        self._Stx += dt_old * dx_new       # [H,W]  — dt uses old/new, dx uses new
        self._Sxx += dx_old * dx_new       # [H,W]  — Welford M2 for x

    def finalize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (slope, intercept, residual_std).

        slope         — dark rate [ADU/sec],  shape [H,W]
        intercept     — bias residual [ADU],  shape [H,W]
        residual_std  — per-pixel scatter around the fit, shape [H,W]

        Degenerate case (all frames at identical exposure time → Stt == 0)
        returns slope=0, intercept=x_mean, residual_std from Sxx only.
        """
        if self.n < 2:
            raise RuntimeError("Need at least 2 frames for linear regression.")

        degenerate = abs(self._Stt) < 1e-12   # all exposures identical

        if degenerate:
            # Cannot determine slope — return zero slope, mean as intercept
            slope     = np.zeros(self._shape)
            intercept = self._x_mean.copy()
            dof = max(self.n - 1, 1)
            residual_std = np.sqrt(np.maximum(self._Sxx, 0.0) / dof)
            return slope, intercept, residual_std

        slope     = self._Stx / self._Stt                        # [H,W]
        intercept = self._x_mean - slope * self._t_mean          # [H,W]

        # Residual sum of squares via the identity:
        #   Sxx_res = Sxx - Stx² / Stt
        # This is numerically safe because both terms are already centred.
        ss_res = self._Sxx - (self._Stx ** 2) / self._Stt
        ss_res = np.maximum(ss_res, 0.0)   # clamp floating-point underflow
        dof = max(self.n - 2, 1)
        residual_std = np.sqrt(ss_res / dof)

        return slope, intercept, residual_std


# ---------------------------------------------------------------------------
# FITS helpers
# ---------------------------------------------------------------------------

# Standard IMAGETYP values used by most capture software
_FRAME_TYPE_KEYWORDS = {
    "bias":      {"bias", "bias frame"},
    "dark":      {"dark", "dark frame"},
    "flat":      {"flat", "flat field", "flat frame"},
    "dark_flat": {"dark flat", "darkflat"},
}

_SUBFOLDER_MAP = {
    "bias":      {"bias", "biases"},
    "dark":      {"dark", "darks"},
    "flat":      {"flat", "flats"},
    "dark_flat": {"dark_flat", "dark_flats", "darkflat", "darkflats"},
}


def _detect_frame_type_from_header(header: fits.Header) -> Optional[str]:
    raw = str(header.get("IMAGETYP", "")).strip().lower()
    for frame_type, keywords in _FRAME_TYPE_KEYWORDS.items():
        if raw in keywords:
            return frame_type
    return None


def _get_exposure_time(header: fits.Header) -> float:
    for kw in ("EXPTIME", "EXPOSURE"):
        if kw in header:
            return float(header[kw])
    raise KeyError("No exposure time keyword (EXPTIME/EXPOSURE) found in header.")


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


# ---------------------------------------------------------------------------
# Chunk iterator
# ---------------------------------------------------------------------------

def _auto_chunk_size(frame_shape: Tuple[int, int],
                     headroom_bytes: int = 2 * 1024 ** 3) -> int:
    """Estimate how many frames fit in available RAM leaving headroom."""
    available = psutil.virtual_memory().available - headroom_bytes
    frame_bytes = int(np.prod(frame_shape)) * 8  # float64
    return max(1, int(available / frame_bytes))


def _iter_chunks(paths: List[Path],
                 chunk_size: int) -> Iterator[List[Path]]:
    for i in range(0, len(paths), chunk_size):
        yield paths[i: i + chunk_size]


def _load_frame(path: Path) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        # Find the first ImageHDU with data
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                return hdu.data.astype(np.float64), hdu.header
    raise ValueError(f"No image data found in {path}")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@dataclass
class InstrumentModel:
    """
    Holds per-pixel statistical models of the instrument derived from
    calibration frames. All arrays are float32 and shaped (H, W).

    Usage
    -----
    model = InstrumentModel.fit_all("path/to/calibration/folder")
    model.save("instrument_model.h5")

    model = InstrumentModel.load("instrument_model.h5")
    """

    # --- Bias / read noise ---
    bias_mean: Optional[np.ndarray] = field(default=None, repr=False)
    read_noise: Optional[np.ndarray] = field(default=None, repr=False)

    # --- Dark ---
    dark_rate: Optional[np.ndarray] = field(default=None, repr=False)
    hot_pixel_mask: Optional[np.ndarray] = field(default=None, repr=False)

    # --- Flat ---
    flat_gain: Optional[np.ndarray] = field(default=None, repr=False)
    flat_uncertainty: Optional[np.ndarray] = field(default=None, repr=False)

    # --- Dark flat ---
    amp_glow_profile: Optional[np.ndarray] = field(default=None, repr=False)

    # --- Instrument metadata ---
    frame_shape: Optional[Tuple[int, int]] = None
    gain_setting: Optional[float] = None
    temperature: Optional[float] = None

    # --- Fit diagnostics ---
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_paths(folder: Path) -> dict[str, List[Path]]:
        """
        Discover calibration frames under `folder`.

        Tries two strategies:
        1. Subfolders named after frame types (bias/, darks/, flats/, dark_flats/)
        2. All FITS files in folder, classified by IMAGETYP header keyword
        """
        folder = Path(folder)
        result: dict[str, List[Path]] = {
            "bias": [], "dark": [], "flat": [], "dark_flat": []
        }

        # Strategy 1 — subfolders
        for frame_type, names in _SUBFOLDER_MAP.items():
            for name in names:
                sub = folder / name
                if sub.is_dir():
                    found = sorted(
                        p for p in sub.iterdir()
                        if p.suffix.lower() in (".fits", ".fit", ".fts")
                    )
                    result[frame_type].extend(found)

        # If subfolders found enough files, return early
        if any(result.values()):
            for k, v in result.items():
                logger.info("Found %d %s frames via subfolder discovery", len(v), k)
            return result

        # Strategy 2 — flat folder, classify by header
        logger.info("No subfolders found — classifying by IMAGETYP header")
        all_fits = sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in (".fits", ".fit", ".fts")
        )
        for path in all_fits:
            try:
                with fits.open(path, memmap=False) as hdul:
                    header = hdul[0].header
                frame_type = _detect_frame_type_from_header(header)
                if frame_type:
                    result[frame_type].append(path)
                else:
                    logger.debug("Could not classify %s (IMAGETYP=%s)",
                                 path.name, header.get("IMAGETYP", "missing"))
            except Exception as exc:
                logger.warning("Skipping %s: %s", path.name, exc)

        for k, v in result.items():
            logger.info("Found %d %s frames via header discovery", len(v), k)
        return result

    # ------------------------------------------------------------------
    # Shape and metadata bookkeeping
    # ------------------------------------------------------------------

    def _check_or_set_shape(self, shape: Tuple[int, int], source: str) -> None:
        if self.frame_shape is None:
            self.frame_shape = shape
        elif self.frame_shape != shape:
            raise ValueError(
                f"Shape mismatch in {source}: expected {self.frame_shape}, got {shape}"
            )

    def _update_instrument_meta(self, header: fits.Header) -> None:
        if self.gain_setting is None:
            self.gain_setting = _get_gain(header)
        if self.temperature is None:
            self.temperature = _get_temperature(header)

    # ------------------------------------------------------------------
    # Fitting — bias
    # ------------------------------------------------------------------

    def fit_bias(self,
                 paths: List[Path],
                 chunk_size: str | int = "auto") -> None:
        """
        Fit bias_mean and read_noise from bias frames.
        Uses Welford's online algorithm — O(H×W) memory.
        """
        if not paths:
            logger.warning("No bias frames provided — skipping.")
            return

        logger.info("Fitting bias model from %d frames", len(paths))
        acc = WelfordAccumulator()

        cs = self._resolve_chunk_size(chunk_size, paths)
        for chunk in _iter_chunks(paths, cs):
            for path in chunk:
                try:
                    frame, header = _load_frame(path)
                    self._check_or_set_shape(frame.shape[-2:], "bias")
                    self._update_instrument_meta(header)
                    # Use 2D slice if frame has extra dimensions
                    acc.update(frame.squeeze())
                except Exception as exc:
                    logger.warning("Skipping bias frame %s: %s", path.name, exc)

        mean, std = acc.finalize()
        self.bias_mean = mean.astype(np.float32)
        self.read_noise = std.astype(np.float32)
        self.metadata["n_bias_frames"] = acc.n
        self.metadata["bias_mean_global"] = float(np.median(self.bias_mean))
        self.metadata["read_noise_median"] = float(np.median(self.read_noise))
        logger.info("Bias fit complete. Median bias=%.1f  Median read noise=%.2f ADU",
                    self.metadata["bias_mean_global"],
                    self.metadata["read_noise_median"])

    # ------------------------------------------------------------------
    # Fitting — dark
    # ------------------------------------------------------------------

    def fit_dark(self,
                 paths: List[Path],
                 hot_pixel_sigma: float = 5.0,
                 chunk_size: str | int = "auto") -> None:
        """
        Fit dark_rate (ADU/sec) per pixel via online linear regression.
        Also derives hot_pixel_mask where dark_rate > hot_pixel_sigma * median.
        Bias is subtracted if bias_mean is available.
        """
        if not paths:
            logger.warning("No dark frames provided — skipping.")
            return

        logger.info("Fitting dark model from %d frames", len(paths))
        acc = LinearRegressionAccumulator()

        cs = self._resolve_chunk_size(chunk_size, paths)
        for chunk in _iter_chunks(paths, cs):
            for path in chunk:
                try:
                    frame, header = _load_frame(path)
                    self._check_or_set_shape(frame.shape[-2:], "dark")
                    self._update_instrument_meta(header)
                    frame = frame.squeeze()
                    exp_time = _get_exposure_time(header)
                    if self.bias_mean is not None:
                        frame = frame - self.bias_mean.astype(np.float64)
                    acc.update(frame, exp_time)
                except Exception as exc:
                    logger.warning("Skipping dark frame %s: %s", path.name, exc)

        slope, intercept, residual_std = acc.finalize()
        # Clamp negative rates — physically meaningless
        self.dark_rate = np.maximum(slope, 0.0).astype(np.float32)

        # Hot pixel mask — pixels with dark rate >> median
        median_rate = float(np.median(self.dark_rate))
        mad = float(np.median(np.abs(self.dark_rate - median_rate)))
        threshold = median_rate + hot_pixel_sigma * mad * 1.4826
        self.hot_pixel_mask = (self.dark_rate > threshold)

        self.metadata["n_dark_frames"] = acc.n
        self.metadata["dark_rate_median"] = median_rate
        self.metadata["hot_pixel_count"] = int(self.hot_pixel_mask.sum())
        logger.info(
            "Dark fit complete. Median rate=%.4f ADU/sec  Hot pixels=%d",
            median_rate, self.metadata["hot_pixel_count"]
        )

    # ------------------------------------------------------------------
    # Fitting — flat
    # ------------------------------------------------------------------

    def fit_flat(self,
                 paths: List[Path],
                 chunk_size: str | int = "auto") -> None:
        """
        Fit flat_gain (normalized throughput) and flat_uncertainty per pixel.

        Each frame is bias-subtracted (if available) then normalized by its
        own median before accumulation, removing lamp intensity variation.
        flat_uncertainty captures variance above the expected shot noise floor.
        """
        if not paths:
            logger.warning("No flat frames provided — skipping.")
            return

        logger.info("Fitting flat model from %d frames", len(paths))
        acc = WelfordAccumulator()
        # Also track mean of raw (un-normalized) counts for shot noise estimate
        raw_acc = WelfordAccumulator()

        cs = self._resolve_chunk_size(chunk_size, paths)
        for chunk in _iter_chunks(paths, cs):
            for path in chunk:
                try:
                    frame, header = _load_frame(path)
                    self._check_or_set_shape(frame.shape[-2:], "flat")
                    self._update_instrument_meta(header)
                    frame = frame.squeeze()
                    if self.bias_mean is not None:
                        frame = frame - self.bias_mean.astype(np.float64)
                    raw_acc.update(frame)
                    # Normalize by median to remove lamp variation
                    median_val = float(np.median(frame))
                    if median_val <= 0:
                        logger.warning("Flat frame %s has non-positive median — skipping",
                                       path.name)
                        continue
                    acc.update(frame / median_val)
                except Exception as exc:
                    logger.warning("Skipping flat frame %s: %s", path.name, exc)

        norm_mean, norm_std = acc.finalize()
        raw_mean, _ = raw_acc.finalize()

        # Normalize gain map so median = 1.0
        median_gain = float(np.median(norm_mean))
        self.flat_gain = (norm_mean / median_gain).astype(np.float32)

        # flat_uncertainty: observed std of normalized frames.
        # Expected shot noise contribution (normalized): sqrt(raw_mean / n) / raw_mean
        # = 1 / sqrt(raw_mean * n).  Residual above this is "true" flat uncertainty.
        n = acc.n
        shot_noise_norm = np.where(
            raw_mean > 0,
            1.0 / np.sqrt(np.maximum(raw_mean, 1.0) * n),
            norm_std
        )
        excess_var = np.maximum(norm_std ** 2 - shot_noise_norm ** 2, 0.0)
        self.flat_uncertainty = np.sqrt(excess_var).astype(np.float32)

        self.metadata["n_flat_frames"] = n
        self.metadata["flat_gain_min"] = float(self.flat_gain.min())
        self.metadata["flat_gain_max"] = float(self.flat_gain.max())
        self.metadata["flat_uncertainty_median"] = float(
            np.median(self.flat_uncertainty))
        logger.info(
            "Flat fit complete. Gain range=[%.3f, %.3f]  Median uncertainty=%.5f",
            self.metadata["flat_gain_min"],
            self.metadata["flat_gain_max"],
            self.metadata["flat_uncertainty_median"],
        )

    # ------------------------------------------------------------------
    # Fitting — dark flat
    # ------------------------------------------------------------------

    def fit_dark_flat(self,
                      paths: List[Path],
                      chunk_size: str | int = "auto") -> None:
        """
        Fit amp_glow_profile from dark flat frames.
        This is the residual spatial structure after subtracting a scaled
        dark frame — for ASI533 this will be ~zero.
        """
        if not paths:
            logger.warning("No dark flat frames provided — skipping.")
            return

        logger.info("Fitting dark flat model from %d frames", len(paths))
        acc = WelfordAccumulator()

        cs = self._resolve_chunk_size(chunk_size, paths)
        for chunk in _iter_chunks(paths, cs):
            for path in chunk:
                try:
                    frame, header = _load_frame(path)
                    self._check_or_set_shape(frame.shape[-2:], "dark_flat")
                    frame = frame.squeeze()
                    exp_time = _get_exposure_time(header)
                    # Subtract bias
                    if self.bias_mean is not None:
                        frame = frame - self.bias_mean.astype(np.float64)
                    # Subtract scaled dark
                    if self.dark_rate is not None:
                        frame = frame - self.dark_rate.astype(np.float64) * exp_time
                    acc.update(frame)
                except Exception as exc:
                    logger.warning("Skipping dark flat frame %s: %s", path.name, exc)

        mean, _ = acc.finalize()
        self.amp_glow_profile = mean.astype(np.float32)
        amp_glow_peak = float(np.max(np.abs(self.amp_glow_profile)))
        self.metadata["n_dark_flat_frames"] = acc.n
        self.metadata["amp_glow_peak_adu"] = amp_glow_peak
        logger.info("Dark flat fit complete. Amp glow peak=%.2f ADU", amp_glow_peak)

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    @classmethod
    def fit_all(cls,
                folder: str | Path,
                chunk_size: str | int = "auto",
                hot_pixel_sigma: float = 5.0) -> "InstrumentModel":
        """
        Discover and fit all calibration frame types found under `folder`.
        Returns a fully populated InstrumentModel.
        """
        folder = Path(folder)
        model = cls()
        model.metadata["fit_date"] = datetime.utcnow().isoformat()
        model.metadata["source_folder"] = str(folder)

        paths = cls._discover_paths(folder)

        # Order matters — bias first so subsequent fits can subtract it
        model.fit_bias(paths["bias"], chunk_size=chunk_size)
        model.fit_dark(paths["dark"],
                       hot_pixel_sigma=hot_pixel_sigma,
                       chunk_size=chunk_size)
        model.fit_flat(paths["flat"], chunk_size=chunk_size)
        model.fit_dark_flat(paths["dark_flat"], chunk_size=chunk_size)

        model.validate()
        return model

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Sanity-check fitted models and log warnings for suspicious values."""
        issues = []

        if self.read_noise is not None:
            high_rn = float(np.percentile(self.read_noise, 99))
            if high_rn > 50:
                issues.append(f"99th-percentile read noise is {high_rn:.1f} ADU — suspiciously high")

        if self.dark_rate is not None:
            n_hot = int(self.hot_pixel_mask.sum()) if self.hot_pixel_mask is not None else 0
            total = int(np.prod(self.frame_shape)) if self.frame_shape else 1
            frac = n_hot / total
            if frac > 0.01:
                issues.append(f"{frac*100:.2f}% hot pixels — consider lower sensor temperature")

        if self.flat_gain is not None:
            min_g = float(self.flat_gain.min())
            if min_g < 0.1:
                issues.append(f"Flat gain has values as low as {min_g:.3f} — severe vignetting or bad frames")

        if self.frame_shape is None:
            issues.append("frame_shape not set — no frames were successfully loaded")

        if issues:
            for issue in issues:
                logger.warning("Validation warning: %s", issue)
        else:
            logger.info("Validation passed — no issues detected")

    # ------------------------------------------------------------------
    # Calibration application
    # ------------------------------------------------------------------

    def calibrate_frame(self,
                        raw: np.ndarray,
                        exposure_time: float) -> np.ndarray:
        """
        Apply bias, dark, and flat corrections to a raw light frame.
        Returns calibrated frame as float32.
        raw: [H, W] array in ADU
        exposure_time: seconds
        """
        frame = raw.astype(np.float32)
        if self.bias_mean is not None:
            frame -= self.bias_mean
        if self.dark_rate is not None:
            frame -= self.dark_rate * exposure_time
        if self.flat_gain is not None:
            # Avoid division by zero at masked/dead pixels
            safe_gain = np.where(self.flat_gain > 0.01, self.flat_gain, 1.0)
            frame /= safe_gain
        return frame

    # ------------------------------------------------------------------
    # Compatibility check
    # ------------------------------------------------------------------

    def is_compatible(self,
                      other: "InstrumentModel",
                      temp_tolerance: float = 2.0) -> bool:
        """
        Check whether another model was acquired under compatible conditions.
        Useful before applying a previously saved model to a new session.
        """
        if self.frame_shape != other.frame_shape:
            logger.warning("Shape mismatch: %s vs %s", self.frame_shape, other.frame_shape)
            return False
        if (self.gain_setting is not None and other.gain_setting is not None
                and self.gain_setting != other.gain_setting):
            logger.warning("Gain mismatch: %s vs %s", self.gain_setting, other.gain_setting)
            return False
        if (self.temperature is not None and other.temperature is not None):
            delta = abs(self.temperature - other.temperature)
            if delta > temp_tolerance:
                logger.warning("Temperature difference %.1f°C exceeds tolerance %.1f°C",
                               delta, temp_tolerance)
                return False
        return True

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize model to HDF5."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        _arrays = {
            "bias/mean":        self.bias_mean,
            "bias/read_noise":  self.read_noise,
            "dark/rate":        self.dark_rate,
            "dark/hot_pixels":  self.hot_pixel_mask,
            "flat/gain":        self.flat_gain,
            "flat/uncertainty": self.flat_uncertainty,
            "dark_flat/amp_glow": self.amp_glow_profile,
        }

        with h5py.File(path, "w") as f:
            for key, arr in _arrays.items():
                if arr is not None:
                    f.create_dataset(key, data=arr, compression="gzip",
                                     compression_opts=4)

            meta = f.require_group("metadata")
            if self.frame_shape is not None:
                meta.attrs["frame_shape"] = list(self.frame_shape)
            if self.gain_setting is not None:
                meta.attrs["gain_setting"] = self.gain_setting
            if self.temperature is not None:
                meta.attrs["temperature"] = self.temperature
            for k, v in self.metadata.items():
                try:
                    meta.attrs[k] = v
                except TypeError:
                    meta.attrs[k] = str(v)

        logger.info("InstrumentModel saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "InstrumentModel":
        """Deserialize model from HDF5."""
        path = Path(path)
        model = cls()

        _field_map = {
            "bias/mean":          "bias_mean",
            "bias/read_noise":    "read_noise",
            "dark/rate":          "dark_rate",
            "dark/hot_pixels":    "hot_pixel_mask",
            "flat/gain":          "flat_gain",
            "flat/uncertainty":   "flat_uncertainty",
            "dark_flat/amp_glow": "amp_glow_profile",
        }

        with h5py.File(path, "r") as f:
            for hdf_key, attr in _field_map.items():
                if hdf_key in f:
                    setattr(model, attr, f[hdf_key][:])

            if "metadata" in f:
                meta = f["metadata"]
                shape = meta.attrs.get("frame_shape")
                if shape is not None:
                    model.frame_shape = tuple(int(x) for x in shape)
                gs = meta.attrs.get("gain_setting")
                if gs is not None:
                    model.gain_setting = float(gs)
                temp = meta.attrs.get("temperature")
                if temp is not None:
                    model.temperature = float(temp)
                for k in meta.attrs:
                    if k not in ("frame_shape", "gain_setting", "temperature"):
                        model.metadata[k] = meta.attrs[k]

        logger.info("InstrumentModel loaded from %s", path)
        return model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_chunk_size(self,
                            chunk_size: str | int,
                            paths: List[Path]) -> int:
        if chunk_size == "auto":
            if self.frame_shape is not None:
                cs = _auto_chunk_size(self.frame_shape)
            else:
                # Peek at first frame to get shape
                try:
                    frame, _ = _load_frame(paths[0])
                    cs = _auto_chunk_size(frame.shape[-2:])
                except Exception:
                    cs = 10  # safe fallback
            logger.debug("Auto chunk size: %d frames", cs)
            return cs
        return int(chunk_size)

    def summary(self) -> str:
        lines = ["InstrumentModel Summary", "=" * 40]
        lines.append(f"Frame shape    : {self.frame_shape}")
        lines.append(f"Gain setting   : {self.gain_setting}")
        lines.append(f"Temperature    : {self.temperature}°C")
        lines.append(f"Fit date       : {self.metadata.get('fit_date', 'unknown')}")
        lines.append("")
        if self.bias_mean is not None:
            lines.append(f"Bias frames    : {self.metadata.get('n_bias_frames', '?')}")
            lines.append(f"  Median bias  : {self.metadata.get('bias_mean_global', '?'):.1f} ADU")
            lines.append(f"  Median RN    : {self.metadata.get('read_noise_median', '?'):.2f} ADU")
        if self.dark_rate is not None:
            lines.append(f"Dark frames    : {self.metadata.get('n_dark_frames', '?')}")
            lines.append(f"  Median rate  : {self.metadata.get('dark_rate_median', '?'):.4f} ADU/s")
            lines.append(f"  Hot pixels   : {self.metadata.get('hot_pixel_count', '?')}")
        if self.flat_gain is not None:
            lines.append(f"Flat frames    : {self.metadata.get('n_flat_frames', '?')}")
            lines.append(f"  Gain range   : [{self.metadata.get('flat_gain_min', '?'):.3f}, "
                         f"{self.metadata.get('flat_gain_max', '?'):.3f}]")
        if self.amp_glow_profile is not None:
            lines.append(f"Dark flat frames: {self.metadata.get('n_dark_flat_frames', '?')}")
            lines.append(f"  Amp glow peak: {self.metadata.get('amp_glow_peak_adu', '?'):.2f} ADU")
        return "\n".join(lines)

    def __repr__(self) -> str:
        fitted = [k for k in ("bias_mean", "dark_rate", "flat_gain", "amp_glow_profile")
                  if getattr(self, k) is not None]
        return (f"InstrumentModel(shape={self.frame_shape}, "
                f"fitted={fitted}, "
                f"gain={self.gain_setting}, temp={self.temperature}°C)")
