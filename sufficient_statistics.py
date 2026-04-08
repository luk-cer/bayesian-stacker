"""
sufficient_statistics.py
========================
Streaming accumulation of the sufficient statistics needed by the Phase 4
MAP super-resolution stacker.

The stacker solves for a 2× super-resolved scene λ on a [2H × 2W] grid by
minimising:

    Loss = Σ_i Poisson_NLL(x_i | t_i · (H_i ⊛ W_i[λ])) + α·TV(λ) + β·KL(λ‖prior)

where
    x_i   — calibrated light frame i, sky-subtracted   [H, W]
    t_i   — per-frame transparency
    H_i   — per-frame total PSF kernel
    W_i   — sub-pixel shift operator (phase-shift in Fourier space)

Before the GPU optimisation can run, every frame must be loaded, calibrated,
sky-subtracted, and characterised.  This module does that work in a single
streaming pass that fits in limited RAM regardless of how many frames exist.

What is accumulated
-------------------
Per-frame quantities stored for the MAP solver:
    shift_list         list of FrameShift | None, one per frame
    psf_list           list of [K, K] float32 PSF kernels
    transparency_list  list of float

Pixel-grid summaries (the actual "sufficient statistics"):
    weighted_sum[H, W]     Σ_i  t_i · x_i^(p)      (sky-subtracted)
    weight_sum[H, W]       Σ_i  t_i
    sky_sum[H, W]          Σ_i  sky_i^(p)           (for mean sky model)
    sq_sum[H, W]           Σ_i  t_i · x_i^(p)²      (for variance estimate)
    frame_count            N total frames accumulated

Fast stack (no GPU needed)
--------------------------
    weighted_mean = weighted_sum / weight_sum

This is the Gamma-Poisson posterior mean — a transparency-weighted mean
stack.  It is a complete, useful result on its own for visual inspection
before the MAP optimisation.  quality_map gives per-pixel reliability.

Memory model
------------
The accumulator holds only six [H, W] float64 arrays in memory at any time
plus the per-frame metadata lists.  Individual frames are loaded, processed,
and discarded.  For a 3008×3008 sensor the six arrays require ~420 MB.

HDF5 persistence
----------------
Call save() after every N frames for crash recovery.  Call load() to resume
a partially accumulated run.  The on-disk layout mirrors the in-memory
attributes; per-frame lists are stored as ragged HDF5 groups.

Usage
-----
    from sufficient_statistics import SufficientStatsAccumulator, SufficientStats
    from frame_characterizer import FrameCharacterizer
    from instrument_model_artifact import InstrumentModel
    from optics import ScopeGeometry

    model = InstrumentModel.load("instrument.h5")
    scope = ScopeGeometry(aperture_mm=100, focal_length_mm=550, pixel_size_um=3.76)
    fc    = FrameCharacterizer(scope)
    acc   = SufficientStatsAccumulator(frame_shape=(3008, 3008))

    for i, path in enumerate(light_paths):
        acc.add_frame(path, model, fc, exposure_s=300.,
                      is_reference=(i == 0))
        if (i + 1) % 20 == 0:
            acc.save("stats_checkpoint.h5")

    stats = acc.finalize()
    print(stats.summary())
    stats.save("sufficient_stats.h5")

    # Fast preview stack
    fast_stack = stats.weighted_mean   # [H, W] float32

Dependencies
------------
    numpy  h5py  astropy  frame_characterizer  instrument_model_artifact
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
from astropy.io import fits
from scipy.ndimage import affine_transform as _affine_transform

logger = logging.getLogger(__name__)


def _apply_phase_shift(arr: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Shift a 2-D array by (dx, dy) pixels using the Fourier phase-shift theorem.
    Used only for the sub-pixel residual after affine resampling.
    """
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return arr
    H, W  = arr.shape
    fy    = np.fft.fftfreq(H).reshape(-1, 1)
    fx    = np.fft.rfftfreq(W).reshape(1, -1)
    phase = np.exp(-2j * np.pi * (fy * dy + fx * dx))
    return np.real(np.fft.irfft2(np.fft.rfft2(arr) * phase, s=(H, W)))


def _warp_to_canvas(
    arr:           np.ndarray,
    frame_wcs,                   # WCSGeometry of this frame
    ref_wcs,                     # WCSGeometry of the reference frame
    canvas_shape:  tuple,        # (cH, cW) output canvas size
    sensor_origin: tuple,        # (r0, c0) reference frame origin in canvas
    osc_scale:     int = 1,      # 2 for OSC (half-res), 1 for mono
) -> np.ndarray:
    """
    Resample `arr` (a single 2-D channel) onto the union canvas using a full
    affine transform derived from the WCS solutions of the frame and the
    reference.  This correctly handles rotation, sub-pixel translation, and
    small scale differences in a single interpolation pass.

    The mapping is: for each canvas pixel (r_c, c_c) find the corresponding
    frame pixel (r_f, c_f) via WCS sky projection, then interpolate.

    Parameters
    ----------
    arr          : [H, W] float64 frame channel data
    frame_wcs    : WCSGeometry (plate-solved) for this frame
    ref_wcs      : WCSGeometry for the reference frame
    canvas_shape : (cH, cW) — size of the output canvas
    sensor_origin: (r0, c0) — where the reference frame top-left sits
    osc_scale    : 1 for mono, 2 for OSC (pixel coords are half-res)
    """
    cH, cW   = canvas_shape
    r0, c0   = sensor_origin
    fH, fW   = arr.shape

    # Build the affine matrix mapping canvas pixels → frame pixels.
    # We use a 3-point WCS round-trip to derive the linear map:
    #   canvas_px → sky (via reference WCS) → frame_px (via frame WCS inverse)
    # Sample at canvas centre and two offset points to get a 2×2 Jacobian + offset.

    cc_r = (cH - 1) / 2.0
    cc_c = (cW - 1) / 2.0
    delta = min(cH, cW) / 4.0

    def canvas_to_frame(cr, cc):
        """Canvas (row,col) → frame (row,col) via WCS sky roundtrip."""
        # Canvas pixel → reference pixel (subtract sensor origin)
        ref_col = (cc - c0) * osc_scale
        ref_row = (cr - r0) * osc_scale
        # Reference pixel → sky (astropy uses col,row = x,y order)
        sky = ref_wcs.wcs.pixel_to_world(ref_col, ref_row)
        # Sky → frame pixel
        fx, fy = frame_wcs.wcs.world_to_pixel(sky)
        # Frame pixel → frame array index (divide by osc_scale for half-res)
        return float(fy) / osc_scale, float(fx) / osc_scale

    p0 = canvas_to_frame(cc_r,         cc_c)
    px = canvas_to_frame(cc_r,         cc_c + delta)
    py = canvas_to_frame(cc_r + delta, cc_c)

    # Jacobian columns: d(frame)/d(canvas_col) and d(frame)/d(canvas_row)
    dcol = ((px[0]-p0[0])/delta, (px[1]-p0[1])/delta)
    drow = ((py[0]-p0[0])/delta, (py[1]-p0[1])/delta)

    # Affine matrix A [2×2]: maps canvas offset → frame offset
    A = np.array([[drow[0], dcol[0]],
                  [drow[1], dcol[1]]])

    # Offset: frame_centre - A @ canvas_centre
    offset = np.array([p0[0] - A[0,0]*cc_r - A[0,1]*cc_c,
                       p0[1] - A[1,0]*cc_r - A[1,1]*cc_c])

    out = _affine_transform(
        arr, A, offset=offset,
        output_shape=(cH, cW),
        order=3,          # bicubic — best quality for astronomical images
        mode='constant',
        cval=0.0,
    )
    return out


# ---------------------------------------------------------------------------
# Optional project imports
# ---------------------------------------------------------------------------
try:
    from frame_characterizer import FrameCharacterizer, FrameMetadata
    _FC_OK = True
except ImportError:
    _FC_OK = False
    FrameCharacterizer = None    # type: ignore
    FrameMetadata      = None    # type: ignore

try:
    from instrument_model_artifact import InstrumentModel
    _IM_OK = True
except ImportError:
    try:
        from instrument_model import InstrumentModel
        _IM_OK = True
    except ImportError:
        _IM_OK = False
        InstrumentModel = None   # type: ignore


# ============================================================================
# SufficientStats — the read-only product
# ============================================================================

@dataclass
class SufficientStats:
    """
    The complete set of accumulated statistics needed by the MAP stacker.

    All pixel arrays are float32, shape [H, W].

    Attributes
    ----------
    weighted_sum
        Σ_i  t_i · (x_i − sky_i)   transparency-weighted sky-subtracted sum
    weight_sum
        Σ_i  t_i                    sum of transparency weights
    sky_sum
        Σ_i  sky_i                  sum of sky background models
    sq_sum
        Σ_i  t_i · (x_i − sky_i)²  weighted sum of squares (for variance)
    frame_count
        Number of frames accumulated.
    shift_list
        Per-frame FrameShift objects (or None when solve failed).
    psf_list
        Per-frame PSF kernels [K, K] float32.
    transparency_list
        Per-frame transparency values.
    fwhm_list
        Per-frame FWHM in arcsec.
    frame_shape
        (H, W) of the sensor.
    """
    weighted_sum:      np.ndarray            # [H, W] float32
    weight_sum:        np.ndarray            # [H, W] float32
    sky_sum:           np.ndarray            # [H, W] float32
    sq_sum:            np.ndarray            # [H, W] float32
    frame_count:       int
    shift_list:        List[Optional[object]]
    psf_list:          List[np.ndarray]
    transparency_list: List[float]
    fwhm_list:         List[float]
    frame_shape:       Tuple[int, int]

    # ------------------------------------------------------------------
    # Derived products
    # ------------------------------------------------------------------

    @property
    def weighted_mean(self) -> np.ndarray:
        """
        Transparency-weighted mean stack (fast preview result).

        This is the Gamma-Poisson posterior mean:
            E[λ | data] ≈ weighted_sum / weight_sum

        Quality increases with total exposure — equivalent to stacking all
        frames with per-frame transparency weighting.  No deconvolution.
        """
        w = np.where(self.weight_sum > 0, self.weight_sum, 1.0)
        return (self.weighted_sum / w).astype(np.float32)

    @property
    def sky_mean(self) -> np.ndarray:
        """Mean sky background across all frames [ADU]."""
        if self.frame_count == 0:
            return np.zeros(self.frame_shape, dtype=np.float32)
        return (self.sky_sum / self.frame_count).astype(np.float32)

    @property
    def variance_map(self) -> np.ndarray:
        """
        Per-pixel empirical variance of the sky-subtracted, weighted frames.

        Var = (sq_sum / weight_sum) - weighted_mean²
        Useful for identifying pixels with excess noise (satellites, planes).
        """
        w    = np.where(self.weight_sum > 0, self.weight_sum, 1.0)
        var  = self.sq_sum / w - self.weighted_mean ** 2
        return np.maximum(var, 0.0).astype(np.float32)

    @property
    def quality_map(self) -> np.ndarray:
        """
        Per-pixel quality weight in [0, 1].

        quality = weight_sum / max(weight_sum)
        Pixels that were vignetted or masked in many frames will have
        lower quality.  The MAP solver can use this as a prior weight.
        """
        wmax = float(self.weight_sum.max())
        if wmax <= 0:
            return np.zeros(self.frame_shape, dtype=np.float32)
        return (self.weight_sum / wmax).astype(np.float32)

    @property
    def mean_fwhm_arcsec(self) -> float:
        """Median seeing FWHM across all characterised frames."""
        if not self.fwhm_list:
            return 0.0
        return float(np.median(self.fwhm_list))

    @property
    def mean_transparency(self) -> float:
        """Median transparency across all frames."""
        if not self.transparency_list:
            return 1.0
        return float(np.median(self.transparency_list))

    def summary(self) -> str:
        t_arr = np.array(self.transparency_list) if self.transparency_list else np.array([1.0])
        f_arr = np.array(self.fwhm_list)         if self.fwhm_list         else np.array([0.0])
        lines = [
            "SufficientStats",
            "=" * 52,
            f"  Frame shape    : {self.frame_shape[0]} × {self.frame_shape[1]} px",
            f"  Frames         : {self.frame_count}",
            f"  Transparency   : median={float(np.median(t_arr)):.3f}"
            f"  min={float(t_arr.min()):.3f}  max={float(t_arr.max()):.3f}",
            f"  FWHM           : median={float(np.median(f_arr)):.2f}\""
            f"  min={float(f_arr.min()):.2f}\"  max={float(f_arr.max()):.2f}\"",
            f"  weight_sum     : min={float(self.weight_sum.min()):.2f}"
            f"  median={float(np.median(self.weight_sum)):.2f}"
            f"  max={float(self.weight_sum.max()):.2f}",
            f"  weighted_mean  : min={float(self.weighted_mean.min()):.1f}"
            f"  median={float(np.median(self.weighted_mean)):.1f}"
            f"  max={float(self.weighted_mean.max()):.1f}  ADU",
            f"  PSF kernels    : {len(self.psf_list)} stored"
            f"  size={self.psf_list[0].shape if self.psf_list else 'n/a'}",
            f"  Shifts         : "
            f"{sum(s is not None for s in self.shift_list)}/{len(self.shift_list)}"
            f" solved",
            "=" * 52,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HDF5 persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save to HDF5.  Safe to call on an existing file (overwrites /stats/).

        Layout
        ------
        /stats/weighted_sum    [H, W] float32
        /stats/weight_sum      [H, W] float32
        /stats/sky_sum         [H, W] float32
        /stats/sq_sum          [H, W] float32
        /stats/frame_count     scalar int
        /stats/frame_shape     [2] int
        /stats/psf_kernels/0 … [K, K] float32
        /stats/transparency    [N] float32
        /stats/fwhm            [N] float32
        /stats/shifts/dx_px    [N] float32  (-9999 where None)
        /stats/shifts/dy_px    [N] float32
        /stats/shifts/rot_deg  [N] float32
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "a") as f:
            # Clear previous stats group if present
            if "stats" in f:
                del f["stats"]
            g = f.require_group("stats")

            opts = {"compression": "gzip", "compression_opts": 4}
            g.create_dataset("weighted_sum", data=self.weighted_sum.astype(np.float32), **opts)
            g.create_dataset("weight_sum",   data=self.weight_sum.astype(np.float32),   **opts)
            g.create_dataset("sky_sum",      data=self.sky_sum.astype(np.float32),      **opts)
            g.create_dataset("sq_sum",       data=self.sq_sum.astype(np.float32),       **opts)
            g.attrs["frame_count"] = self.frame_count
            g.attrs["frame_shape"] = list(self.frame_shape)

            # Per-frame transparency and FWHM
            g.create_dataset("transparency",
                             data=np.array(self.transparency_list, dtype=np.float32))
            g.create_dataset("fwhm",
                             data=np.array(self.fwhm_list, dtype=np.float32))

            # PSF kernels
            pk = g.require_group("psf_kernels")
            for i, psf in enumerate(self.psf_list):
                pk.create_dataset(str(i), data=psf.astype(np.float32), **opts)

            # Shifts (sentinel -9999 for None)
            sg = g.require_group("shifts")
            N  = len(self.shift_list)
            dx  = np.full(N, -9999.0, dtype=np.float32)
            dy  = np.full(N, -9999.0, dtype=np.float32)
            rot = np.full(N, -9999.0, dtype=np.float32)
            scl = np.full(N, 1.0,     dtype=np.float32)
            for i, sh in enumerate(self.shift_list):
                if sh is not None:
                    dx[i]  = sh.dx_px
                    dy[i]  = sh.dy_px
                    rot[i] = sh.rotation_deg
                    scl[i] = sh.scale_ratio
            sg.create_dataset("dx_px",   data=dx)
            sg.create_dataset("dy_px",   data=dy)
            sg.create_dataset("rot_deg", data=rot)
            sg.create_dataset("scale",   data=scl)

        logger.info("SufficientStats saved to %s  (%d frames)", path, self.frame_count)

    @classmethod
    def load(cls, path: str | Path) -> "SufficientStats":
        """Load SufficientStats from HDF5 file saved by save()."""
        path = Path(path)
        with h5py.File(path, "r") as f:
            g   = f["stats"]
            ws  = g["weighted_sum"][:]
            wt  = g["weight_sum"][:]
            ss  = g["sky_sum"][:]
            sq  = g["sq_sum"][:]
            nc  = int(g.attrs["frame_count"])
            shp = tuple(int(x) for x in g.attrs["frame_shape"])

            t_list    = g["transparency"][:].tolist()
            fwhm_list = g["fwhm"][:].tolist()

            pk        = g["psf_kernels"]
            psf_list  = [pk[str(i)][:] for i in range(len(pk))]

            sg  = g["shifts"]
            dx  = sg["dx_px"][:]
            dy  = sg["dy_px"][:]
            rot = sg["rot_deg"][:]
            scl = sg["scale"][:]

        # Reconstruct shift objects
        _SENTINEL = -9999.0
        shift_list: List[Optional[object]] = []
        for i in range(len(dx)):
            if abs(dx[i] - _SENTINEL) < 1.0:
                shift_list.append(None)
            else:
                # Minimal duck-typed shift object
                shift_list.append(_ShiftProxy(
                    float(dx[i]), float(dy[i]),
                    float(rot[i]), float(scl[i]),
                ))

        logger.info("SufficientStats loaded from %s  (%d frames)", path, nc)
        return cls(
            weighted_sum      = ws.astype(np.float32),
            weight_sum        = wt.astype(np.float32),
            sky_sum           = ss.astype(np.float32),
            sq_sum            = sq.astype(np.float32),
            frame_count       = nc,
            shift_list        = shift_list,
            psf_list          = [p.astype(np.float32) for p in psf_list],
            transparency_list = t_list,
            fwhm_list         = fwhm_list,
            frame_shape       = shp,
        )

    def save_fast_stack_fits(
        self,
        path:          str | Path,
        bayer_pattern: Optional[str] = None,
    ) -> None:
        """
        Write the weighted_mean fast stack as a FITS file.

        For OSC cameras pass bayer_pattern (e.g. 'RGGB') to split the Bayer
        mosaic into a [4, H//2, W//2] data cube with one plane per channel.
        Channel order follows _BAYER_OFFSETS: R, G0, G1, B (or equivalent
        for non-RGGB patterns).  The BAYERPAT and CHANn keywords record the
        layout.  For mono cameras (bayer_pattern=None) a plain 2-D FITS is
        written as before.
        """
        from instrument_model_artifact import bayer_split
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        hdr = fits.Header()
        hdr["NFRAMES"]  = self.frame_count
        hdr["MEDTRANS"] = round(self.mean_transparency, 4)
        hdr["MEDFWHM"]  = round(self.mean_fwhm_arcsec, 3)
        hdr["COMMENT"]  = "Weighted mean stack (Gamma-Poisson posterior mean)"
        wm = self.weighted_mean
        if bayer_pattern is not None:
            hdr["BAYERPAT"] = bayer_pattern
            if wm.ndim == 3:
                # Already split by calibrate_frame() — write cube directly.
                # Channel names follow _BAYER_OFFSETS order for this pattern.
                from instrument_model_artifact import _BAYER_OFFSETS
                names = list(_BAYER_OFFSETS[bayer_pattern].keys())
                for i, n in enumerate(names, 1):
                    hdr[f"CHAN{i}"] = n
                fits.writeto(str(path), wm, hdr, overwrite=True)
            else:
                planes, names = bayer_split(wm, bayer_pattern)
                for i, n in enumerate(names, 1):
                    hdr[f"CHAN{i}"] = n
                fits.writeto(str(path), planes, hdr, overwrite=True)
        else:
            fits.writeto(str(path), wm, hdr, overwrite=True)
        logger.info("Fast stack saved to %s", path)


class _ShiftProxy:
    """Minimal shift object reconstructed from HDF5 scalars."""
    __slots__ = ("dx_px", "dy_px", "rotation_deg", "scale_ratio")

    def __init__(self, dx, dy, rot, scale):
        self.dx_px        = dx
        self.dy_px        = dy
        self.rotation_deg = rot
        self.scale_ratio  = scale

    def __repr__(self):
        return (f"Shift(dx={self.dx_px:+.3f}px  dy={self.dy_px:+.3f}px  "
                f"rot={self.rotation_deg:+.4f}°)")


# ============================================================================
# SufficientStatsAccumulator — streaming accumulator
# ============================================================================

class SufficientStatsAccumulator:
    """
    Stream light frames one at a time and accumulate sufficient statistics.

    Memory usage is O(6 × H × W × 8 bytes) regardless of frame count.
    For a 3008 × 3008 sensor: ~420 MB.

    Parameters
    ----------
    frame_shape : (H, W)
        Expected sensor dimensions.  Set from the first frame if None.
    outlier_sigma : float
        Frames with transparency below (median - outlier_sigma × MAD) are
        flagged but still included (they contribute less because t_i is small).
        Set to 0 to disable per-frame outlier reporting.
    """

    def __init__(
        self,
        frame_shape:   Optional[Tuple[int, int]] = None,
        outlier_sigma: float = 3.0,
    ) -> None:
        self._shape         = frame_shape   # sensor pixel shape (H, W) per channel
        self._outlier_sigma = outlier_sigma

        # Union-canvas geometry — set by set_canvas() before first add_calibrated.
        # _canvas_shape : (cH, cW) — the union bounding box in sensor pixels
        # _sensor_origin: (r0, c0) — where the reference frame's top-left corner
        #                  sits within the canvas (always >= 0)
        self._canvas_shape:  Optional[Tuple[int, int]] = None
        self._sensor_origin: Tuple[int, int]           = (0, 0)

        # Running accumulators — initialised on first frame (or set_canvas)
        self._weighted_sum: Optional[np.ndarray] = None
        self._weight_sum:   Optional[np.ndarray] = None
        self._sky_sum:      Optional[np.ndarray] = None
        self._sq_sum:       Optional[np.ndarray] = None

        self._frame_count:      int               = 0
        self._shift_list:       List               = []
        self._psf_list:         List[np.ndarray]  = []
        self._transparency_list: List[float]       = []
        self._fwhm_list:        List[float]        = []

    # ------------------------------------------------------------------
    # Canvas setup
    # ------------------------------------------------------------------

    def set_canvas(
        self,
        canvas_shape:  Tuple[int, int],
        sensor_origin: Tuple[int, int],
    ) -> None:
        """
        Configure the union bounding-box canvas before accumulation begins.

        Must be called before the first add_calibrated() / add_frame() when
        union-mode stacking is desired.  If never called the canvas defaults
        to the sensor footprint (original behaviour).

        Parameters
        ----------
        canvas_shape : (cH, cW)
            Size of the union canvas in sensor-resolution pixels (per channel).
            For OSC cameras this is already in half-resolution units.
        sensor_origin : (r0, c0)
            Pixel position within the canvas where the reference frame's
            top-left corner (row 0, col 0) is placed.
        """
        if self._weighted_sum is not None:
            raise RuntimeError("set_canvas() must be called before any frames are accumulated")
        self._canvas_shape  = canvas_shape
        self._sensor_origin = sensor_origin

    # ------------------------------------------------------------------
    # Core accumulation
    # ------------------------------------------------------------------

    def add_frame(
        self,
        fits_path:    str | Path,
        model,                         # InstrumentModel
        characterizer: "FrameCharacterizer",
        exposure_s:   float,
        is_reference: bool = False,
    ) -> FrameMetadata:
        """
        Load, calibrate, characterise, and accumulate one light frame.

        Parameters
        ----------
        fits_path    : path to raw FITS file
        model        : fitted InstrumentModel
        characterizer: FrameCharacterizer instance (shared across frames)
        exposure_s   : exposure time in seconds
        is_reference : True for the first / best-seeing frame of the session

        Returns
        -------
        FrameMetadata (for inspection or logging)
        """
        fits_path = Path(fits_path)

        # Load and calibrate
        raw, header = _load_fits(fits_path)
        cal         = model.calibrate_frame(raw, exposure_s)

        # Characterise
        meta = characterizer.characterize_calibrated(
            cal, header, exposure_s,
            is_reference=is_reference,
            frame_path=fits_path,
        )

        # Accumulate
        self._accumulate(cal, meta)
        return meta

    def add_calibrated(
        self,
        calibrated:   np.ndarray,
        meta:         "FrameMetadata",
    ) -> None:
        """
        Accumulate a frame that has already been calibrated and characterised.

        Use this when the calling code has its own calibration loop and
        calls ``characterizer.characterize_calibrated()`` separately.
        """
        self._accumulate(calibrated, meta)

    def _accumulate(
        self,
        calibrated: np.ndarray,
        meta:       "FrameMetadata",
    ) -> None:
        cal = calibrated.astype(np.float64)
        sky = meta.sky_bg.astype(np.float64)
        t   = float(meta.transparency)

        is_osc = (cal.ndim == 3)
        if is_osc:
            n_ch       = cal.shape[0]
            data_shape = cal.shape[-2:]   # (H//2, W//2)
            osc_scale  = 2
        else:
            n_ch       = 1
            data_shape = cal.shape        # (H, W)
            osc_scale  = 1

        if self._shape is None:
            self._shape = data_shape
        elif self._shape != data_shape:
            raise ValueError(
                f"Frame shape {data_shape} differs from expected {self._shape}"
            )

        # Canvas geometry
        if self._canvas_shape is not None:
            cH, cW = self._canvas_shape
        else:
            cH, cW = data_shape
        r0, c0 = self._sensor_origin

        # Lazy initialisation of accumulator arrays
        if self._weighted_sum is None:
            canvas_arr_shape = (n_ch, cH, cW) if is_osc else (cH, cW)
            self._weighted_sum = np.zeros(canvas_arr_shape, dtype=np.float64)
            self._weight_sum   = np.zeros(canvas_arr_shape, dtype=np.float64)
            self._sky_sum      = np.zeros(canvas_arr_shape, dtype=np.float64)
            self._sq_sum       = np.zeros(canvas_arr_shape, dtype=np.float64)

        # ── Per-channel sky for OSC ───────────────────────────────────────────
        # sky_bg is estimated from the channel-mean luminance, so it encodes
        # the *spatial shape* of the sky but at the wrong per-channel level.
        # For narrowband (Ha→R, OIII→G+B) the per-channel sky levels differ
        # significantly.  Scale the spatial sky map per channel using each
        # channel's own median pixel value as the level reference.
        if is_osc:
            # sky is [H//2, W//2] — the spatial sky shape at lum-mean level
            # cal  is [4, H//2, W//2]
            sky_per_ch = np.empty((n_ch,) + sky.shape, dtype=np.float64)
            lum_sky_median = float(np.median(sky))
            for c in range(n_ch):
                ch_median = float(np.median(cal[c]))
                scale     = ch_median / lum_sky_median if lum_sky_median > 0 else 1.0
                sky_per_ch[c] = sky * scale  # [H//2, W//2] — per-channel sky
        # sky_per_ch only used in the OSC branch below

        # ── Warp frame onto canvas ────────────────────────────────────────────
        # Prefer full WCS affine warp (handles rotation + scale + translation
        # exactly in one bicubic pass).  Fall back to phase-shift-only when
        # WCS is unavailable (plate solve failed).
        use_wcs = (meta.wcs_geom is not None and meta.ref_wcs is not None)

        if use_wcs:
            # Sky-subtract first (sky is in sensor coords)
            if is_osc:
                sub_sensor = cal - sky_per_ch   # [4, H//2, W//2] per-channel sky
            else:
                sub_sensor = cal - sky          # [H, W]

            # Warp each plane to the canvas via full WCS affine transform
            def warp(plane):
                return _warp_to_canvas(
                    plane, meta.wcs_geom, meta.ref_wcs,
                    (cH, cW), (r0, c0), osc_scale,
                )

            if is_osc:
                warped_sub = np.stack([warp(sub_sensor[c]) for c in range(n_ch)])
                warped_sky = np.stack([warp(sky_per_ch[c]) for c in range(n_ch)])
            else:
                warped_sub = warp(sub_sensor)
                warped_sky = warp(sky)

            # Build a coverage mask: canvas pixels that received data from this frame
            # (affine_transform fills out-of-bounds with cval=0; use a ones-mask
            # warped the same way to know where valid data landed)
            ones = np.ones(data_shape, dtype=np.float64)
            coverage = _warp_to_canvas(ones, meta.wcs_geom, meta.ref_wcs,
                                       (cH, cW), (r0, c0), osc_scale)
            valid = coverage > 0.5   # [cH, cW] bool

            if is_osc:
                self._weighted_sum[:, valid] += t * warped_sub[:, valid]
                self._weight_sum[:, valid]   += t
                self._sky_sum[:, valid]      += warped_sky[:, valid]
                self._sq_sum[:, valid]       += t * warped_sub[:, valid] ** 2
            else:
                self._weighted_sum[valid] += t * warped_sub[valid]
                self._weight_sum[valid]   += t
                self._sky_sum[valid]      += warped_sky[valid]
                self._sq_sum[valid]       += t * warped_sub[valid] ** 2

        else:
            # Fallback: pure translation (phase-shift), no rotation
            sh = meta.shift
            dx_total = sh.dx_px if sh is not None else 0.0
            dy_total = sh.dy_px if sh is not None else 0.0
            dx_int = int(round(dx_total));  dx_frac = dx_total - dx_int
            dy_int = int(round(dy_total));  dy_frac = dy_total - dy_int

            if is_osc:
                sub = cal - sky_per_ch   # per-channel sky subtraction
                if dx_frac or dy_frac:
                    for c in range(n_ch):
                        sub[c]        = _apply_phase_shift(sub[c], dx_frac, dy_frac)
                        sky_per_ch[c] = _apply_phase_shift(sky_per_ch[c], dx_frac, dy_frac)
                sky = sky_per_ch
            else:
                sub = cal - sky
                if dx_frac or dy_frac:
                    sub = _apply_phase_shift(sub, dx_frac, dy_frac)
                    sky = _apply_phase_shift(sky, dx_frac, dy_frac)

            fr0 = r0 + dy_int;  fc0 = c0 + dx_int
            fH, fW = data_shape
            r_src0 = max(0,-fr0);       r_src1 = fH - max(0, fr0+fH-cH)
            c_src0 = max(0,-fc0);       c_src1 = fW - max(0, fc0+fW-cW)
            r_dst0 = max(0, fr0);       r_dst1 = r_dst0 + (r_src1-r_src0)
            c_dst0 = max(0, fc0);       c_dst1 = c_dst0 + (c_src1-c_src0)
            if r_src1 <= r_src0 or c_src1 <= c_src0:
                logger.warning("Frame shift outside canvas — skipping")
                return
            if is_osc:
                self._weighted_sum[:, r_dst0:r_dst1, c_dst0:c_dst1] += \
                    t * sub[:, r_src0:r_src1, c_src0:c_src1]
                self._weight_sum[:, r_dst0:r_dst1, c_dst0:c_dst1] += t
                self._sky_sum[:, r_dst0:r_dst1, c_dst0:c_dst1] += \
                    sky[:, r_src0:r_src1, c_src0:c_src1]
                self._sq_sum[:, r_dst0:r_dst1, c_dst0:c_dst1] += \
                    t * sub[:, r_src0:r_src1, c_src0:c_src1] ** 2
            else:
                self._weighted_sum[r_dst0:r_dst1, c_dst0:c_dst1] += \
                    t * sub[r_src0:r_src1, c_src0:c_src1]
                self._weight_sum[r_dst0:r_dst1, c_dst0:c_dst1] += t
                self._sky_sum[r_dst0:r_dst1, c_dst0:c_dst1] += \
                    sky[r_src0:r_src1, c_src0:c_src1]
                self._sq_sum[r_dst0:r_dst1, c_dst0:c_dst1] += \
                    t * sub[r_src0:r_src1, c_src0:c_src1] ** 2

        self._shift_list.append(meta.shift)
        self._psf_list.append(meta.psf_total.copy())
        self._transparency_list.append(t)
        self._fwhm_list.append(meta.fwhm_arcsec)
        self._frame_count += 1

        if self._frame_count % 10 == 0:
            logger.info(
                "Accumulated %d frames  (last: t=%.3f  FWHM=%.2f\")",
                self._frame_count, t, meta.fwhm_arcsec,
            )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def finalize(self) -> SufficientStats:
        """
        Return the accumulated SufficientStats.

        Can be called multiple times (non-destructive).
        Raises RuntimeError if no frames have been accumulated.
        """
        if self._frame_count == 0 or self._weighted_sum is None:
            raise RuntimeError(
                "No frames accumulated. Call add_frame() at least once."
            )

        self._report_outliers()

        # frame_shape reported to MAP stacker is the canvas (union bbox) size,
        # which equals the sensor size when set_canvas() was not called.
        reported_shape = self._canvas_shape if self._canvas_shape is not None else self._shape

        return SufficientStats(
            weighted_sum      = self._weighted_sum.astype(np.float32),
            weight_sum        = self._weight_sum.astype(np.float32),
            sky_sum           = self._sky_sum.astype(np.float32),
            sq_sum            = self._sq_sum.astype(np.float32),
            frame_count       = self._frame_count,
            shift_list        = list(self._shift_list),
            psf_list          = [p.copy() for p in self._psf_list],
            transparency_list = list(self._transparency_list),
            fwhm_list         = list(self._fwhm_list),
            frame_shape       = reported_shape,
        )

    def save(self, path: str | Path) -> None:
        """Checkpoint: save current state to HDF5 (calls finalize internally)."""
        stats = self.finalize()
        stats.save(path)
        logger.info("Checkpoint saved to %s", path)

    @classmethod
    def resume(cls, path: str | Path) -> "SufficientStatsAccumulator":
        """
        Resume accumulation from a saved checkpoint.

        Returns an accumulator whose internal state mirrors the checkpoint.
        New frames added via add_frame() will extend the existing data.
        """
        stats = SufficientStats.load(path)
        acc   = cls(frame_shape=stats.frame_shape)
        acc._weighted_sum      = stats.weighted_sum.astype(np.float64)
        acc._weight_sum        = stats.weight_sum.astype(np.float64)
        acc._sky_sum           = stats.sky_sum.astype(np.float64)
        acc._sq_sum            = stats.sq_sum.astype(np.float64)
        acc._frame_count       = stats.frame_count
        acc._shift_list        = list(stats.shift_list)
        acc._psf_list          = [p.copy() for p in stats.psf_list]
        acc._transparency_list = list(stats.transparency_list)
        acc._fwhm_list         = list(stats.fwhm_list)
        logger.info("Resumed from %s  (%d frames already accumulated)",
                    path, stats.frame_count)
        return acc

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _report_outliers(self) -> None:
        if self._outlier_sigma <= 0 or len(self._transparency_list) < 4:
            return
        t   = np.array(self._transparency_list)
        med = float(np.median(t))
        mad = float(np.median(np.abs(t - med))) * 1.4826
        lo  = med - self._outlier_sigma * mad
        bad = np.where(t < lo)[0]
        if len(bad):
            logger.warning(
                "%d frames have low transparency (< %.3f): indices %s",
                len(bad), lo, bad.tolist(),
            )

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def frame_shape(self) -> Optional[Tuple[int, int]]:
        return self._shape

    def transparency_stats(self) -> dict:
        """Return summary statistics for accumulated transparencies."""
        if not self._transparency_list:
            return {}
        t = np.array(self._transparency_list)
        return {
            "n":      len(t),
            "min":    float(t.min()),
            "median": float(np.median(t)),
            "max":    float(t.max()),
            "std":    float(t.std()),
        }

    def fwhm_stats(self) -> dict:
        """Return summary statistics for accumulated FWHM values."""
        if not self._fwhm_list:
            return {}
        f = np.array(self._fwhm_list)
        return {
            "n":       len(f),
            "min":     float(f.min()),
            "median":  float(np.median(f)),
            "max":     float(f.max()),
            "best_n":  int(np.sum(f < float(np.percentile(f, 25)))),
        }


# ============================================================================
# Utilities
# ============================================================================

def _load_fits(path: Path) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        data   = hdul[0].data.astype(np.float32).squeeze()
        header = hdul[0].header
    return data, header


def select_best_frames(
    stats:    SufficientStats,
    top_frac: float = 0.5,
    key:      str   = "transparency",
) -> List[int]:
    """
    Return indices of the best frames by a given metric.

    Parameters
    ----------
    stats    : SufficientStats from finalize()
    top_frac : keep this fraction of frames (default 0.5 = best 50%)
    key      : 'transparency' or 'fwhm'
               For fwhm, lower is better (best seeing).

    Returns
    -------
    list of int  indices into stats.shift_list / psf_list / transparency_list
    """
    n = stats.frame_count
    k = max(1, int(round(n * top_frac)))

    if key == "transparency":
        vals = np.array(stats.transparency_list)
        idx  = np.argsort(-vals)       # descending: highest transparency first
    elif key == "fwhm":
        vals = np.array(stats.fwhm_list)
        idx  = np.argsort(vals)        # ascending: lowest FWHM (best seeing) first
    else:
        raise ValueError(f"key must be 'transparency' or 'fwhm', got '{key}'")

    selected = idx[:k].tolist()
    logger.info(
        "select_best_frames: %d/%d frames selected by %s",
        len(selected), n, key,
    )
    return selected


def rebuild_stats_from_subset(
    stats:   SufficientStats,
    indices: List[int],
) -> SufficientStats:
    """
    Rebuild SufficientStats using only a subset of the accumulated frames.

    This re-runs the pixel-level accumulation from the per-frame PSF and
    transparency lists, but WITHOUT the raw pixel data (which was discarded).

    NOTE: The pixel arrays (weighted_sum, weight_sum, etc.) cannot be
    reconstructed without the original frames.  This function returns a
    SufficientStats whose pixel arrays are re-scaled by the subset weights:

        weighted_sum_subset ≈ weighted_sum_full × (Σ_subset t_i) / (Σ_all t_i)

    This is an approximation.  For exact per-pixel subset accumulation,
    run add_frame() only on the selected frames from the start.

    The returned stats has correct shift_list, psf_list, and transparency_list
    for the subset, and correctly scaled pixel arrays.
    """
    if not indices:
        raise ValueError("indices list is empty")

    # Compute weight ratio for pixel rescaling
    all_t    = np.array(stats.transparency_list, dtype=np.float64)
    sub_t    = all_t[indices]
    w_all    = float(all_t.sum())
    w_sub    = float(sub_t.sum())
    scale    = w_sub / max(w_all, 1e-12)

    logger.info(
        "rebuild_stats_from_subset: %d/%d frames  weight_ratio=%.3f",
        len(indices), stats.frame_count, scale,
    )

    return SufficientStats(
        weighted_sum      = (stats.weighted_sum.astype(np.float64) * scale
                             ).astype(np.float32),
        weight_sum        = (stats.weight_sum.astype(np.float64) * scale
                             ).astype(np.float32),
        sky_sum           = (stats.sky_sum.astype(np.float64) * scale
                             ).astype(np.float32),
        sq_sum            = (stats.sq_sum.astype(np.float64) * scale
                             ).astype(np.float32),
        frame_count       = len(indices),
        shift_list        = [stats.shift_list[i]        for i in indices],
        psf_list          = [stats.psf_list[i].copy()   for i in indices],
        transparency_list = [stats.transparency_list[i] for i in indices],
        fwhm_list         = [stats.fwhm_list[i]         for i in indices],
        frame_shape       = stats.frame_shape,
    )
