"""
map_stacker.py
==============
Phase 4 — Bayesian MAP super-resolution stacker.

Solves for a super-resolved scene λ on an (S·H) × (S·W) grid (S = scale_factor)
by minimising a regularised Poisson negative log-likelihood:

    Loss(λ) = L_data(λ)
            + alpha_tv   · TV(λ)
            + alpha_kl   · KL(λ ‖ λ₀)
            + alpha_wav  · Wavelet_L1(λ)

Non-negativity is enforced by the softplus reparameterisation:
    λ = softplus(θ)  =  log(1 + exp(θ))
The optimiser sees θ ∈ ℝ; λ is always > 0.

Data likelihood (two modes)
----------------------------

FAST mode  (default)
    Uses the Phase 3 sufficient statistics as a single proxy observation.

        L_data = Poisson_NLL(weighted_sum | weight_sum · D(H̄ ⊛ λ_hr))

    where D(·) = average-pool downsampling to the native sensor resolution,
    H̄ = mean PSF, weight_sum = Σ t_i.

    This treats the weighted stack as if it were a single noiseless
    "super-exposure" with effective transparency = weight_sum.  It is an
    approximation but is ~50× faster than EXACT mode and produces nearly
    identical results for well-calibrated, dithered datasets.

EXACT mode
    Uses individual raw FITS frames loaded on demand in mini-batches.
    For each batch {i₁, …, i_B}:

        L_data = Σ_b Poisson_NLL(x_b | t_b · D(H_b ⊛ W_b[λ_hr]))

    where W_b[λ_hr] is the sub-pixel shift operator applied in Fourier space.
    The gradient is an unbiased estimate of the full-dataset gradient.

Sub-pixel shift operator
-------------------------
For each frame i with shift (dx, dy):

    1. Apply phase-shift in Fourier domain of λ_hr:
          λ̃_hr(f) = λ_hr(f) · exp(-2πi (fy·dy/S + fx·dx/S))
    2. Average-pool by factor S to get λ̃_lr [H, W]

This is exact (no interpolation) for shifts that are integer multiples of
the HR pixel size.  For arbitrary sub-pixel shifts it is accurate to the
Nyquist limit of the HR grid.

PSF convolution
----------------
    λ_conv = real(IFFT2(FFT2(λ_hr_padded) · FFT2(H_padded)))

Padding to (next_power_of_2(2·S·H), next_power_of_2(2·S·W)) avoids circular
convolution artefacts.

Regularisation
--------------
TV   — isotropic total variation:
           TV(λ) = Σ sqrt((∂x λ)² + (∂y λ)² + ε_tv)
           ε_tv = 1e-8 avoids the gradient singularity at flat regions.

KL   — Poisson KL divergence from a smooth prior λ₀:
           KL(λ‖λ₀) = Σ [λ·log(λ/λ₀) - λ + λ₀]
           λ₀ is the Phase 3 weighted_mean upsampled to the HR grid.
           Penalises departures from the expected flux.

Wavelet — one-level Haar L1 sparsity on detail coefficients:
           Wav(λ) = ||HH||₁ + ||HL||₁ + ||LH||₁
           where HH, HL, LH are the diagonal, horizontal, vertical
           Haar detail sub-bands.  Promotes star sharpness.

Optimiser
---------
Adam with cosine-annealing LR schedule.
Early stopping when |ΔLoss|/|Loss| < rel_tol for patience consecutive steps.

Outputs
-------
    MapResult
        lambda_hr    : [S·H, S·W] float32  super-resolved scene
        loss_history : list[float]          loss at every logged step
        n_iter       : int                  actual iterations run
        converged    : bool
        device       : str
        config       : MapConfig

    Written to FITS:
        lambda_hr.fits        primary HDU  — the super-resolved scene
        quality_map_hr.fits   quality map upsampled to HR grid (optional)
        convergence.png       loss curve (optional)

Dependencies
------------
    torch >= 2.0   numpy   scipy   astropy   h5py
    sufficient_statistics  (Phase 3)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PyTorch import — everything degrades to NotImplementedError with
# a clear message when torch is absent.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    logger.warning(
        "PyTorch not found.  Install with: pip install torch\n"
        "map_stacker.py requires PyTorch for GPU-accelerated MAP optimisation."
    )

# ---------------------------------------------------------------------------
# Project imports (all optional for isolated testing)
# ---------------------------------------------------------------------------
try:
    from sufficient_statistics import SufficientStats
    _SS_OK = True
except ImportError:
    _SS_OK = False
    SufficientStats = None   # type: ignore


# ============================================================================
# MapConfig — all hyperparameters in one dataclass
# ============================================================================

@dataclass
class MapConfig:
    """
    All hyperparameters for the MAP stacker.

    Tune alpha_* weights first.  Start with the defaults and inspect the
    convergence plot; if the result is over-smoothed, halve alpha_tv.
    If stars look spiky/noisy, double alpha_tv or add alpha_wav.

    Parameters
    ----------
    scale_factor : int
        Super-resolution upscaling factor.  2 = 2× linear resolution.
        Must be a positive integer.  Values > 4 rarely help because the
        information content of the data limits resolution gain.

    n_iter : int
        Maximum number of optimisation iterations.

    lr : float
        Initial Adam learning rate.  1e-2 is a good default.

    lr_min : float
        Minimum LR for cosine annealing schedule.

    alpha_tv : float
        Total Variation regularisation weight.
        0 = no TV (raw likelihood solution — typically very noisy).
        Typical range: 1e-3 – 1e-1.  Start at 1e-2.

    alpha_kl : float
        KL-from-prior regularisation weight.
        0 = disabled.
        Keeps λ near the Phase 3 weighted_mean prior.
        Typical range: 1e-4 – 1e-2.

    alpha_wav : float
        Haar wavelet L1 sparsity weight.
        0 = disabled (default).  Useful for star-field images.
        Typical range: 1e-4 – 1e-2.

    tv_eps : float
        Isotropic TV numerical stability epsilon.  1e-8 default.

    rel_tol : float
        Early stopping: stop when |ΔLoss|/|Loss| < rel_tol for
        `patience` consecutive steps.

    patience : int
        Number of consecutive steps below rel_tol before stopping.

    log_every : int
        Log loss every this many iterations.

    mode : str
        'fast'  — use Phase 3 sufficient statistics as proxy observation.
        'exact' — load raw FITS frames in mini-batches (requires fits_paths).

    batch_size : int
        Number of frames per mini-batch (EXACT mode only).

    device : str | None
        PyTorch device string: 'cuda', 'mps', 'cpu', or None (auto-detect).

    seed : int
        Random seed for reproducible mini-batch ordering (EXACT mode).
    """
    scale_factor: int   = 2

    # Optimiser
    n_iter:   int   = 300
    lr:       float = 1e-2
    lr_min:   float = 1e-4

    # Regularisation weights (0 = disabled)
    alpha_tv:  float = 1e-2
    alpha_kl:  float = 0.0
    alpha_wav: float = 0.0

    # Numerical stability
    tv_eps:    float = 1e-8

    # Convergence
    rel_tol:   float = 1e-6
    patience:  int   = 20

    # Logging
    log_every: int   = 10

    # Mode
    mode:       str  = 'fast'
    batch_size: int  = 8          # EXACT mode: frames per mini-batch

    # Forward model mini-batch (FAST mode): number of frames to process per
    # rfft2 call. Larger = more parallelism but more RAM. 4-8 is good for CPU;
    # 16-32 for GPU with 8GB+; set to N to process all frames in one shot.
    forward_batch_size: int = 8

    # Hardware
    device:    Optional[str] = None
    seed:      int           = 0

    def validate(self) -> None:
        assert self.scale_factor >= 1,  "scale_factor must be >= 1"
        assert self.n_iter > 0,         "n_iter must be > 0"
        assert self.lr > 0,             "lr must be > 0"
        assert self.alpha_tv  >= 0,     "alpha_tv must be >= 0"
        assert self.alpha_kl  >= 0,     "alpha_kl must be >= 0"
        assert self.alpha_wav >= 0,     "alpha_wav must be >= 0"
        assert self.mode in ('fast', 'exact'), "mode must be 'fast' or 'exact'"
        assert self.batch_size >= 1,    "batch_size must be >= 1"


# ============================================================================
# MapResult — output of solve()
# ============================================================================

@dataclass
class MapResult:
    """
    Output of the MAP stacker.

    Attributes
    ----------
    lambda_hr : [S·H, S·W] float32
        Super-resolved scene estimate.  Units: ADU (same as calibrated frames).
    loss_history : list[float]
        Loss value at each logged iteration.
    grad_norm_history : list[float]
        Gradient L2-norm at each logged iteration (useful for diagnosing LR).
    n_iter : int
        Actual number of iterations completed.
    converged : bool
        True if early stopping criterion was met.
    device : str
        PyTorch device that was used.
    config : MapConfig
        The configuration used to produce this result.
    elapsed_s : float
        Wall-clock time in seconds.
    """
    lambda_hr:         np.ndarray
    loss_history:      List[float]
    grad_norm_history: List[float]
    n_iter:            int
    converged:         bool
    device:            str
    config:            MapConfig
    elapsed_s:         float

    def summary(self) -> str:
        return "\n".join([
            "MapResult",
            "=" * 52,
            f"  HR shape      : {self.lambda_hr.shape}",
            f"  scale_factor  : {self.config.scale_factor}x",
            f"  mode          : {self.config.mode}",
            f"  iterations    : {self.n_iter} / {self.config.n_iter}",
            f"  converged     : {self.converged}",
            f"  final loss    : {self.loss_history[-1]:.4f}"
                          if self.loss_history else "  final loss    : n/a",
            f"  device        : {self.device}",
            f"  elapsed       : {self.elapsed_s:.1f} s",
            f"  lambda range  : [{float(self.lambda_hr.min()):.2f},"
                               f" {float(self.lambda_hr.max()):.2f}] ADU",
            "=" * 52,
        ])

    def save_fits(
        self,
        path:          str | Path,
        quality_map:   Optional[np.ndarray] = None,
        bayer_pattern: Optional[str]        = None,
    ) -> None:
        """
        Write the super-resolved scene to a FITS file.

        Parameters
        ----------
        path          : output .fits path
        quality_map   : optional [H, W] quality weight; upsampled to HR and
                        stored as a named extension.
        bayer_pattern : if set (e.g. 'RGGB'), the super-resolved Bayer mosaic
                        is split into a [4, sH//2, sW//2] data cube — one
                        plane per colour channel at native sub-pixel positions.
                        Channel labels are written to CHANn keywords.
                        For mono cameras leave as None (plain 2-D FITS).
        """
        from instrument_model_artifact import bayer_split
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        hdr = fits.Header()
        hdr["SCALE"]    = self.config.scale_factor, "SR upscaling factor"
        hdr["MODE"]     = self.config.mode,         "stacker mode"
        hdr["NITER"]    = self.n_iter,              "optimisation iterations"
        hdr["CONVERGE"] = self.converged,           "early stopping triggered"
        hdr["ALPHA_TV"] = self.config.alpha_tv,     "TV weight"
        hdr["ALPHA_KL"] = self.config.alpha_kl,     "KL weight"
        hdr["ALPHA_WV"] = self.config.alpha_wav,    "wavelet weight"
        hdr["ELAPSED"]  = round(self.elapsed_s, 1), "wall time seconds"
        hdr["COMMENT"]  = "Bayesian MAP super-resolved scene (Phase 4)"

        if bayer_pattern is not None:
            hdr["BAYERPAT"] = bayer_pattern
            lhr = self.lambda_hr
            if lhr.ndim == 3:
                # Already split — write the [4, sH, sW] cube directly.
                from instrument_model_artifact import _BAYER_OFFSETS
                names = list(_BAYER_OFFSETS[bayer_pattern].keys())
                for i, n in enumerate(names, 1):
                    hdr[f"CHAN{i}"] = n
                data = lhr
            else:
                planes, names = bayer_split(lhr, bayer_pattern)
                for i, n in enumerate(names, 1):
                    hdr[f"CHAN{i}"] = n
                data = planes
        else:
            data = self.lambda_hr

        primary = fits.PrimaryHDU(data.astype(np.float32), header=hdr)
        hdul    = fits.HDUList([primary])

        if quality_map is not None:
            qm = quality_map
            if qm.ndim == 3:
                # Already [4, H//2, W//2]; upsample each channel plane
                q_planes = np.stack([
                    _upsample_numpy(qm[c], self.config.scale_factor)
                    for c in range(qm.shape[0])
                ], axis=0)
                hdul.append(fits.ImageHDU(q_planes.astype(np.float32), name="QUALITY"))
            else:
                q_hr = _upsample_numpy(qm, self.config.scale_factor)
                if bayer_pattern is not None and q_hr.ndim == 2:
                    q_planes, _ = bayer_split(q_hr, bayer_pattern)
                    hdul.append(fits.ImageHDU(q_planes.astype(np.float32), name="QUALITY"))
                else:
                    hdul.append(fits.ImageHDU(q_hr.astype(np.float32), name="QUALITY"))

        hdul.writeto(str(path), overwrite=True)
        logger.info("MapResult saved to %s", path)

    def save_convergence_plot(self, path: str | Path) -> None:
        """Save loss curve as PNG.  Requires matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available — convergence plot skipped")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        iters = np.arange(1, len(self.loss_history) + 1) * self.config.log_every

        loss = np.array(self.loss_history, dtype=np.float64)
        if loss.min() <= 0:
            # Shift so all values are positive before log-scaling.
            # If all negative (typical for Poisson NLL), negate so the plot
            # shows decreasing loss as descending curve.
            if loss.max() <= 0:
                loss = -loss   # all negative → flip sign, curve goes down
            else:
                loss = loss - loss.min() + 1.0  # mixed → shift above zero
        ax1.semilogy(iters, loss, lw=1.5, color="steelblue")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss (log scale, negated)" if self.loss_history[0] < 0 else "Loss (log scale)")
        ax1.set_title("Convergence")
        ax1.grid(True, alpha=0.3)
        if self.converged:
            ax1.axvline(self.n_iter, color="red", ls="--", lw=1, label="converged")
            ax1.legend(fontsize=9)

        grad = np.array(self.grad_norm_history, dtype=np.float64)
        grad = np.where(grad > 0, grad, grad.max() * 1e-10 if grad.max() > 0 else 1e-10)
        ax2.semilogy(iters, grad, lw=1.5, color="darkorange")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Gradient L2-norm")
        ax2.set_title("Gradient norm")
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            f"MAP stacker — scale={self.config.scale_factor}x  "
            f"mode={self.config.mode}  iters={self.n_iter}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(str(path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Convergence plot saved to %s", path)


# ============================================================================
# Pure-NumPy reference helpers
# (used for testing and for the CPU fallback when torch is absent)
# ============================================================================

def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _upsample_numpy(arr: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour upsampling by integer factor."""
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _downsample_numpy(arr: np.ndarray, scale: int) -> np.ndarray:
    """Average-pool downsampling by integer factor."""
    H, W = arr.shape
    return arr.reshape(H // scale, scale, W // scale, scale).mean(axis=(1, 3))


def _convolve_psf_numpy(
    hr: np.ndarray,
    psf: np.ndarray,
) -> np.ndarray:
    """
    Convolve HR scene with PSF using reflection-padded FFT.

    Reflection padding creates a smooth periodic extension that eliminates
    the rectangular window / Gibbs ringing artefacts produced by zero-padding.
    The PSF kernel is still zero-padded (reflection makes no sense for a kernel).

    hr  : [sH, sW] float64
    psf : [K, K]   float64  (sum = 1)
    returns [sH, sW] float64
    """
    sH, sW = hr.shape
    K       = psf.shape[0]
    pH      = _next_power_of_2(sH + K)
    pW      = _next_power_of_2(sW + K)

    # Reflect-pad the scene to avoid edge discontinuities
    pad_h, pad_w = pH - sH, pW - sW
    hr_pad = np.pad(hr, ((0, pad_h), (0, pad_w)), mode='reflect')

    # Zero-pad the PSF kernel (reflection is meaningless for a kernel)
    psf_pad = np.zeros((pH, pW), dtype=np.float64)
    psf_cy, psf_cx = K // 2, K // 2
    psf_pad[:K, :K] = psf
    psf_pad = np.roll(np.roll(psf_pad, -psf_cy, axis=0), -psf_cx, axis=1)

    conv = np.real(np.fft.irfft2(
        np.fft.rfft2(hr_pad) * np.fft.rfft2(psf_pad),
        s=(pH, pW)
    ))
    return conv[:sH, :sW]


def _phase_shift_numpy(
    hr: np.ndarray,
    dx: float,
    dy: float,
    scale: int,
) -> np.ndarray:
    """
    Apply sub-pixel shift (dx, dy) in native-resolution pixels to HR array
    using Fourier phase-shift theorem.

    The shift is (dx/scale, dy/scale) in HR pixels.
    """
    from numpy.fft import rfft2, irfft2, rfftfreq, fftfreq
    sH, sW  = hr.shape
    dx_hr   = dx / scale
    dy_hr   = dy / scale
    fy      = fftfreq(sH).reshape(-1, 1)
    fx      = rfftfreq(sW).reshape(1, -1)
    phase   = np.exp(-2j * np.pi * (fy * dy_hr + fx * dx_hr))
    return np.real(irfft2(rfft2(hr) * phase, s=(sH, sW)))


def _tv_numpy(arr: np.ndarray, eps: float = 1e-8) -> float:
    """Isotropic total variation."""
    dx = arr[:, 1:] - arr[:, :-1]
    dy = arr[1:, :] - arr[:-1, :]
    return float(np.sqrt(dx[:arr.shape[0]-1, :]**2 +
                          dy[:, :arr.shape[1]-1]**2 + eps).sum())


def _kl_numpy(lam: np.ndarray, prior: np.ndarray, eps: float = 1e-12) -> float:
    """Poisson KL: Σ [λ·log(λ/λ₀) - λ + λ₀]"""
    l  = np.maximum(lam,   eps)
    l0 = np.maximum(prior, eps)
    return float((l * np.log(l / l0) - l + l0).sum())


def _haar_l1_numpy(arr: np.ndarray) -> float:
    """One-level Haar wavelet L1 detail norm."""
    H, W = arr.shape
    if H < 2 or W < 2:
        return 0.0
    # LL, LH, HL, HH via 2-D Haar transform
    a = arr.astype(np.float64)
    # Row transform
    lo_r = (a[:, 0::2] + a[:, 1::2]) / 2.0
    hi_r = (a[:, 0::2] - a[:, 1::2]) / 2.0
    # Column transform on low and high rows
    ll = (lo_r[0::2, :] + lo_r[1::2, :]) / 2.0
    lh = (lo_r[0::2, :] - lo_r[1::2, :]) / 2.0
    hl = (hi_r[0::2, :] + hi_r[1::2, :]) / 2.0
    hh = (hi_r[0::2, :] - hi_r[1::2, :]) / 2.0
    return float(np.abs(lh).sum() + np.abs(hl).sum() + np.abs(hh).sum())


def poisson_nll_numpy(
    observed: np.ndarray,
    rate:     np.ndarray,
    eps:      float = 1e-12,
) -> float:
    """
    Poisson negative log-likelihood:
        NLL = Σ [rate - observed·log(rate)]
    (constant log(observed!) term dropped)
    """
    r = np.maximum(rate, eps)
    return float((r - observed * np.log(r)).sum())


# ============================================================================
# Forward model (NumPy reference — mirrors the PyTorch version)
# ============================================================================

def forward_model_numpy(
    lambda_hr: np.ndarray,    # [sH, sW]
    psf:       np.ndarray,    # [K, K]
    dx:        float,
    dy:        float,
    scale:     int,
    transparency: float = 1.0,
    rotation_deg: float = 0.0,
    scale_ratio:  float = 1.0,
) -> np.ndarray:
    """
    Apply the full forward model to produce the expected LR observation.

        λ_lr = t · AvgPool(PSF ⊛ PhaseShift(AffineWarp(λ_hr), dx, dy))

    Parameters
    ----------
    lambda_hr    : super-resolved scene [S·H, S·W]
    psf          : per-frame PSF kernel [K, K]  sum=1
    dx, dy       : frame shift in native (LR) pixels
    scale        : upscaling factor
    transparency : per-frame throughput t_i
    rotation_deg : CCW rotation to apply to λ_hr before shift (spatial)
    scale_ratio  : uniform scale to apply to λ_hr before shift (spatial)

    Returns
    -------
    lambda_lr : [H, W] float64 — expected LR pixel values
    """
    # Rotation + scale warp (spatial, before frequency-domain phase shift)
    if abs(rotation_deg) > 1e-4 or abs(scale_ratio - 1.0) > 1e-5:
        from scipy.ndimage import affine_transform as _affine_transform
        s = 1.0 / scale_ratio
        r = math.radians(-rotation_deg)
        A = s * np.array([[math.cos(r), -math.sin(r)],
                           [math.sin(r),  math.cos(r)]])
        cy = (lambda_hr.shape[0] - 1) / 2.0
        cx = (lambda_hr.shape[1] - 1) / 2.0
        offset = np.array([cy, cx]) - A @ np.array([cy, cx])
        lambda_hr = _affine_transform(lambda_hr, A, offset=offset, order=1, cval=0.0)
    shifted     = _phase_shift_numpy(lambda_hr, dx, dy, scale)
    convolved   = _convolve_psf_numpy(shifted, psf)
    downsampled = _downsample_numpy(convolved, scale)
    return transparency * np.maximum(downsampled, 0.0)


# ============================================================================
# PyTorch forward model helpers
# ============================================================================

def _get_device(device: Optional[str]) -> "torch.device":
    if not _TORCH_OK:
        raise ImportError("PyTorch required for MAP stacker.  pip install torch")
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_psf_spectrum(
    psf: np.ndarray,
    pH:  int,
    pW:  int,
    dev: "torch.device",
) -> "torch.Tensor":
    """
    Pad a PSF kernel to (pH, pW) and return its rfft2 spectrum.
    The PSF is shifted so the centre aligns with (0,0) in the padded array.
    """
    K = psf.shape[0]
    psf_pad = np.zeros((pH, pW), dtype=np.float32)
    cy, cx  = K // 2, K // 2
    psf_pad[:K, :K] = psf
    # Roll centre to (0,0)
    psf_pad = np.roll(np.roll(psf_pad, -cy, axis=0), -cx, axis=1)
    t       = torch.from_numpy(psf_pad).to(dev)
    return torch.fft.rfft2(t, s=(pH, pW))


def _build_phase_ramp(
    dx:    float,
    dy:    float,
    sH:    int,
    sW:    int,
    scale: int,
    dev:   "torch.device",
) -> "torch.Tensor":
    """
    Return the rfft2 phase ramp for shift (dx/scale, dy/scale) in HR pixels.
    Shape: [sH, sW//2+1] complex.
    """
    dx_hr  = dx / scale
    dy_hr  = dy / scale
    fy     = torch.fft.fftfreq(sH, device=dev).reshape(-1, 1)
    fx     = torch.fft.rfftfreq(sW, device=dev).reshape(1, -1)
    return torch.exp(-2j * math.pi * (fy * dy_hr + fx * dx_hr))


def _build_affine_grid(
    rotation_deg: float,
    scale_ratio:  float,
    sH: int,
    sW: int,
    device: "torch.device",
) -> "Optional[torch.Tensor]":
    """
    Build a grid_sample flow field for rotation + uniform scale centred at
    image centre.  Returns [1, sH, sW, 2] float32 tensor, or None if the
    transform is effectively the identity.

    rotation_deg : CCW rotation to apply (positive = CCW)
    scale_ratio  : uniform scale factor to apply
    """
    if abs(rotation_deg) < 1e-4 and abs(scale_ratio - 1.0) < 1e-5:
        return None
    theta_rad = math.radians(rotation_deg)
    s = float(scale_ratio)
    # [2, 3] affine matrix in normalised [-1,1] coords (grid_sample convention)
    M = torch.tensor(
        [[s * math.cos(theta_rad), -s * math.sin(theta_rad), 0.0],
         [s * math.sin(theta_rad),  s * math.cos(theta_rad), 0.0]],
        dtype=torch.float32, device=device,
    ).unsqueeze(0)  # [1, 2, 3]
    grid = F.affine_grid(M, size=(1, 1, sH, sW), align_corners=False)  # [1,sH,sW,2]
    return grid


def _apply_affine_grid(
    lam:  "torch.Tensor",           # [sH, sW]
    grid: "Optional[torch.Tensor]", # [1, sH, sW, 2] or None
) -> "torch.Tensor":
    """Apply a precomputed affine grid to a [sH, sW] tensor. Returns [sH, sW]."""
    if grid is None:
        return lam
    warped = F.grid_sample(
        lam.unsqueeze(0).unsqueeze(0),  # [1, 1, sH, sW]
        grid,                            # [1, sH, sW, 2]
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    )
    return warped.squeeze(0).squeeze(0)  # [sH, sW]


def _forward_single(
    lam:         "torch.Tensor",   # [sH, sW]  λ_hr (already softplus-transformed)
    psf_spec:    "torch.Tensor",   # rfft2 spectrum of padded PSF
    phase_ramp:  "torch.Tensor",   # rfft2 phase ramp
    scale:       int,
    transparency: float,
    pH: int,
    pW: int,
    affine_grid: "Optional[torch.Tensor]" = None,  # rotation+scale grid or None
) -> "torch.Tensor":
    """
    Single-frame forward model in PyTorch.

    λ_hr → AffineWarp(rot,scale) → PhaseShift(freq) → PSF conv → downsample → t·λ_lr

    Caller is responsible for the softplus(theta) → lam transform.
    Rotation+scale are applied spatially (grid_sample) before the FFT.
    Translation remains in frequency domain (phase ramp) — exact and alias-free.
    """
    # Rotation + scale warp (spatial, before FFT).  No-op when affine_grid is None.
    lam = _apply_affine_grid(lam, affine_grid)                # [sH, sW]

    sH, sW = lam.shape

    # Reflection-pad to (pH, pW) before FFT to eliminate the rectangular
    # window / Gibbs ringing that zero-padding would introduce.
    # Only the scene is reflection-padded; the PSF spectrum was built with
    # zero-padding (_make_psf_spectrum) which is correct for a kernel.
    # F.pad with mode='reflect' and 4 pad values requires >= 3D input.
    lam_padded = F.pad(
        lam.unsqueeze(0).unsqueeze(0),          # [1, 1, sH, sW]
        (0, pW - sW, 0, pH - sH),
        mode='reflect',
    ).squeeze(0).squeeze(0)                      # [pH, pW]
    lam_f  = torch.fft.rfft2(lam_padded)                    # [pH, pW//2+1]

    # Phase-shift (sub-pixel translation in LR pixels → HR pixel units)
    lam_f  = lam_f * phase_ramp                              # element-wise

    # PSF convolution
    lam_f  = lam_f * psf_spec

    # Inverse FFT and crop to HR size
    lam_conv = torch.fft.irfft2(lam_f, s=(pH, pW))[:sH, :sW]  # [sH, sW]

    # Average-pool downsample to LR
    lam_lr = F.avg_pool2d(
        lam_conv.unsqueeze(0).unsqueeze(0),                  # [1,1,sH,sW]
        kernel_size=scale,
        stride=scale,
    ).squeeze()                                               # [H, W]

    return transparency * torch.clamp(lam_lr, min=0.0)


def _poisson_nll_torch(
    observed: "torch.Tensor",
    rate:     "torch.Tensor",
    eps:      float = 1e-12,
) -> "torch.Tensor":
    """Poisson NLL: Σ [rate - observed·log(rate+eps)]"""
    return (rate - observed * torch.log(rate + eps)).sum()


def _tv_torch(lam: "torch.Tensor", eps: float = 1e-8) -> "torch.Tensor":
    """
    Truly isotropic total variation using four gradient directions.

    The standard 2-term formulation (dx, dy only) assigns diagonal gradients
    a sqrt(2) higher cost than axis-aligned ones, which biases the optimizer
    toward axis-aligned features and reinforces cross-pattern artefacts.
    Adding diagonal difference terms with 1/sqrt(2) scaling equalises all
    directions.
    """
    dx   = lam[:, 1:]  - lam[:, :-1]           # [H,   W-1]
    dy   = lam[1:, :]  - lam[:-1, :]           # [H-1, W  ]
    dxy  = (lam[1:, 1:]  - lam[:-1, :-1]) * 0.70711   # [H-1, W-1] / sqrt(2)
    dyx  = (lam[1:, :-1] - lam[:-1, 1:])  * 0.70711   # [H-1, W-1] / sqrt(2)
    H, W = lam.shape
    return torch.sqrt(
        dx[:H-1, :]**2 + dy[:, :W-1]**2 + dxy**2 + dyx**2 + eps
    ).sum()


def _kl_torch(
    lam:   "torch.Tensor",
    prior: "torch.Tensor",
    eps:   float = 1e-12,
) -> "torch.Tensor":
    """Poisson KL: Σ [λ·log(λ/λ₀) - λ + λ₀]"""
    l  = torch.clamp(lam,   min=eps)
    l0 = torch.clamp(prior, min=eps)
    return (l * torch.log(l / l0) - l + l0).sum()


def _haar_l1_torch(lam: "torch.Tensor") -> "torch.Tensor":
    """One-level Haar wavelet L1 detail coefficients."""
    a = lam
    if a.shape[0] < 2 or a.shape[1] < 2:
        return torch.tensor(0.0, device=lam.device)

    # Row Haar: split even/odd columns
    lo_r = (a[:, 0::2] + a[:, 1::2]) * 0.5
    hi_r = (a[:, 0::2] - a[:, 1::2]) * 0.5

    # Column Haar
    lh = (lo_r[0::2, :] - lo_r[1::2, :]) * 0.5    # horizontal detail
    hl = (hi_r[0::2, :] + hi_r[1::2, :]) * 0.5    # vertical detail
    hh = (hi_r[0::2, :] - hi_r[1::2, :]) * 0.5    # diagonal detail

    return lh.abs().sum() + hl.abs().sum() + hh.abs().sum()


# ============================================================================
# Initialisation helpers
# ============================================================================

def _build_prior(
    weighted_mean_lr: np.ndarray,
    scale: int,
) -> np.ndarray:
    """
    Upsample Phase 3 weighted_mean to HR grid using bilinear interpolation.
    Returns float32 [S·H, S·W].
    """
    from scipy.ndimage import zoom
    # zoom with order=1 = bilinear
    hr = zoom(weighted_mean_lr.astype(np.float64), scale, order=1)
    return np.maximum(hr, 0.0).astype(np.float32)


def _theta_init(prior_hr: np.ndarray) -> np.ndarray:
    """
    Inverse softplus of the prior: θ = log(exp(λ) - 1).
    Initialises θ so that softplus(θ) ≈ prior_hr.
    """
    lam   = np.maximum(prior_hr, 1e-3)
    # Stable inverse softplus: log(exp(x)-1) = x + log(1 - exp(-x))
    # For large x: ≈ x; for small x: ≈ log(x)
    theta = lam + np.log(-np.expm1(-lam))
    return theta.astype(np.float32)


# ============================================================================
# FAST mode solver — multi-channel (OSC) helper
# ============================================================================

def _solve_fast_multichannel(
    stats:  "SufficientStats",
    config: MapConfig,
    dev:    "torch.device",
) -> "MapResult":
    """
    Run _solve_fast independently for each Bayer channel and stack results.

    stats.weighted_sum is [4, H//2, W//2].  A proxy SufficientStats with a
    single channel is constructed for each channel so that the existing
    _solve_fast() mono code path is reused without modification.
    """
    n_channels = stats.weighted_sum.shape[0]
    channel_results: List[np.ndarray] = []
    t0 = time.time()

    # Aggregate loss/grad histories across channels
    all_loss: List[float] = []
    all_grad: List[float] = []
    last_n_iter = 0
    last_converged = False

    for c in range(n_channels):
        logger.info("FAST multi-channel: solving channel %d/%d", c + 1, n_channels)

        # Build a mono-equivalent SufficientStats for this channel
        ch_stats = SufficientStats(
            weighted_sum      = stats.weighted_sum[c].astype(np.float32),
            weight_sum        = stats.weight_sum[c].astype(np.float32),
            sky_sum           = stats.sky_sum[c].astype(np.float32),
            sq_sum            = stats.sq_sum[c].astype(np.float32),
            frame_count       = stats.frame_count,
            shift_list        = stats.shift_list,
            psf_list          = stats.psf_list,
            transparency_list = stats.transparency_list,
            fwhm_list         = stats.fwhm_list,
            frame_shape       = stats.frame_shape,  # (H//2, W//2)
        )
        ch_result = _solve_fast(ch_stats, config, dev)
        channel_results.append(ch_result.lambda_hr)
        all_loss.extend(ch_result.loss_history)
        all_grad.extend(ch_result.grad_norm_history)
        last_n_iter     = ch_result.n_iter
        last_converged  = ch_result.converged

    lambda_hr_4ch = np.stack(channel_results, axis=0)   # [4, sH, sW]

    return MapResult(
        lambda_hr         = lambda_hr_4ch,
        loss_history      = all_loss,
        grad_norm_history = all_grad,
        n_iter            = last_n_iter,
        converged         = last_converged,
        device            = str(dev),
        config            = config,
        elapsed_s         = time.time() - t0,
    )


# ============================================================================
# FAST mode solver
# ============================================================================

def _solve_fast(
    stats:  "SufficientStats",
    config: MapConfig,
    dev:    "torch.device",
) -> MapResult:
    """
    FAST mode: treats the Phase 3 weighted_sum / weight_sum as a single
    synthetic super-exposure and optimises on it.

    For OSC cameras the accumulated statistics are [4, H//2, W//2].
    Each Bayer channel is solved independently and the results are stacked
    into a [4, sH, sW] array in MapResult.lambda_hr.
    """
    # --- Multi-channel (OSC) dispatch ---
    if stats.weighted_sum.ndim == 3:
        return _solve_fast_multichannel(stats, config, dev)

    H, W   = stats.frame_shape
    S      = config.scale_factor
    sH, sW = H * S, W * S

    N = stats.frame_count

    # ---- Observed proxy: weighted_sum ----------------------------------------
    # obs_t is the t_i²-weighted sum of frames (see sufficient_statistics.py).
    # The forward model predicts: Σ t_i² · D(PSF_i ⊛ AffineWarp_i(PhaseShift_i(λ_hr)))
    obs_lr  = stats.weighted_sum.astype(np.float32)          # [H, W]

    # ---- FFT padding: use max PSF size across frames -----------------------
    psf_max = max(p.shape[0] for p in stats.psf_list)
    pH = _next_power_of_2(sH + psf_max)
    pW = _next_power_of_2(sW + psf_max)

    # ---- Per-frame caches (precomputed once, before the optimisation loop) -
    # Batch PSF spectra and phase ramps as [N, pH, pW//2+1] tensors so the
    # entire forward model is a single batched rfft2 — no Python loop at all.
    t_list: List[float] = stats.transparency_list
    t_sq = torch.tensor([t*t for t in t_list], dtype=torch.float32, device=dev)  # [N]

    psf_stack  = []
    ramp_stack = []
    affine_grids: List = []

    for i in range(N):
        psf_i = stats.psf_list[i].astype(np.float32)
        psf_i /= psf_i.sum()
        psf_stack.append(_make_psf_spectrum(psf_i, pH, pW, dev))   # [pH, pW//2+1]

        sh = stats.shift_list[i]
        dx_i  = sh.dx_px       if sh is not None else 0.0
        dy_i  = sh.dy_px       if sh is not None else 0.0
        rot_i = -(sh.rotation_deg  if sh is not None else 0.0)
        sc_i  = 1.0 / (sh.scale_ratio if (sh is not None and sh.scale_ratio > 0) else 1.0)
        ramp_stack.append(_build_phase_ramp(dx_i, dy_i, pH, pW, S, dev))
        affine_grids.append(_build_affine_grid(rot_i, sc_i, sH, sW, dev))

    # [N, pH, pW//2+1] — combined filter H_i = PSF_i × PhaseRamp_i
    # Precomputed once; reused every optimisation step.
    # Per step we only need: rfft2(lam_hr_warped_i) × H_i
    psf_batch  = torch.stack(psf_stack,  dim=0)   # complex64 [N, pH, pW//2+1]
    ramp_batch = torch.stack(ramp_stack, dim=0)   # complex64 [N, pH, pW//2+1]
    H_batch    = psf_batch * ramp_batch            # combined filter — ONE multiply, done once

    # Frames that need spatial affine warp (rotation/scale != identity)
    warp_indices = [i for i, g in enumerate(affine_grids) if g is not None]

    # ---- Initialise λ on HR grid -------------------------------------------
    prior_hr = _build_prior(stats.weighted_mean, S)          # [sH, sW]
    theta_np = _theta_init(prior_hr)                         # [sH, sW]

    theta   = torch.tensor(theta_np, device=dev, requires_grad=True)
    obs_t   = torch.tensor(obs_lr,   device=dev)
    prior_t = torch.tensor(prior_hr, device=dev)

    # ---- Optimiser + scheduler --------------------------------------------
    opt  = torch.optim.Adam([theta], lr=config.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.n_iter, eta_min=config.lr_min
    )

    # ---- Optimisation loop ------------------------------------------------
    loss_history: List[float]      = []
    grad_history: List[float]      = []
    stall_count:  int              = 0
    prev_loss:    Optional[float]  = None
    converged     = False
    t0            = time.time()

    for step in range(1, config.n_iter + 1):
        opt.zero_grad()

        lam_hr   = F.softplus(theta)                                # [sH, sW]

        # Mini-batched forward model: process config.forward_batch_size frames
        # at a time, accumulate predicted_sum. Keeps peak RAM to
        # batch_size × FFT_tensor_size regardless of total N.
        predicted_sum = torch.zeros_like(obs_t)
        bs = config.forward_batch_size
        for b_start in range(0, N, bs):
            b_end = min(b_start + bs, N)
            idx   = list(range(b_start, b_end))
            B     = len(idx)

            # Build [B, sH, sW] — apply affine warp where needed
            frames = []
            for i in idx:
                f = _apply_affine_grid(lam_hr, affine_grids[i]) if affine_grids[i] is not None else lam_hr
                frames.append(f)
            lam_batch = torch.stack(frames, dim=0)              # [B, sH, sW]

            # Pad + rfft2 — one batched call
            lam_padded = F.pad(
                lam_batch.unsqueeze(1),                         # [B, 1, sH, sW]
                (0, pW - sW, 0, pH - sH),
                mode='reflect',
            ).squeeze(1)                                         # [B, pH, pW]
            lam_f = torch.fft.rfft2(lam_padded)                 # [B, pH, pW//2+1]

            # Phase-shift + PSF conv — single multiply with precomputed H_i
            lam_f = lam_f * H_batch[b_start:b_end]

            # irfft2, crop, downsample, weight and accumulate
            lam_conv = torch.fft.irfft2(lam_f, s=(pH, pW))[:, :sH, :sW]   # [B, sH, sW]
            lam_lr   = F.avg_pool2d(
                lam_conv.unsqueeze(1), kernel_size=S, stride=S,
            ).squeeze(1)                                         # [B, H, W]
            lam_lr   = torch.clamp(lam_lr, min=0.0)

            predicted_sum = predicted_sum + (t_sq[b_start:b_end, None, None] * lam_lr).sum(dim=0)

        # ---- Data loss
        loss = _poisson_nll_torch(obs_t, predicted_sum)

        # ---- TV
        if config.alpha_tv > 0:
            loss = loss + config.alpha_tv * _tv_torch(lam_hr, config.tv_eps)

        # ---- KL from prior
        if config.alpha_kl > 0:
            loss = loss + config.alpha_kl * _kl_torch(lam_hr, prior_t)

        # ---- Haar wavelet
        if config.alpha_wav > 0:
            loss = loss + config.alpha_wav * _haar_l1_torch(lam_hr)

        loss.backward()
        opt.step()
        sched.step()

        # ---- Logging & convergence ----------------------------------------
        if step % config.log_every == 0 or step == 1:
            l_val = float(loss.item())
            g_val = float(theta.grad.norm().item()) if theta.grad is not None else 0.0
            loss_history.append(l_val)
            grad_history.append(g_val)
            logger.info(
                "FAST  step %4d/%d  loss=%.4e  |grad|=%.3e  lr=%.2e",
                step, config.n_iter, l_val, g_val, sched.get_last_lr()[0],
            )

            # Check convergence
            if prev_loss is not None:
                rel = abs(l_val - prev_loss) / (abs(prev_loss) + 1e-12)
                stall_count = stall_count + 1 if rel < config.rel_tol else 0
                if stall_count >= config.patience:
                    logger.info("Early stop at step %d (rel_tol met)", step)
                    converged = True
                    break
            prev_loss = l_val

    # ---- Extract result ---------------------------------------------------
    with torch.no_grad():
        lam_hr_np = F.softplus(theta).cpu().numpy().astype(np.float32)

    return MapResult(
        lambda_hr         = lam_hr_np,
        loss_history      = loss_history,
        grad_norm_history = grad_history,
        n_iter            = step,
        converged         = converged,
        device            = str(dev),
        config            = config,
        elapsed_s         = time.time() - t0,
    )


# ============================================================================
# EXACT mode solver
# ============================================================================

def _solve_exact(
    stats:      "SufficientStats",
    fits_paths: List[Path],
    config:     MapConfig,
    dev:        "torch.device",
) -> MapResult:
    """
    EXACT mode: loads raw FITS frames in mini-batches and computes the
    unbiased Poisson NLL over each batch.

    Requires fits_paths — one FITS path per frame in stats.psf_list order.
    Also requires a fitted InstrumentModel (loaded from the first frame's
    neighbour .h5 or passed separately via model= kwarg of solve()).
    """
    H, W   = stats.frame_shape
    S      = config.scale_factor
    sH, sW = H * S, W * S
    N      = stats.frame_count

    if len(fits_paths) != N:
        raise ValueError(
            f"fits_paths length ({len(fits_paths)}) != frame_count ({N})"
        )

    # ---- Initialise -------------------------------------------------------
    prior_hr = _build_prior(stats.weighted_mean, S)
    theta_np = _theta_init(prior_hr)
    theta    = torch.tensor(theta_np,  device=dev, requires_grad=True)
    prior_t  = torch.tensor(prior_hr,  device=dev)

    psf_specs: List["torch.Tensor"]   = []
    phase_ramps: List["torch.Tensor"] = []
    affine_grids: List                = []

    # All PSFs use the same padding for simplicity — use the max
    psf_max = max(p.shape[0] for p in stats.psf_list)
    pH = _next_power_of_2(sH + psf_max)
    pW = _next_power_of_2(sW + psf_max)

    for i in range(N):
        psf = stats.psf_list[i].astype(np.float32)
        psf /= psf.sum()
        psf_specs.append(_make_psf_spectrum(psf, pH, pW, dev))

        sh = stats.shift_list[i]
        dx  = sh.dx_px        if sh is not None else 0.0
        dy  = sh.dy_px        if sh is not None else 0.0
        rot = -(sh.rotation_deg  if sh is not None else 0.0)
        sc  = 1.0 / (sh.scale_ratio if (sh is not None and sh.scale_ratio > 0) else 1.0)
        phase_ramps.append(_build_phase_ramp(dx, dy, pH, pW, S, dev))
        affine_grids.append(_build_affine_grid(rot, sc, sH, sW, dev))

    opt   = torch.optim.Adam([theta], lr=config.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.n_iter, eta_min=config.lr_min
    )

    rng_state = np.random.default_rng(config.seed)

    loss_history: List[float]      = []
    grad_history: List[float]      = []
    stall_count:  int              = 0
    prev_loss:    Optional[float]  = None
    converged     = False
    t0            = time.time()

    for step in range(1, config.n_iter + 1):
        opt.zero_grad()

        # Mini-batch of frame indices
        idx = rng_state.choice(N, size=min(config.batch_size, N), replace=False)

        batch_loss = torch.tensor(0.0, device=dev)
        lam_hr_cur = F.softplus(theta)

        for i in idx.tolist():
            # Load raw frame
            with fits.open(fits_paths[i], memmap=False) as hdul:
                raw_np = hdul[0].data.astype(np.float32).squeeze()
                exp_s  = float(hdul[0].header.get("EXPTIME", 300.0))

            # Sky-subtract (use Phase 3 sky_mean as proxy)
            sky_lr = stats.sky_sum.astype(np.float32) / max(stats.frame_count, 1)
            obs_np = (raw_np - sky_lr).astype(np.float32)
            obs_t  = torch.tensor(obs_np, device=dev)

            t_i = stats.transparency_list[i]

            # Forward model
            lam_lr = _forward_single(
                lam_hr_cur, psf_specs[i], phase_ramps[i], S, t_i, pH, pW,
                affine_grid=affine_grids[i],
            )

            batch_loss = batch_loss + _poisson_nll_torch(obs_t, lam_lr)

        # Scale loss to full dataset
        scale_factor_loss = N / len(idx)
        total_loss = batch_loss * scale_factor_loss

        if config.alpha_tv > 0:
            total_loss = total_loss + config.alpha_tv * _tv_torch(lam_hr_cur, config.tv_eps)

        if config.alpha_kl > 0:
            total_loss = total_loss + config.alpha_kl * _kl_torch(lam_hr_cur, prior_t)

        if config.alpha_wav > 0:
            total_loss = total_loss + config.alpha_wav * _haar_l1_torch(lam_hr_cur)

        total_loss.backward()
        opt.step()
        sched.step()

        if step % config.log_every == 0 or step == 1:
            l_val = float(total_loss.item())
            g_val = float(theta.grad.norm().item()) if theta.grad is not None else 0.0
            loss_history.append(l_val)
            grad_history.append(g_val)
            logger.info(
                "EXACT step %4d/%d  loss=%.4e  |grad|=%.3e  batch=%d",
                step, config.n_iter, l_val, g_val, len(idx),
            )
            if prev_loss is not None:
                rel = abs(l_val - prev_loss) / (abs(prev_loss) + 1e-12)
                stall_count = stall_count + 1 if rel < config.rel_tol else 0
                if stall_count >= config.patience:
                    logger.info("Early stop at step %d", step)
                    converged = True
                    break
            prev_loss = l_val

    with torch.no_grad():
        lam_hr_np = F.softplus(theta).cpu().numpy().astype(np.float32)

    return MapResult(
        lambda_hr         = lam_hr_np,
        loss_history      = loss_history,
        grad_norm_history = grad_history,
        n_iter            = step,
        converged         = converged,
        device            = str(dev),
        config            = config,
        elapsed_s         = time.time() - t0,
    )


# ============================================================================
# Public API
# ============================================================================

def solve(
    stats:      "SufficientStats",
    config:     Optional[MapConfig] = None,
    fits_paths: Optional[List[Path]] = None,
) -> MapResult:
    """
    Run the MAP super-resolution optimisation.

    Parameters
    ----------
    stats : SufficientStats
        Output of Phase 3 accumulation.  Must be fully populated.
    config : MapConfig
        Hyperparameters.  Uses sensible defaults if None.
    fits_paths : list[Path] | None
        Required only for config.mode == 'exact'.  One FITS path per
        frame in the same order as stats.psf_list.

    Returns
    -------
    MapResult

    Examples
    --------
    Fast mode (default):

        from sufficient_statistics import SufficientStats
        from map_stacker import solve, MapConfig

        stats  = SufficientStats.load("sufficient_stats.h5")
        result = solve(stats)
        result.save_fits("lambda_hr.fits")
        result.save_convergence_plot("convergence.png")

    Exact mode:

        from pathlib import Path
        config = MapConfig(mode='exact', n_iter=200, batch_size=10)
        paths  = sorted(Path("lights/").glob("*.fits"))
        result = solve(stats, config=config, fits_paths=paths)
    """
    if not _TORCH_OK:
        raise ImportError(
            "PyTorch is required for the MAP stacker.\n"
            "Install: pip install torch\n"
            "See https://pytorch.org/get-started for GPU builds."
        )

    if config is None:
        config = MapConfig()
    config.validate()

    if config.mode == 'exact' and fits_paths is None:
        raise ValueError("config.mode='exact' requires fits_paths to be provided")

    dev = _get_device(config.device)
    logger.info(
        "MAP stacker: mode=%s  scale=%dx  device=%s  n_iter=%d",
        config.mode, config.scale_factor, dev, config.n_iter,
    )

    if config.mode == 'fast':
        return _solve_fast(stats, config, dev)
    else:
        return _solve_exact(stats, [Path(p) for p in fits_paths], config, dev)


# ============================================================================
# Convenience: run full pipeline from a stats file
# ============================================================================

def run_from_stats_file(
    stats_path:  str | Path,
    output_dir:  str | Path,
    config:      Optional[MapConfig] = None,
    fits_paths:  Optional[List[Path]] = None,
) -> MapResult:
    """
    Load SufficientStats from HDF5, run MAP stacker, write outputs.

    Outputs written:
        {output_dir}/lambda_hr.fits
        {output_dir}/convergence.png   (if matplotlib available)

    Parameters
    ----------
    stats_path : path to sufficient_stats.h5
    output_dir : directory for output files
    config     : MapConfig (defaults if None)
    fits_paths : required for exact mode
    """
    if not _SS_OK:
        raise ImportError("sufficient_statistics.py required")

    stats_path = Path(stats_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SufficientStats from %s", stats_path)
    stats  = SufficientStats.load(stats_path)

    result = solve(stats, config=config, fits_paths=fits_paths)
    logger.info(result.summary())

    result.save_fits(
        output_dir / "lambda_hr.fits",
        quality_map=stats.quality_map,
    )
    result.save_convergence_plot(output_dir / "convergence.png")

    return result
