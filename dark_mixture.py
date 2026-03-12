"""
dark_mixture.py
===============
Three-component Gamma mixture model for classifying pixels by dark current
behaviour: normal / hot / cold.

Fitted by Expectation-Maximisation (EM) on a per-pixel dark rate map
[ADU/s] produced by InstrumentModel.fit_dark().

Physical motivation
-------------------
Dark current across a CMOS sensor is not homogeneous:

  Normal pixels  — thermal dark current, follows the sensor mean.  The
                   vast majority of pixels (typically > 99.5%).

  Hot pixels     — elevated dark current from lattice defects, surface
                   trapping sites, or radiation damage.  Bright even in
                   short exposures.  Rate can be 10–1000× the normal mean.

  Cold pixels    — anomalously low dark current.  Less common than hot
                   pixels; can indicate trapping that suppresses dark
                   generation.

Each class is modelled as a Gamma distribution on the dark rate λ^(p):

  p(λ^(p)) = π_n · Γ(α_n, β_n) + π_h · Γ(α_h, β_h) + π_c · Γ(α_c, β_c)

  E[λ | normal] = α_n / β_n  ≈ sensor mean dark rate
  E[λ | hot]    = α_h / β_h  >> sensor mean
  E[λ | cold]   = α_c / β_c  << sensor mean

The EM algorithm alternates:
  E-step — compute per-pixel class responsibility r^(p)_k
            using the current component parameters
  M-step — update (π_k, α_k, β_k) from weighted sufficient statistics
            using the Choi-Wette approximation for α (shape MLE)

Output
------
  DarkMixtureModel.class_probs [H, W, 3]  — soft class membership
      [:, :, 0] = p_normal
      [:, :, 1] = p_hot
      [:, :, 2] = p_cold

  This replaces the hard hot_pixel_mask from InstrumentModel with a
  continuous-valued weight that flows naturally into the MAP stacker and
  into calibrate_frame() for pixel interpolation.

Calibrated frame outputs
------------------------
  DarkMixtureModel.calibrate_frame() returns two things:
    1. interpolated frame  — hot/cold pixels replaced by weighted
       neighbour average, for display and simple stacking
    2. pixel weight map    — [H, W] values in [0, 1]:
       1.0 = normal pixel (trust fully)
       0.0 = flagged defect (exclude from stacker)
       intermediate = soft weighting

HDF5 layout (under /dark_mixture/ group)
-----------------------------------------
  /dark_mixture/class_probs    [H, W, 3]  float32
  /dark_mixture/pi             [3]         float64  mixing weights
  /dark_mixture/alpha          [3]         float64  Gamma shape per component
  /dark_mixture/beta           [3]         float64  Gamma rate per component
  /dark_mixture/n_iterations   scalar int
  /dark_mixture/log_likelihood_curve  [n_iterations]  float64
  /dark_mixture/labels         [H, W]      uint8   hard MAP label

Dependencies
------------
  numpy  scipy.special (digamma, gammaln)  h5py

No PyTorch required — the EM runs on CPU; the dark_rate map is small
(9 Mpx × float32 = 36 MB for ASI533) and EM converges in < 30 iterations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
from scipy.special import digamma, gammaln

logger = logging.getLogger(__name__)

# Component indices
_NORMAL = 0
_HOT    = 1
_COLD   = 2

# ============================================================================
# Choi-Wette shape parameter MLE
# ============================================================================

def _gamma_shape_mle(
    weighted_mean_log: np.ndarray,   # E[log λ] weighted by responsibilities
    weighted_log_mean: np.ndarray,   # log(E[λ]) weighted by responsibilities
    max_iter: int = 20,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Estimate Gamma shape parameter α via the Choi-Wette (1994) approximation
    followed by Newton-Raphson refinement.

    For a weighted Gamma MLE with sufficient statistics:
      s = log(E[λ]) - E[log λ]     (always > 0)
      Initial estimate: α₀ = (3 - s + √((s-3)² + 24s)) / (12s)
      Newton step:      α ← α - (log α - ψ(α) - s) / (1/α - ψ'(α))

    Parameters
    ----------
    weighted_mean_log : array
        Weighted mean of log(λ) for each component.
    weighted_log_mean : array
        log of the weighted mean of λ for each component.

    Returns
    -------
    alpha : array  shape MLE
    """
    s = weighted_log_mean - weighted_mean_log   # always >= 0 by Jensen
    s = np.maximum(s, 1e-8)

    # Choi-Wette initial estimate
    alpha = (3.0 - s + np.sqrt((s - 3.0)**2 + 24.0 * s)) / (12.0 * s)
    alpha = np.maximum(alpha, 1e-6)

    # Newton-Raphson refinement
    for _ in range(max_iter):
        g   = np.log(alpha) - digamma(alpha) - s
        gp  = 1.0 / alpha - _trigamma(alpha)
        step = g / gp
        alpha = alpha - step
        alpha = np.maximum(alpha, 1e-6)
        if np.max(np.abs(step)) < tol:
            break

    return alpha


def _trigamma(x: np.ndarray) -> np.ndarray:
    """Trigamma function ψ'(x) = d²/dx² log Γ(x) via series approximation."""
    # Use the recurrence ψ'(x) = ψ'(x+1) + 1/x² and asymptotic for large x
    result = np.zeros_like(x, dtype=np.float64)
    xx = x.copy()

    # Shift small values up via recurrence
    mask = xx < 6.0
    shift = np.zeros_like(xx)
    while np.any(mask):
        result[mask] += 1.0 / (xx[mask] ** 2)
        xx[mask]     += 1.0
        shift[mask]  += 1.0
        mask = xx < 6.0

    # Asymptotic expansion for large x
    inv_x  = 1.0 / xx
    inv_x2 = inv_x ** 2
    result += inv_x + inv_x2/2 + inv_x2*inv_x/6 - inv_x2*inv_x2*inv_x/30

    return result


# ============================================================================
# Mixture model dataclass
# ============================================================================

@dataclass
class DarkMixtureModel:
    """
    Fitted three-component Gamma mixture for dark pixel classification.

    Attributes
    ----------
    pi : np.ndarray, shape (3,)
        Mixing weights [π_normal, π_hot, π_cold].  Sum to 1.

    alpha : np.ndarray, shape (3,)
        Gamma shape parameters [α_normal, α_hot, α_cold].

    beta : np.ndarray, shape (3,)
        Gamma rate parameters [β_normal, β_hot, β_cold].
        E[λ | k] = alpha[k] / beta[k].

    class_probs : np.ndarray, shape (H, W, 3), float32
        Per-pixel soft class membership.
        [:, :, 0] p_normal,  [:, :, 1] p_hot,  [:, :, 2] p_cold.

    labels : np.ndarray, shape (H, W), uint8
        Hard MAP class label (argmax of class_probs).

    log_likelihood_curve : list[float]
        Complete data log-likelihood at each EM iteration.
        Should be monotonically non-decreasing.

    normal_threshold_sigma : float
        Pixels with p_normal < norm_cdf(-normal_threshold_sigma) are
        considered defective for the weight map.  Default 3.0.
    """
    pi:                     np.ndarray
    alpha:                  np.ndarray
    beta:                   np.ndarray
    class_probs:            np.ndarray           # [H, W, 3] float32
    labels:                 np.ndarray           # [H, W] uint8
    log_likelihood_curve:   List[float]          = field(default_factory=list)
    normal_threshold:       float                = 0.5   # min p_normal to be "good"

    # ---- Component properties ---------------------------------------------

    @property
    def component_means(self) -> np.ndarray:
        """E[λ | k] = α/β for each component."""
        return self.alpha / self.beta

    @property
    def n_hot(self) -> int:
        """Number of pixels classified as hot (MAP label = 1)."""
        return int((self.labels == _HOT).sum())

    @property
    def n_cold(self) -> int:
        """Number of pixels classified as cold (MAP label = 2)."""
        return int((self.labels == _COLD).sum())

    @property
    def n_normal(self) -> int:
        return int((self.labels == _NORMAL).sum())

    # ---- Pixel weight map -------------------------------------------------

    def pixel_weight_map(self) -> np.ndarray:
        """
        Continuous per-pixel reliability weight in [0, 1].

        Weight = p_normal (posterior probability of belonging to the normal
        dark current component).  A pixel with p_normal = 0.98 is weighted
        0.98 in the stacker; a hot pixel with p_normal = 0.01 contributes
        almost nothing.

        This is the weight map for the MAP stacker — do NOT hard-threshold.
        """
        return self.class_probs[:, :, _NORMAL].astype(np.float32)

    # ---- Frame calibration ------------------------------------------------

    def calibrate_frame(
        self,
        frame: np.ndarray,
        interp_radius: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply dark pixel correction to a calibrated light frame.

        Returns
        -------
        interpolated : np.ndarray [H, W] float32
            Frame with hot/cold pixels replaced by a weighted local median
            of their neighbours.  Neighbour weights are inversely proportional
            to their own p_defect, so other defect pixels do not corrupt the
            interpolation.  Suitable for display and simple stacking.

        weight_map : np.ndarray [H, W] float32
            Per-pixel reliability weight = p_normal.  Use this in the MAP
            stacker for principled soft exclusion of defective pixels.
        """
        frame_f32  = frame.astype(np.float32)
        weight_map = self.pixel_weight_map()
        interp     = frame_f32.copy()

        # Identify defect pixels requiring interpolation
        defect_mask = weight_map < self.normal_threshold
        if not np.any(defect_mask):
            return interp, weight_map

        H, W = frame_f32.shape
        r    = interp_radius

        ys, xs = np.where(defect_mask)
        for y, x in zip(ys, xs):
            # Extract neighbourhood
            y0, y1 = max(0, y - r), min(H, y + r + 1)
            x0, x1 = max(0, x - r), min(W, x + r + 1)

            nbr_vals    = frame_f32[y0:y1, x0:x1].ravel()
            nbr_weights = weight_map[y0:y1, x0:x1].ravel()

            # Exclude the defect pixel itself
            # (centre of the patch, which may be inside due to boundary clipping)
            cy = y - y0
            cx = x - x0
            centre_idx = cy * (x1 - x0) + cx
            nbr_weights = nbr_weights.copy()
            nbr_weights[centre_idx] = 0.0

            total_w = nbr_weights.sum()
            if total_w > 0:
                interp[y, x] = np.dot(nbr_vals, nbr_weights) / total_w
            # If no valid neighbours (shouldn't happen for r≥1), leave as-is

        logger.debug(
            "calibrate_frame: interpolated %d defect pixels (r=%d)",
            int(defect_mask.sum()), r
        )
        return interp, weight_map

    # ---- Serialization ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Append mixture model to HDF5 file under /dark_mixture/."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "a") as f:
            if "dark_mixture" in f:
                del f["dark_mixture"]
            g = f.require_group("dark_mixture")
            g.create_dataset("class_probs",
                             data=self.class_probs.astype(np.float32),
                             compression="gzip", compression_opts=4)
            g.create_dataset("labels",
                             data=self.labels.astype(np.uint8),
                             compression="gzip", compression_opts=4)
            g.create_dataset("pi",    data=self.pi)
            g.create_dataset("alpha", data=self.alpha)
            g.create_dataset("beta",  data=self.beta)
            g.create_dataset("log_likelihood_curve",
                             data=np.array(self.log_likelihood_curve))
            g.attrs["normal_threshold"] = self.normal_threshold
            g.attrs["n_hot"]    = self.n_hot
            g.attrs["n_cold"]   = self.n_cold
            g.attrs["n_normal"] = self.n_normal

        logger.info("DarkMixtureModel saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "DarkMixtureModel":
        """Load from HDF5."""
        path = Path(path)
        with h5py.File(path, "r") as f:
            if "dark_mixture" not in f:
                raise ValueError(f"No /dark_mixture group in {path}")
            g = f["dark_mixture"]
            return cls(
                pi                   = g["pi"][:],
                alpha                = g["alpha"][:],
                beta                 = g["beta"][:],
                class_probs          = g["class_probs"][:],
                labels               = g["labels"][:],
                log_likelihood_curve = list(g["log_likelihood_curve"][:]),
                normal_threshold     = float(g.attrs.get("normal_threshold", 0.5)),
            )

    # ---- Display ----------------------------------------------------------

    def summary(self) -> str:
        H, W, _ = self.class_probs.shape
        n_px    = H * W
        lines   = ["DarkMixtureModel Summary", "=" * 42]
        labels  = ["normal", "hot   ", "cold  "]
        for k in range(3):
            lines.append(
                f"  {labels[k]}: π={self.pi[k]:.4f}  "
                f"α={self.alpha[k]:.3f}  β={self.beta[k]:.3f}  "
                f"E[λ]={self.component_means[k]:.5f} ADU/s  "
                f"n={(self.labels == k).sum():,}"
            )
        lines.append(f"  EM iterations: {len(self.log_likelihood_curve)}")
        ll_delta = (self.log_likelihood_curve[-1] - self.log_likelihood_curve[0]
                    if len(self.log_likelihood_curve) >= 2 else 0.0)
        lines.append(f"  Log-likelihood gain: {ll_delta:.2f}")
        defect_frac = (self.n_hot + self.n_cold) / n_px
        lines.append(f"  Defect pixel fraction: {defect_frac*100:.3f}%")
        return "\n".join(lines)


# ============================================================================
# EM fitting
# ============================================================================

def fit_dark_mixture(
    dark_rate: np.ndarray,
    sensor_mean_rate: Optional[float]      = None,
    n_iter:           int                  = 100,
    tol:              float                = 1e-4,
    normal_threshold: float                = 0.5,
    min_pi:           float                = 1e-4,
    seed:             int                  = 42,
) -> DarkMixtureModel:
    """
    Fit a three-component Gamma mixture to a dark rate map by EM.

    Parameters
    ----------
    dark_rate : np.ndarray [H, W] float32 or float64
        Per-pixel dark current rate [ADU/s] from InstrumentModel.fit_dark().
        Negative values (measurement noise around zero) are clipped to a
        small positive value before fitting.

    sensor_mean_rate : float, optional
        Expected dark rate for a normal pixel [ADU/s].
        Used to initialise the normal component mean.
        If None, estimated from the 10th–90th percentile mean of dark_rate.

    n_iter : int
        Maximum EM iterations.

    tol : float
        Convergence threshold on relative change in log-likelihood.

    normal_threshold : float
        Passed to DarkMixtureModel; pixels with p_normal < this are
        treated as defective in calibrate_frame().  Default 0.5.

    min_pi : float
        Minimum component weight to prevent component collapse.

    seed : int
        Random seed for reproducibility of initialisation.

    Returns
    -------
    DarkMixtureModel
    """
    rng = np.random.default_rng(seed)
    H, W = dark_rate.shape

    # Clip to small positive — Gamma is defined on (0, ∞)
    # Use a floor of 1e-6 ADU/s (essentially zero dark current)
    rates = np.maximum(dark_rate.astype(np.float64).ravel(), 1e-6)
    N     = rates.size

    logger.info(
        "fit_dark_mixture: fitting %d pixels  "
        "rate range=[%.5f, %.5f] ADU/s",
        N, rates.min(), rates.max()
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    if sensor_mean_rate is None:
        # Robust estimate of normal component mean
        p10, p90 = np.percentile(rates, [10, 90])
        sensor_mean_rate = float(rates[(rates >= p10) & (rates <= p90)].mean())
        logger.debug("Auto sensor_mean_rate = %.5f ADU/s", sensor_mean_rate)

    # Normal component: centred on sensor mean
    # Hot component:    10× the sensor mean (will be refined)
    # Cold component:   0.1× the sensor mean (will be refined)
    mean_init = np.array([
        sensor_mean_rate,          # normal
        sensor_mean_rate * 10.0,   # hot
        sensor_mean_rate * 0.1,    # cold
    ])
    mean_init = np.maximum(mean_init, 1e-6)

    # Concentration parameter for each component (strength of initial estimate)
    alpha = np.array([5.0, 2.0, 2.0])   # normal more concentrated initially
    beta  = alpha / mean_init

    # Mixing weights
    pi = np.array([0.995, 0.004, 0.001])

    log_likelihood_curve: List[float] = []

    # ------------------------------------------------------------------
    # EM loop
    # ------------------------------------------------------------------
    responsibilities = np.zeros((N, 3), dtype=np.float64)

    for iteration in range(n_iter):

        # ---- E-step: compute log responsibilities ----------------------
        log_r = np.zeros((N, 3), dtype=np.float64)
        for k in range(3):
            log_r[:, k] = (
                np.log(pi[k])
                + (alpha[k] - 1.0) * np.log(rates)
                - beta[k] * rates
                + alpha[k] * np.log(beta[k])
                - gammaln(alpha[k])
            )

        # Log-sum-exp for numerical stability
        log_r_max  = log_r.max(axis=1, keepdims=True)
        log_sum_r  = log_r_max.squeeze() + np.log(
            np.exp(log_r - log_r_max).sum(axis=1)
        )
        log_r     -= log_sum_r[:, np.newaxis]
        responsibilities = np.exp(log_r)          # [N, 3]

        # Complete-data log-likelihood
        ll = log_sum_r.sum()
        log_likelihood_curve.append(float(ll))

        if iteration > 0:
            rel_change = abs(ll - log_likelihood_curve[-2]) / (abs(log_likelihood_curve[-2]) + 1e-10)
            if rel_change < tol:
                logger.info("EM converged at iteration %d (rel Δll=%.2e)", iteration, rel_change)
                break

        # ---- M-step: update parameters --------------------------------
        r_sum = responsibilities.sum(axis=0)   # [3]
        pi    = np.maximum(r_sum / N, min_pi)
        pi   /= pi.sum()

        for k in range(3):
            r_k = responsibilities[:, k]
            r_k_sum = r_k.sum()
            if r_k_sum < 1e-10:
                continue  # degenerate component — skip update

            # Weighted sufficient statistics
            weighted_mean     = np.dot(r_k, rates) / r_k_sum
            weighted_mean_log = np.dot(r_k, np.log(rates)) / r_k_sum

            # Shape MLE via Choi-Wette + Newton
            log_weighted_mean = np.log(weighted_mean)
            alpha_k = _gamma_shape_mle(
                np.array([weighted_mean_log]),
                np.array([log_weighted_mean]),
            )[0]
            beta_k = alpha_k / weighted_mean

            alpha[k] = max(alpha_k, 1e-6)
            beta[k]  = max(beta_k,  1e-6)

    # Ensure hot > normal > cold ordering by mean
    # Sort components by their mean rate
    means = alpha / beta
    order = np.argsort(means)[::-1]   # descending: hot first, then normal, cold
    # Re-map to [hot, normal, cold] → [normal, hot, cold]
    # We want: index 0=normal(middle), 1=hot(highest), 2=cold(lowest)
    sorted_means = means[order]
    # Identify which sorted position corresponds to our semantic labels
    # Middle mean → normal, highest → hot, lowest → cold
    # order[0]=highest, order[1]=middle, order[2]=lowest
    remap = np.array([order[1], order[0], order[2]])   # [normal, hot, cold]
    pi_out    = pi[remap]
    alpha_out = alpha[remap]
    beta_out  = beta[remap]

    # Recompute final responsibilities with remapped parameters
    log_r = np.zeros((N, 3), dtype=np.float64)
    for k in range(3):
        log_r[:, k] = (
            np.log(pi_out[k])
            + (alpha_out[k] - 1.0) * np.log(rates)
            - beta_out[k] * rates
            + alpha_out[k] * np.log(beta_out[k])
            - gammaln(alpha_out[k])
        )
    log_r_max = log_r.max(axis=1, keepdims=True)
    log_r    -= log_r_max + np.log(np.exp(log_r - log_r_max).sum(axis=1, keepdims=True))
    responsibilities = np.exp(log_r)

    class_probs = responsibilities.reshape(H, W, 3).astype(np.float32)
    labels      = np.argmax(class_probs, axis=2).astype(np.uint8)

    model = DarkMixtureModel(
        pi                   = pi_out,
        alpha                = alpha_out,
        beta                 = beta_out,
        class_probs          = class_probs,
        labels               = labels,
        log_likelihood_curve = log_likelihood_curve,
        normal_threshold     = normal_threshold,
    )

    logger.info("\n%s", model.summary())
    return model
