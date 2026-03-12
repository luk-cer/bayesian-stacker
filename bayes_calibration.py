"""
bayes_calibration.py
====================
Bayesian conjugate-prior accumulators for astrophotography calibration.

Each accumulator wraps one noise source with its natural conjugate prior,
enabling **closed-form posterior updates** from streaming calibration frames.
No MCMC.  All updates are array operations — the posterior for a 9 Mpx
sensor updates in milliseconds.

Design goals
------------
- All posteriors serialize to / deserialize from HDF5 alongside the
  InstrumentModel, so cross-session accumulation is lossless.
- Posteriors carry genuine uncertainty — point estimates are just the
  posterior mean, not the whole story.
- Priors default to published ZWO ASI533MC-Pro sensor specifications at
  gain 100.  Completely overridable; any camera can be supported by
  supplying a SensorPriors instance.
- Flat gain prior deliberately carries forward only the uncertainty
  (variance structure) from the previous session, NOT the gain mean —
  because dust moves and optics change between sessions.

Noise source → conjugate pair
------------------------------

  Bias mean (per pixel)
      Likelihood   : Gaussian(μ, σ_r²)     [known variance = read noise]
      Prior on μ   : Gaussian(μ₀, τ₀²)
      Posterior    : Gaussian(μ_n, τ_n²)   — closed form

  Read noise variance (per pixel)
      Likelihood   : Gaussian(μ, σ²)       [known mean = bias]
      Prior on σ²  : InverseGamma(α₀, β₀)
      Posterior    : InverseGamma(αₙ, βₙ)  — closed form

  Dark current rate (per pixel)
      Likelihood   : Poisson(λ · t)
      Prior on λ   : Gamma(α₀, β₀)
      Posterior    : Gamma(αₙ, βₙ)         — closed form

  Flat gain (per pixel)
      Likelihood   : Gaussian(g · sky, σ²) — Gaussian approximation to
                     Poisson, valid for well-exposed flats (>1000 ADU)
      Prior on g   : Gaussian(g₀, v₀)
      Posterior    : Gaussian(gₙ, vₙ)      — closed form
      NOTE: prior mean (g₀) is always reset to 1.0 between sessions;
            only prior variance (v₀) carries over.

Temperature correction for dark rate
--------------------------------------
Dark current doubles approximately every 5.5–7°C for silicon CMOS sensors.
The exact scaling follows the Arrhenius law:

    λ_d(T) = λ_d(T_ref) · exp(-E_g/2k · (1/T - 1/T_ref))

    E_g ≈ 1.12 eV  (silicon band gap at 300K)
    k   = 8.617×10⁻⁵ eV/K  (Boltzmann constant)

When loading a prior from a previous session recorded at a different
temperature, call `dark_acc.temperature_correct(T_prev, T_now)` before
running `update()` on new frames.

HDF5 layout (under /bayes/ group)
-----------------------------------
  /bayes/bias/mu_n      [H,W]  float64   posterior mean
  /bayes/bias/tau2_n    [H,W]  float64   posterior variance
  /bayes/bias/n         scalar int
  /bayes/read_noise/alpha_n  [H,W]  float64
  /bayes/read_noise/beta_n   [H,W]  float64
  /bayes/read_noise/n        scalar int
  /bayes/dark/alpha_n   [H,W]  float64
  /bayes/dark/beta_n    [H,W]  float64
  /bayes/dark/n         scalar int
  /bayes/dark/ref_temp  scalar float64   [°C] temperature when prior was set
  /bayes/flat/g_n       [H,W]  float64   posterior mean gain
  /bayes/flat/v_n       [H,W]  float64   posterior variance
  /bayes/flat/n         scalar int

Dependencies
------------
  numpy  h5py  scipy (scipy.stats for validation only, not in hot path)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
_E_G_EV   = 1.12                  # silicon band gap [eV]
_K_EV_PER_K = 8.617333e-5         # Boltzmann [eV/K]
_CELSIUS_TO_K = 273.15


# ============================================================================
# Sensor specification priors
# ============================================================================

@dataclass
class SensorPriors:
    """
    Weakly informative priors derived from published sensor specifications.

    These are used to initialise the conjugate accumulators at the very first
    session.  They are intentionally broad — wide enough that two or three
    calibration frames will dominate the posterior.  They prevent pathological
    behaviour (e.g. negative read noise) while imposing minimal information.

    Defaults correspond to ZWO ASI533MC-Pro at gain=100, -10°C cooling.

    Parameters
    ----------
    bias_mean_adu : float
        Expected bias pedestal.  ZWO ASI533 at gain 100: ~200–400 ADU.
    bias_mean_std_adu : float
        How uncertain we are about the bias mean *a priori*.
        Set large (100 ADU) so data dominates after 2–3 frames.

    read_noise_adu : float
        Prior guess for read noise [ADU].  At gain 100, ASI533 ≈ 1.7 e⁻;
        with ~0.27 e⁻/ADU this is ~6.3 ADU.  We use a round 6.0.
    read_noise_concentration : float
        Effective number of prior pseudo-observations for the variance.
        2.0 → very weak; 10.0 → moderately informative.

    dark_rate_adu_per_s : float
        Prior dark current rate [ADU/s] at the reference temperature.
        ASI533 at -10°C: ~0.002 ADU/s.  (0.005 e⁻/s ÷ 0.27 e⁻/ADU).
    dark_rate_concentration : float
        Effective prior sample count for the Gamma.  2.0 = very weak.

    dark_ref_temp_c : float
        Reference temperature for the dark rate prior [°C].
        Should match the temperature your darks were taken at.

    flat_gain_std : float
        Prior standard deviation on the flat gain (per pixel).
        1.0 → near-uninformative (gain could be anywhere from 0 to 2+).
        After the first session this is replaced by the measured posterior.
    """
    # Bias
    bias_mean_adu:           float = 300.0
    bias_mean_std_adu:       float = 100.0

    # Read noise
    read_noise_adu:          float = 6.0
    read_noise_concentration: float = 2.0   # pseudo-obs for InvGamma prior

    # Dark
    dark_rate_adu_per_s:     float = 0.002
    dark_rate_concentration: float = 2.0    # pseudo-obs for Gamma prior
    dark_ref_temp_c:         float = -10.0

    # Flat
    flat_gain_std:           float = 1.0    # reset each session; variance carries

    @classmethod
    def for_asi533_gain100(cls) -> "SensorPriors":
        """Factory for ZWO ASI533MC-Pro at gain 100, -10°C cooling."""
        return cls(
            bias_mean_adu            = 280.0,
            bias_mean_std_adu        = 100.0,
            read_noise_adu           = 6.0,       # ~1.7 e⁻ / 0.27 e⁻/ADU
            read_noise_concentration = 2.0,
            dark_rate_adu_per_s      = 0.002,     # ~0.005 e⁻/s / 0.27
            dark_rate_concentration  = 2.0,
            dark_ref_temp_c          = -10.0,
            flat_gain_std            = 1.0,
        )


# ============================================================================
# Temperature correction helper
# ============================================================================

def dark_rate_temperature_correction(
    rate:     np.ndarray,
    temp_from_c: float,
    temp_to_c:   float,
) -> np.ndarray:
    """
    Scale a dark current rate map from one temperature to another using
    the Arrhenius silicon model.

    λ(T_to) = λ(T_from) · exp(-E_g/2k · (1/T_to - 1/T_from))

    Parameters
    ----------
    rate : np.ndarray [H, W]
        Dark rate map [ADU/s] measured at temp_from_c.
    temp_from_c, temp_to_c : float
        Source and target temperatures [°C].

    Returns
    -------
    np.ndarray [H, W]
        Rate map scaled to temp_to_c.
    """
    T_from = temp_from_c + _CELSIUS_TO_K
    T_to   = temp_to_c   + _CELSIUS_TO_K
    exponent = -(_E_G_EV / (2.0 * _K_EV_PER_K)) * (1.0 / T_to - 1.0 / T_from)
    scale    = np.exp(exponent)
    logger.debug(
        "Dark rate temperature correction: %.1f°C → %.1f°C, scale=%.4f",
        temp_from_c, temp_to_c, scale
    )
    return rate * scale


# ============================================================================
# Conjugate prior accumulators
# ============================================================================

class BiasPriorAccumulator:
    """
    Gaussian-Gaussian conjugate update for the bias mean.

    Model
    -----
      μ^(p)  ~ Gaussian(μ₀, τ₀²)          prior on per-pixel mean
      x_i^(p) | μ^(p) ~ Gaussian(μ^(p), σ_r^(p)²)   likelihood

    With known per-pixel read noise σ_r^(p), the posterior is:

      τₙ² = 1 / (1/τ₀² + n/σ_r²)
      μₙ   = τₙ² · (μ₀/τ₀² + Σxᵢ/σ_r²)

    For the streaming update after each new frame x:
      1/τₙ² += 1/σ_r²
      numerator += x / σ_r²
      μₙ = numerator · τₙ²

    Memory: 3 × [H, W] float64 (precision, numerator accumulator, prior μ).
    """

    def __init__(
        self,
        prior_mean:     np.ndarray,   # μ₀ [H, W]
        prior_variance: np.ndarray,   # τ₀² [H, W]
    ) -> None:
        self._mu0     = prior_mean.astype(np.float64)
        self._tau2_0  = prior_variance.astype(np.float64)
        # Running accumulators in information (precision) form
        self._precision  = (1.0 / self._tau2_0).copy()   # 1/τₙ²
        self._numerator  = (self._mu0 / self._tau2_0).copy()  # Σxᵢ/σ_r² + μ₀/τ₀²
        self.n = 0

    def update(self, frame: np.ndarray, read_noise: np.ndarray) -> None:
        """
        Incorporate one bias frame.

        Parameters
        ----------
        frame : [H, W]  ADU values of one bias frame.
        read_noise : [H, W]  Per-pixel read noise [ADU].
            If unknown yet (first session before read noise fit), supply
            a constant array: np.full(shape, prior.read_noise_adu).
        """
        x  = frame.astype(np.float64)
        sr = read_noise.astype(np.float64)
        sr = np.maximum(sr, 1e-6)   # guard against zero
        inv_sr2 = 1.0 / (sr ** 2)
        self._precision += inv_sr2
        self._numerator += x * inv_sr2
        self.n += 1

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (posterior_mean [H,W], posterior_std [H,W]).

        posterior_mean = μₙ = numerator / precision
        posterior_std  = √(1/precision)
        """
        if self.n == 0:
            raise RuntimeError("BiasPriorAccumulator: no frames processed.")
        tau2_n = 1.0 / self._precision
        mu_n   = self._numerator * tau2_n
        return mu_n.astype(np.float64), np.sqrt(tau2_n).astype(np.float64)

    @property
    def posterior_mean(self) -> np.ndarray:
        return (self._numerator / self._precision).astype(np.float64)

    @property
    def posterior_variance(self) -> np.ndarray:
        return (1.0 / self._precision).astype(np.float64)

    def to_hdf5(self, group: h5py.Group) -> None:
        mu_n, _ = self.finalize()
        group.create_dataset("mu_n",   data=mu_n,              compression="gzip", compression_opts=4)
        group.create_dataset("tau2_n", data=self.posterior_variance, compression="gzip", compression_opts=4)
        group.attrs["n"] = self.n

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "BiasPriorAccumulator":
        mu_n   = group["mu_n"][:]
        tau2_n = group["tau2_n"][:]
        acc = cls.__new__(cls)
        acc._mu0       = mu_n.copy()
        acc._tau2_0    = tau2_n.copy()
        acc._precision = 1.0 / tau2_n
        acc._numerator = mu_n / tau2_n
        acc.n          = int(group.attrs["n"])
        return acc


class ReadNoisePriorAccumulator:
    """
    Inverse-Gamma conjugate update for per-pixel read noise variance.

    Model
    -----
      σ²^(p)  ~ InvGamma(α₀, β₀)   prior on per-pixel variance
      x_i^(p) | μ, σ² ~ Gaussian(μ^(p), σ²^(p))   likelihood

    With known per-pixel mean μ^(p) (= bias_mean), the posterior after
    n frames is:

      αₙ = α₀ + n/2
      βₙ = β₀ + Σ(xᵢ - μ)²/2

    The posterior mean (point estimate) is:
      E[σ²] = βₙ / (αₙ - 1)   [for αₙ > 1]

    Both α and β accumulate additively — perfectly suited to streaming.

    The prior parameters (α₀, β₀) are set from sensor specs:
      E[σ²] = β₀/(α₀-1)  →  β₀ = E[σ²] · (α₀-1)
      Use α₀ = concentration + 1, β₀ = concentration · σ²_prior
    """

    def __init__(
        self,
        prior_alpha: np.ndarray,   # α₀ [H, W] or scalar
        prior_beta:  np.ndarray,   # β₀ [H, W] or scalar
        shape:       Tuple[int, int],
    ) -> None:
        self._alpha = np.broadcast_to(
            np.asarray(prior_alpha, dtype=np.float64), shape).copy()
        self._beta  = np.broadcast_to(
            np.asarray(prior_beta,  dtype=np.float64), shape).copy()
        self.shape = shape
        self.n = 0

    def update(self, frame: np.ndarray, bias_mean: np.ndarray) -> None:
        """
        Incorporate one bias frame (used to estimate read noise).

        Parameters
        ----------
        frame : [H, W]  Raw ADU values.
        bias_mean : [H, W]  Current best estimate of per-pixel bias mean.
        """
        x   = frame.astype(np.float64)
        mu  = bias_mean.astype(np.float64)
        residual_sq = (x - mu) ** 2
        self._alpha += 0.5
        self._beta  += 0.5 * residual_sq
        self.n += 1

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (posterior_mean_variance [H,W], posterior_std_variance [H,W]).

        E[σ²]   = β/(α-1)     [valid when α > 1]
        Var[σ²] = β²/((α-1)²(α-2))  [valid when α > 2]
        """
        alpha_safe = np.maximum(self._alpha, 1.0 + 1e-9)
        mean_var   = self._beta / (alpha_safe - 1.0)
        alpha2     = np.maximum(self._alpha, 2.0 + 1e-9)
        var_var    = (self._beta ** 2) / ((alpha2 - 1.0) ** 2 * (alpha2 - 2.0))
        return mean_var.astype(np.float64), np.sqrt(var_var).astype(np.float64)

    @property
    def posterior_alpha(self) -> np.ndarray:
        return self._alpha.copy()

    @property
    def posterior_beta(self) -> np.ndarray:
        return self._beta.copy()

    @property
    def posterior_mean_std(self) -> np.ndarray:
        """Point estimate: posterior mean of σ (not σ²), for display."""
        mean_var, _ = self.finalize()
        return np.sqrt(np.maximum(mean_var, 0.0))

    def to_hdf5(self, group: h5py.Group) -> None:
        group.create_dataset("alpha_n", data=self._alpha, compression="gzip", compression_opts=4)
        group.create_dataset("beta_n",  data=self._beta,  compression="gzip", compression_opts=4)
        group.attrs["n"]     = self.n
        group.attrs["shape"] = list(self.shape)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "ReadNoisePriorAccumulator":
        alpha = group["alpha_n"][:]
        beta  = group["beta_n"][:]
        shape = tuple(int(x) for x in group.attrs["shape"])
        acc = cls.__new__(cls)
        acc._alpha = alpha
        acc._beta  = beta
        acc.shape  = shape
        acc.n      = int(group.attrs["n"])
        return acc


class DarkRatePriorAccumulator:
    """
    Gamma-Poisson conjugate update for per-pixel dark current rate.

    Model
    -----
      λ^(p)   ~ Gamma(α₀, β₀)     prior on per-pixel dark rate [ADU/s]
      x_i^(p) | λ, t ~ Poisson(λ · t)   likelihood per frame of duration t

    The posterior after observing counts x_i with exposure t_i:

      αₙ = α₀ + Σ x_i
      βₙ = β₀ + Σ t_i

    Posterior mean: E[λ] = αₙ / βₙ

    This is the cleanest conjugate pair in the pipeline — both α and β
    accumulate additively, perfectly suited to streaming updates.

    Note: x_i here should be the bias-subtracted ADU count, not including
    dark-flat residuals.  Negative values (from bias over-subtraction) are
    clamped to 0 before accumulation.

    Temperature correction
    ----------------------
    Call temperature_correct(T_from, T_to) before starting a new session
    at a different temperature.  This scales the prior mean (α/β) to the
    new operating temperature via the Arrhenius law, without changing the
    effective prior strength.
    """

    def __init__(
        self,
        prior_alpha: np.ndarray,   # α₀ [H, W]
        prior_beta:  np.ndarray,   # β₀ [H, W]  (units: seconds)
        ref_temp_c:  float,
        shape:       Tuple[int, int],
    ) -> None:
        self._alpha    = np.broadcast_to(
            np.asarray(prior_alpha, dtype=np.float64), shape).copy()
        self._beta     = np.broadcast_to(
            np.asarray(prior_beta,  dtype=np.float64), shape).copy()
        self.ref_temp_c = float(ref_temp_c)
        self.shape      = shape
        self.n          = 0
        self._total_exposure = 0.0   # Σ tᵢ [seconds]

    def update(self, frame: np.ndarray, exposure_s: float,
               bias_mean: np.ndarray) -> None:
        """
        Incorporate one dark frame.

        Parameters
        ----------
        frame : [H, W]  Raw dark frame ADU values.
        exposure_s : float  Exposure time in seconds.
        bias_mean : [H, W]  Per-pixel bias to subtract before accumulating.
        """
        x_bias_sub = frame.astype(np.float64) - bias_mean.astype(np.float64)
        x_clamped  = np.maximum(x_bias_sub, 0.0)
        self._alpha          += x_clamped
        self._beta           += exposure_s
        self._total_exposure += exposure_s
        self.n += 1

    def temperature_correct(self, temp_from_c: float, temp_to_c: float) -> None:
        """
        Scale the prior mean to a new operating temperature via Arrhenius.

        Call this BEFORE processing new frames at temp_to_c, passing the
        temperature at which the prior was originally measured.

        This adjusts α while keeping α/β (= prior mean rate) correctly
        scaled, and β stays in units of seconds so the conjugate update
        arithmetic remains consistent.

        Implementation: we scale α proportionally (β fixed in seconds).
        New mean = scale · old_mean = scale · (α/β)  →  new α = scale · α
        """
        if abs(temp_from_c - temp_to_c) < 0.1:
            return
        T_from = temp_from_c + _CELSIUS_TO_K
        T_to   = temp_to_c   + _CELSIUS_TO_K
        exponent = -(_E_G_EV / (2.0 * _K_EV_PER_K)) * (1.0 / T_to - 1.0 / T_from)
        scale    = np.exp(exponent)
        self._alpha    *= scale
        self.ref_temp_c = temp_to_c
        logger.info(
            "Dark prior temperature corrected: %.1f°C → %.1f°C  (scale=%.4f)",
            temp_from_c, temp_to_c, scale
        )

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (posterior_mean_rate [H,W], posterior_std_rate [H,W]).

        E[λ]   = α/β
        Var[λ] = α/β²
        Std[λ] = √(α)/β
        """
        beta_safe = np.maximum(self._beta, 1e-9)
        mean = self._alpha / beta_safe
        std  = np.sqrt(self._alpha) / beta_safe
        return mean.astype(np.float64), std.astype(np.float64)

    @property
    def posterior_alpha(self) -> np.ndarray:
        return self._alpha.copy()

    @property
    def posterior_beta(self) -> np.ndarray:
        return self._beta.copy()

    def to_hdf5(self, group: h5py.Group) -> None:
        group.create_dataset("alpha_n", data=self._alpha, compression="gzip", compression_opts=4)
        group.create_dataset("beta_n",  data=self._beta,  compression="gzip", compression_opts=4)
        group.attrs["n"]              = self.n
        group.attrs["shape"]          = list(self.shape)
        group.attrs["ref_temp_c"]     = self.ref_temp_c
        group.attrs["total_exposure"] = self._total_exposure

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "DarkRatePriorAccumulator":
        alpha = group["alpha_n"][:]
        beta  = group["beta_n"][:]
        shape = tuple(int(x) for x in group.attrs["shape"])
        acc = cls.__new__(cls)
        acc._alpha          = alpha
        acc._beta           = beta
        acc.shape           = shape
        acc.n               = int(group.attrs["n"])
        acc.ref_temp_c      = float(group.attrs["ref_temp_c"])
        acc._total_exposure = float(group.attrs.get("total_exposure", 0.0))
        return acc


class FlatGainPriorAccumulator:
    """
    Gaussian conjugate update for per-pixel flat gain.

    Model
    -----
      g^(p)   ~ Gaussian(g₀, v₀)        prior on per-pixel throughput
      f_i^(p) | g ~ Gaussian(g^(p)·S, σ²)  approx likelihood (Poisson → Gaussian
                                            valid for well-exposed flats > 1000 ADU)

    With known sky level S (per-channel median of each flat frame) and
    read noise σ, the posterior updates as:

      vₙ = 1 / (1/v₀ + n·S²/σ²)
      gₙ = vₙ · (g₀/v₀ + Σ fᵢ·S/σ²)

    Cross-session policy
    --------------------
    On a new session, the prior MEAN is ALWAYS reset to 1.0 (flat could be
    completely different due to dust/optics changes).  The prior VARIANCE
    carries over — it encodes how variable the pixel-to-pixel gain structure
    is, which is a stable sensor property.

    Reset behaviour is implemented in `from_hdf5(reset_mean=True)`.
    """

    def __init__(
        self,
        prior_gain_mean:     np.ndarray,   # g₀ [H, W]
        prior_gain_variance: np.ndarray,   # v₀ [H, W]
    ) -> None:
        self._g0    = prior_gain_mean.astype(np.float64)
        self._v0    = prior_gain_variance.astype(np.float64)
        # Information-form accumulators
        self._precision  = (1.0 / self._v0).copy()                 # 1/vₙ
        self._numerator  = (self._g0 / self._v0).copy()            # g₀/v₀ + Σ
        self.n = 0

    def update(
        self,
        flat_norm: np.ndarray,    # f_i / sky_level — normalised flat [H,W]
        sky_level: float,         # median sky level S [ADU] for this frame
        read_noise: np.ndarray,   # σ_r [H,W]
    ) -> None:
        """
        Incorporate one normalised flat frame.

        Parameters
        ----------
        flat_norm : [H, W]
            Flat frame divided by its per-channel sky median — values ~1.0.
        sky_level : float
            The sky median ADU level S used for normalisation.
            Used to weight the likelihood (higher sky = more photons = better SNR).
        read_noise : [H, W]
            Per-pixel read noise [ADU].  Used in the Poisson→Gaussian
            approximation for the likelihood variance.
        """
        f  = flat_norm.astype(np.float64)
        sr = np.maximum(read_noise.astype(np.float64), 1e-6)
        # Poisson shot variance ≈ sky_level (in ADU²); total variance = sky + σ_r²
        total_var = sky_level + sr ** 2
        inv_var   = 1.0 / total_var
        S2        = sky_level ** 2
        self._precision += S2 * inv_var
        self._numerator += f * S2 * inv_var         # f * S²/(S+σ²), matching precision units
        self.n += 1

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (posterior_mean_gain [H,W], posterior_std_gain [H,W])."""
        if self.n == 0:
            raise RuntimeError("FlatGainPriorAccumulator: no frames processed.")
        v_n = 1.0 / self._precision
        g_n = self._numerator * v_n
        return g_n.astype(np.float64), np.sqrt(v_n).astype(np.float64)

    @property
    def posterior_mean(self) -> np.ndarray:
        v_n = 1.0 / self._precision
        return (self._numerator * v_n).astype(np.float64)

    @property
    def posterior_variance(self) -> np.ndarray:
        return (1.0 / self._precision).astype(np.float64)

    def to_hdf5(self, group: h5py.Group) -> None:
        g_n, _ = self.finalize()
        group.create_dataset("g_n", data=g_n,                    compression="gzip", compression_opts=4)
        group.create_dataset("v_n", data=self.posterior_variance, compression="gzip", compression_opts=4)
        group.attrs["n"] = self.n

    @classmethod
    def from_hdf5(
        cls,
        group: h5py.Group,
        reset_mean: bool = True,
    ) -> "FlatGainPriorAccumulator":
        """
        Load accumulator from HDF5.

        Parameters
        ----------
        reset_mean : bool
            If True (default, always use on a new session), reset the prior
            mean to 1.0 per pixel.  The prior variance (v_n) carries over
            — it encodes the stable pixel-to-pixel gain variation structure.
            If False, continue accumulating from where we left off (useful
            for resuming an interrupted session with the same optical train).
        """
        v_n = group["v_n"][:]
        if reset_mean:
            g0 = np.ones_like(v_n)
            logger.info(
                "FlatGainPriorAccumulator: resetting prior mean to 1.0, "
                "carrying forward variance structure."
            )
        else:
            g0 = group["g_n"][:]
        acc = cls(prior_gain_mean=g0, prior_gain_variance=v_n)
        acc.n = 0   # fresh session
        return acc


# ============================================================================
# Top-level Bayesian calibration state
# ============================================================================

@dataclass
class BayesCalibrationState:
    """
    Container for all four conjugate prior accumulators.

    This is the Bayesian counterpart to InstrumentModel: it holds the full
    posterior distribution over each noise parameter, not just point estimates.

    Point estimates for use in calibrate_frame() are available via properties.
    The full posteriors are used in the MAP stacker for proper uncertainty
    propagation.

    Usage (first session)
    ---------------------
    >>> priors = SensorPriors.for_asi533_gain100()
    >>> state  = BayesCalibrationState.from_priors(priors, shape=(3008, 3008))
    >>> # Feed frames...
    >>> for path in bias_paths:
    ...     frame, hdr = fits.getdata(path, header=True)
    ...     state.bias_acc.update(frame, read_noise_init)
    >>> state.save("bayes_state.h5")

    Usage (subsequent session)
    --------------------------
    >>> state = BayesCalibrationState.load("bayes_state.h5",
    ...                                    new_session=True,
    ...                                    new_temp_c=-12.0)
    >>> # state.flat_acc has been reset, dark prior temperature-corrected
    """

    bias_acc:       Optional[BiasPriorAccumulator]      = None
    read_noise_acc: Optional[ReadNoisePriorAccumulator] = None
    dark_acc:       Optional[DarkRatePriorAccumulator]  = None
    flat_acc:       Optional[FlatGainPriorAccumulator]  = None
    shape:          Optional[Tuple[int, int]]            = None
    priors:         Optional[SensorPriors]               = None

    # ---- Point-estimate properties ----------------------------------------

    @property
    def bias_mean(self) -> Optional[np.ndarray]:
        """Posterior mean bias [ADU]."""
        if self.bias_acc is None: return None
        return self.bias_acc.posterior_mean

    @property
    def read_noise(self) -> Optional[np.ndarray]:
        """Posterior mean read noise [ADU]."""
        if self.read_noise_acc is None: return None
        return self.read_noise_acc.posterior_mean_std

    @property
    def dark_rate(self) -> Optional[np.ndarray]:
        """Posterior mean dark rate [ADU/s]."""
        if self.dark_acc is None: return None
        mean, _ = self.dark_acc.finalize()
        return mean

    @property
    def flat_gain(self) -> Optional[np.ndarray]:
        """Posterior mean flat gain (≈ 1.0 median)."""
        if self.flat_acc is None: return None
        return self.flat_acc.posterior_mean

    # ---- Factories --------------------------------------------------------

    @classmethod
    def from_priors(
        cls,
        priors: SensorPriors,
        shape:  Optional[Tuple[int, int]],
    ) -> "BayesCalibrationState":
        """
        Initialise all four accumulators from sensor spec priors.
        Call this for the very first session with a new sensor.

        When shape=None the accumulators are deferred; call
        ensure_initialized(shape) once the first calibration frame has been
        loaded and its shape is known.
        """
        if shape is None:
            return cls(shape=None, priors=priors)
        H, W = shape
        c = priors.read_noise_concentration
        sigma2_prior = priors.read_noise_adu ** 2

        bias_acc = BiasPriorAccumulator(
            prior_mean     = np.full((H, W), priors.bias_mean_adu),
            prior_variance = np.full((H, W), priors.bias_mean_std_adu ** 2),
        )
        rn_acc = ReadNoisePriorAccumulator(
            prior_alpha = np.full((H, W), c + 1.0),
            prior_beta  = np.full((H, W), c * sigma2_prior),
            shape       = shape,
        )
        dk_acc = DarkRatePriorAccumulator(
            prior_alpha = np.full((H, W), priors.dark_rate_concentration + 1.0),
            prior_beta  = np.full((H, W), (priors.dark_rate_concentration + 1.0)
                                           / priors.dark_rate_adu_per_s),
            ref_temp_c  = priors.dark_ref_temp_c,
            shape       = shape,
        )
        fl_acc = FlatGainPriorAccumulator(
            prior_gain_mean     = np.ones((H, W)),
            prior_gain_variance = np.full((H, W), priors.flat_gain_std ** 2),
        )
        logger.info(
            "BayesCalibrationState initialised from priors "
            "(shape=%s, rn=%.1f ADU, dark=%.4f ADU/s @ %.1f°C)",
            shape, priors.read_noise_adu,
            priors.dark_rate_adu_per_s, priors.dark_ref_temp_c,
        )
        return cls(
            bias_acc       = bias_acc,
            read_noise_acc = rn_acc,
            dark_acc       = dk_acc,
            flat_acc       = fl_acc,
            shape          = shape,
            priors         = priors,
        )

    def ensure_initialized(self, shape: Tuple[int, int]) -> None:
        """
        Lazily build accumulators from stored priors the first time a
        calibration frame shape is known.  No-op if already initialised
        or if no priors are stored.
        """
        if self.bias_acc is not None or self.priors is None:
            return
        initialised = self.__class__.from_priors(self.priors, shape)
        self.bias_acc       = initialised.bias_acc
        self.read_noise_acc = initialised.read_noise_acc
        self.dark_acc       = initialised.dark_acc
        self.flat_acc       = initialised.flat_acc
        self.shape          = shape

    # ---- Serialization ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save all posterior parameters to HDF5 under /bayes/."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "a") as f:   # append — InstrumentModel may already be here
            if "bayes" in f:
                del f["bayes"]
            g = f.require_group("bayes")
            if self.bias_acc is not None and self.bias_acc.n > 0:
                self.bias_acc.to_hdf5(g.require_group("bias"))
            if self.read_noise_acc is not None and self.read_noise_acc.n > 0:
                self.read_noise_acc.to_hdf5(g.require_group("read_noise"))
            if self.dark_acc is not None and self.dark_acc.n > 0:
                self.dark_acc.to_hdf5(g.require_group("dark"))
            if self.flat_acc is not None and self.flat_acc.n > 0:
                self.flat_acc.to_hdf5(g.require_group("flat"))
            if self.shape is not None:
                g.attrs["shape"] = list(self.shape)

        logger.info("BayesCalibrationState saved → %s", path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        new_session: bool = False,
        new_temp_c: Optional[float] = None,
    ) -> "BayesCalibrationState":
        """
        Load from HDF5.

        Parameters
        ----------
        new_session : bool
            If True, treat this as a new imaging session:
            - Flat gain mean is reset to 1.0 (variance carries over).
            - Frame counts are zeroed so new data is not double-counted.
        new_temp_c : float, optional
            If provided, the dark prior is temperature-corrected from the
            stored reference temperature to new_temp_c before accumulation.
        """
        path = Path(path)
        state = cls()

        with h5py.File(path, "r") as f:
            if "bayes" not in f:
                raise ValueError(f"No /bayes group found in {path}")
            g = f["bayes"]
            shape_raw = g.attrs.get("shape")
            if shape_raw is not None:
                state.shape = tuple(int(x) for x in shape_raw)

            if "bias" in g:
                state.bias_acc = BiasPriorAccumulator.from_hdf5(g["bias"])
                if new_session:
                    state.bias_acc.n = 0

            if "read_noise" in g:
                state.read_noise_acc = ReadNoisePriorAccumulator.from_hdf5(g["read_noise"])
                if new_session:
                    state.read_noise_acc.n = 0

            if "dark" in g:
                state.dark_acc = DarkRatePriorAccumulator.from_hdf5(g["dark"])
                if new_session:
                    state.dark_acc.n = 0
                if new_temp_c is not None:
                    prev_temp = state.dark_acc.ref_temp_c
                    state.dark_acc.temperature_correct(prev_temp, new_temp_c)

            if "flat" in g:
                state.flat_acc = FlatGainPriorAccumulator.from_hdf5(
                    g["flat"],
                    reset_mean = new_session,
                )

        logger.info(
            "BayesCalibrationState loaded ← %s  (new_session=%s, new_temp=%s)",
            path, new_session, new_temp_c
        )
        return state

    # ---- Display ----------------------------------------------------------

    def summary(self) -> str:
        lines = ["BayesCalibrationState", "=" * 42]
        lines.append(f"Shape: {self.shape}")
        if self.bias_acc is not None:
            mu = self.bias_acc.posterior_mean
            lines.append(f"Bias   n={self.bias_acc.n:4d}  "
                         f"mean={np.median(mu):.1f} ADU  "
                         f"std(posterior)={np.median(np.sqrt(self.bias_acc.posterior_variance)):.3f}")
        if self.read_noise_acc is not None:
            rn = self.read_noise_acc.posterior_mean_std
            lines.append(f"RN     n={self.read_noise_acc.n:4d}  "
                         f"median={np.median(rn):.2f} ADU")
        if self.dark_acc is not None:
            dr, dr_std = self.dark_acc.finalize()
            lines.append(f"Dark   n={self.dark_acc.n:4d}  "
                         f"median={np.median(dr):.5f} ADU/s  "
                         f"ref_temp={self.dark_acc.ref_temp_c:.1f}°C")
        if self.flat_acc is not None:
            fg = self.flat_acc.posterior_mean
            lines.append(f"Flat   n={self.flat_acc.n:4d}  "
                         f"median gain={np.median(fg):.4f}  "
                         f"std(gain)={np.median(np.sqrt(self.flat_acc.posterior_variance)):.5f}")
        return "\n".join(lines)
