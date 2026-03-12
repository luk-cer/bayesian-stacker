# Bayesian Astro Stacker — TODO

Status markers:
- ✅ Done
- 🔲 Planned — design agreed, ready to implement
- 💡 Explore — idea worth investigating, needs more thought
- 📖 Research — open theoretical question

---

## Phase 0a — Instrument calibration model (`instrument_model_artifact.py`) ✅

- [x] `WelfordAccumulator` — online mean + variance, numerically stable
- [x] `LinearRegressionAccumulator` — online per-pixel linear regression
- [x] `InstrumentModel` dataclass — all calibration outputs
- [x] `fit_bias` → `bias_mean`, `read_noise`
- [x] `fit_dark` → `dark_rate`, `hot_pixel_mask`
- [x] `fit_flat` → `flat_gain`, `flat_uncertainty`
- [x] `fit_dark_flat` → `amp_glow_profile`
- [x] `fit_all` — top-level entry point, frame type discovery
- [x] `calibrate_frame` — bias/dark/flat correction
- [x] HDF5 serialization (`save` / `load`)
- [x] Auto chunk sizing from available RAM via `psutil`
- [x] OSC Bayer pattern detection

---

## Phase 0b — Bayesian prior hierarchy (`bayes_calibration.py`) ✅

- [x] `SensorPriors` with factory `for_asi533_gain100()`
- [x] `BiasPriorAccumulator` — Gaussian-Gaussian conjugate
- [x] `ReadNoisePriorAccumulator` — InverseGamma conjugate
- [x] `DarkRatePriorAccumulator` — Gamma-Poisson conjugate
- [x] `FlatGainPriorAccumulator` — Gaussian conjugate; flat mean reset per session
- [x] `dark_rate_temperature_correction()` — Arrhenius scaling (E_g = 1.12 eV)
- [x] `BayesCalibrationState` — container + `from_priors` / `save` / `load`
- [x] Cross-session carry-over: read noise, dark rate (temp-corrected), flat variance
- [x] HDF5 serialization under `/bayes/` group

### 🔲 Remaining

- [ ] **Temporal decay for hot pixel priors** — `α_prior_new = 1 + ρ·(α_post - 1)`;
      integrate with DarkMixtureModel so hot pixel confidence fades across sessions

---

## Phase 0c — Hot/cold pixel mixture model (`dark_mixture.py`) ✅

- [x] `fit_dark_mixture()` — EM, 3-component Gamma mixture; log-sum-exp stable
- [x] `DarkMixtureModel` — `pixel_weight_map()`, `calibrate_frame()`
- [x] 99.4% hot pixel recovery on 256×256 synthetic test

---

## Phase 1 — Optics + WCS (`optics.py`) ✅

- [x] `ScopeGeometry` + `compute_airy_psf()` + `compute_broadband_psf()`
- [x] `WCSGeometry` + `extract_wcs_geometry()` — CD matrix, CDELT, CROTA2 support
- [x] `FrameShift` + `compute_frame_shift()` — sub-pixel dx/dy, rotation, scale
- [x] `ASTAPSolver` — subprocess wrapper; RA/Dec hint from FITS header

### 🔲 Provision only (not implementing soon)

- [ ] Spatially varying optical aberration PSF (`psf_optics[grid_y, grid_x, K, K]`)
- [ ] Isolation of H_optics by deconvolving H_diff from best-seeing empirical PSFs

---

## Phase 2 — Frame characterisation (`frame_characterizer.py`) ✅

- [x] `FrameCharacterizer` class — processes one light frame at a time
- [x] `estimate_sky_background()` — tiled sigma-clip + 2-D Legendre polynomial fit
- [x] `extract_stars()` — local-max detection, saturation/isolation filter, sub-pixel centroids
- [x] `estimate_psf_from_stamps()` — Fourier-align → normalise → median stack
- [x] `fit_moffat()` — `curve_fit` with second-moment fallback
- [x] `deconvolve_instrument_psf()` — Wiener deconvolution for H_seeing
- [x] `estimate_transparency()` — aperture photometry vs reference stellar fluxes
- [x] `characterize()` + `characterize_calibrated()` — full per-frame pipeline
- [x] Graceful degradation: solve failure → shift=None; <min_stars → prior PSF; <2 fluxes → t=1.0
- [x] 9/9 unit tests passing

---

## Phase 3 — Sufficient statistics accumulation (`sufficient_statistics.py`) ✅

- [x] `SufficientStatsAccumulator` — streaming, O(1) RAM regardless of frame count
- [x] Accumulates: `weighted_sum`, `weight_sum`, `sky_sum`, `sq_sum` + per-frame lists
- [x] `SufficientStats` — `weighted_mean`, `variance_map`, `quality_map`, `sky_mean`
- [x] `save()` / `resume()` — HDF5 crash recovery
- [x] `select_best_frames()` + `rebuild_stats_from_subset()`
- [x] `save_fast_stack_fits()` — Gamma-Poisson mean stack as FITS
- [x] 13/13 unit tests passing

---

## Phase 4 — MAP super-resolution stacker (`map_stacker.py`) ✅

- [x] Softplus reparameterisation — enforces λ > 0 exactly, no projection step
- [x] FAST mode — Phase 3 `weighted_sum` as proxy; ~50× faster than EXACT
- [x] EXACT mode — raw FITS mini-batch loading; per-frame PSF + shift + transparency
- [x] Sub-pixel shift via Fourier phase-shift theorem (exact, no interpolation)
- [x] Zero-padded FFT PSF convolution (no circular artefacts)
- [x] Total Variation regularisation (isotropic, weight α_tv, default 1e-2)
- [x] KL-from-Poisson-prior regularisation (weight α_kl, default 0)
- [x] Haar wavelet L1 sparsity (weight α_wav, default 0)
- [x] Scale factor configurable at runtime (2×, 3×, 4×, …)
- [x] Adam + cosine-annealing LR schedule
- [x] Early stopping (rel_tol + patience)
- [x] `MapConfig` — all hyperparameters in one dataclass with validation
- [x] `MapResult` — `save_fits()` with QUALITY extension, `save_convergence_plot()`
- [x] `run_from_stats_file()` — convenience one-liner
- [x] 18/18 unit tests passing (all maths verified via NumPy reference implementation)

### 🔲 Remaining

- [ ] **End-to-end GPU smoke test** — run a tiny (64×64) synthetic stack through
      `solve()` on real PyTorch; verify loss decreases and λ_hr correlates with
      ground truth. Blocked until torch is installable in this environment.

---

## Combined pipeline (`bayesian_astro_stacker.py`) ✅

- [x] `PipelineConfig` — all pipeline hyperparameters in one dataclass
- [x] `BayesianAstroStacker.run()` — Phases 0–4 end-to-end
- [x] `BayesianAstroStacker.resume()` — restart from Phase 3 checkpoint
- [x] `PipelineResult` — summary string, output file map
- [x] Quality frame filtering (min transparency, max FWHM)
- [x] Per-N-frame checkpoint saves
- [x] CLI entry point (`python bayesian_astro_stacker.py lights/ results/ ...`)
- [x] Graceful degradation when PyTorch absent (fast stack still written)

---

## Phase 5 — Validation ✅ (partial)

### Completed ✓

- [x] **Synthetic star field generator** (`synthetic_starfield.py`) — 15/15 tests pass
      - Power-law star flux distribution, Moffat PSF rendered at HR grid resolution
      - Sub-pixel dither via Fourier phase-shift theorem (exact, no interpolation)
      - Per-frame seeing (Moffat FWHM) and transparency variation
      - Gaussian nebula blobs for extended emission
      - WCS header injection encoding true shift per frame
      - Sensor forward model applied via `CalibrationFrameGenerator`
      - `StarfieldGroundTruth` carries all true values (shifts, FWHMs, transparencies, PSFs, scene)
      - `StarfieldConfig.for_asi533()` factory for ZWO ASI533 / Esprit 100ED
      - Bug fixed: `_render_nebula_hr` margin clamped to prevent `low >= high` on small grids

- [x] **Full pipeline ground-truth test** (`ground_truth_test.py`) — 15/15 tests pass
      - Phases 0 → 2 → 3 → (4 when torch available) validated on synthetic data
      - Phase 0: bias MAE, read noise MAE, dark rate MAE (normal pixels), flat gain MAE
      - Phase 2: per-frame FWHM recovery error, transparency error, shift dx/dy error
      - Phase 3: fast stack vs true LR scene — RMSE, PSNR, Pearson r
      - Phase 4: MAP λ_hr vs true HR scene — RMSE, PSNR, Pearson r + SR PSD ratio
      - SR frequency test: PSD above native Nyquist (MAP must exceed fast-stack upsampled)
      - `ValidationReport` with pass/fail thresholds and boxed ASCII summary
      - Diagnostic plots: scene comparison, per-frame metrics, power spectrum
      - CLI: `python ground_truth_test.py --fast / --full / --output-dir results/`
      - `TestConfig.fast()` (64×64, 8 frames, no MAP) and `.full()` (256×256, 30 frames)

### Remaining

- [ ] **Phase 4: GPU smoke test** — run `TestConfig.fast()` through `solve()` on real
      PyTorch (`pip install torch`); verify loss decreases and λ_hr Pearson r > 0.90.
      Blocked until torch is installable in this environment.

- [ ] **Power spectrum SR frequency test on real data** — confirm MAP recovers spatial
      frequencies above native Nyquist on actual ASI533 dithered frames.

- [ ] **Comparison vs PixInsight / DeepSkyStacker** — stack same real dataset;
      compare star FWHM, background σ, faint feature visibility.

- [ ] **Memory profiling** — verify peak RAM < 8 GB for 50-frame ASI533 session
      through full pipeline including MAP optimisation.

---

## Summary of what remains

| Priority | Item | Effort |
|----------|------|--------|
| High | Phase 4: GPU smoke test (needs `pip install torch`) | Small |
| Medium | SR frequency test on real ASI533 data | Small |
| Medium | Real data vs PixInsight / DSS comparison | External |
| Medium | Memory profiling (50-frame session) | Small |
| Low | Phase 0b: hot pixel temporal decay across sessions | Small |
| Low | Phase 1: spatially varying PSF | Large |

---

## 💡 Ideas worth exploring

- **Uncertainty image** — per-pixel posterior variance map from Phase 4 MAP;
  flag low-confidence regions (few frames, bad seeing, vignetting).

- **Adaptive TV weight** — spatially varying α: lower in bright nebula (trust data),
  higher in faint background (trust prior). Derive from `variance_map`.

- **Incremental / online stacker** — update MAP as new frames arrive; warm-start
  from previous λ_hr. Enables live preview during imaging.

- **Deep image prior** — replace TV with untrained CNN as implicit prior.
  No training data needed; stronger results but more compute.

- **Satellite / trail rejection** — per-pixel on-trail mixture component; soft mask
  from posterior probability. More principled than hard streak detection.

- **OSC Bayer SR** — R/G1/G2/B sub-pixels are offset by ½ pixel; built-in
  sub-pixel diversity per frame. Currently unused by the stacker.

---

## 📖 Research questions

- **Poisson-Gaussian approximation cost** — calibrated frames follow
  Poisson(λ) * N(0, σ_r²). At what sub-exposure length does the pure Poisson
  MAP approximation introduce meaningful bias for the ASI533 (σ_r ≈ 6 ADU)?

- **Optimal SR scale factor** — what factor is actually supported by data SNR for
  the Esprit 100ED + ASI533 at 3–4″ seeing and 5–10h integration?

- **Spatially varying PSF in MAP** — efficient formulation? Sectored SVD
  decomposition preserves O(N log N) cost — worth the complexity?

- **Moffat vs von Kármán seeing** — does the physically motivated von Kármán
  model improve H_seeing separation accuracy vs empirical Moffat?

- **Field rotation** — at what polar alignment error / session length does
  residual field rotation become measurable and need including in W_i?
