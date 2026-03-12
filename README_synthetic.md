# Synthetic Data Generation & Calibration Validation

This document covers two modules that together let you test the entire
calibration pipeline against a known ground truth before processing any
real frames:

```
synthetic_calibration.py   — Bayesian sampler + calibration frame generator
synthetic_scene.py         — sky scene builder + calibration validator
```

The core idea: generate frames where you know every parameter exactly, run
the real pipeline on them, and verify the output matches.

---

## How it fits into the pipeline

```
SensorPriors or BayesCalibrationState
         │
         ▼
  sample_instrument()          ← draw true per-pixel parameters
         │
         ├──► generate_bias_frames()    ┐
         ├──► generate_dark_frames()    ├─ synthetic FITS folder
         ├──► generate_flat_frames()    │  (same layout as real data)
         └──► generate_dark_flat_frames()┘
                      │
                      ▼
           InstrumentModel.fit_all()    ← the real pipeline, unchanged
                      │
                      ▼
           calibrate_frame(raw_light)   ← apply to synthetic light frame
                      │
                      ▼
           CalibrationValidator.run()   ← compare to ground truth
                      │
                      ▼
              ValidationReport
         residual map │ param errors │ noise budget χ²
```

---

## Quickest possible test

One function call runs everything end-to-end:

```python
import logging
logging.basicConfig(level=logging.INFO)

from synthetic_scene import run_full_validation

report = run_full_validation(
    shape          = (256, 256),   # sensor size — use small for fast iteration
    n_bias         = 30,
    n_flat         = 25,
    dark_exposures = [60., 120., 300., 300., 600.],
    dark_repeats   = 3,
    light_exposure = 300.0,
    n_repeat_light = 50,           # frames used to estimate noise variance
    bayes_mode     = True,         # use BayesCalibrationState in fit_all
    seed           = 42,
)
print(report.summary())
```

Expected output (numbers will vary slightly with seed):

```
==============================================================
  Calibration Validation Report
==============================================================
  Exposure         : 300 s
  Repeat frames    : 50
  Global χ² median : 0.985  (ideal = 1.00)

── Parameter recovery ───────────────────────────────────────
  bias_mean [ADU]   true= 280.009  est= 280.013  MAE=  0.776 ( 0.28%)  p95=2.30
  read_noise [ADU]  true=   6.226  est=   5.732  MAE=  0.607 ( 9.75%)  p95=1.60
  dark_rate [ADU/s] true=   0.002  est=   0.008  MAE=  0.006 (...)     p95=0.012
  flat_gain         true=   1.000  est=   1.000  MAE=  0.018 ( 1.80%)  p95=0.024

── Spatial region residuals ─────────────────────────────────
  disc           n=  1319  mean=+183  std=13568  σ_exp= 864  χ²=0.991 ✓
  gaussian       n=  1319  mean=+153  std=10468  σ_exp= 688  χ²=0.974 ✓
  background     n= 62486  mean= +39  std= 2378  σ_exp= 357  χ²=0.985 ✓
==============================================================
```

Save FITS diagnostic files to inspect in a viewer:

```python
report = run_full_validation(
    shape=(256, 256),
    output_dir="validation_output/",
    seed=42,
)
# Writes:
#   validation_output/residual.fits            calibrated − true_sky
#   validation_output/variance_empirical.fits
#   validation_output/variance_expected.fits
#   validation_output/chi2_map.fits            should be ≈ 1.0 everywhere
```

---

## Step-by-step usage

### 1. Sample a true instrument

**From sensor spec priors (no real session data needed):**

```python
import numpy as np
from bayes_calibration import SensorPriors
from synthetic_calibration import sample_instrument_from_priors

priors = SensorPriors.for_asi533_gain100()
# Or customise:
# priors = SensorPriors(
#     bias_mean_adu       = 280.0,
#     read_noise_adu      = 6.0,
#     dark_rate_adu_per_s = 0.002,
#     dark_ref_temp_c     = -10.0,
# )

rng   = np.random.default_rng(42)
instr = sample_instrument_from_priors(priors, shape=(256, 256), rng=rng)
print(instr.summary())
```

```
TrueInstrument (ground truth)
==========================================
  Shape          : (256, 256)
  bias_mean      : median=280.04  std=4.57 ADU
  read_noise     : median=6.23 ADU
  dark_rate      : median=0.00190 ADU/s  max=0.1086
  flat_gain      : median=1.0000  range=[0.930, 1.036]
  hot pixels     : 318 (0.484%)
```

**From posteriors (after one or more real sessions):**

```python
from bayes_calibration import BayesCalibrationState
from synthetic_calibration import sample_instrument

state = BayesCalibrationState.load("instrument.h5")
instr = sample_instrument(state, rng=rng)
```

The sampled distributions:

| Parameter | Distribution drawn from |
|-----------|------------------------|
| `bias_mean[p]` | `N(mu_n[p], tau2_n[p])` — posterior mean and variance |
| `read_noise[p]` | `sqrt(InvGamma(alpha_n[p], beta_n[p]))` |
| `dark_rate[p]` | `Gamma(alpha_n[p], 1/beta_n[p])` |
| `flat_gain[p]` | `N(g_n[p], v_n[p])`, then normalised to median=1.0 |

### 2. Generate calibration frames

```python
from synthetic_calibration import CalibrationFrameGenerator

gen = CalibrationFrameGenerator(instr, rng=rng)

bias_frames  = gen.generate_bias_frames(n=30)
dark_frames  = gen.generate_dark_frames(
    exposures = [60., 120., 300., 300., 600.],
    repeats   = 3,          # 3 frames per exposure time = 15 total
)
flat_frames  = gen.generate_flat_frames(n=25, sky_adu=30_000.)
dflat_frames = gen.generate_dark_flat_frames(n=10, exposure_s=2.)
```

Each frame type uses the physically correct noise model:

| Frame | Forward model |
|-------|--------------|
| Bias | `μ[p] + N(0, σ_r[p]²)` |
| Dark (exp=t) | `μ[p] + Poisson(λ[p]·t) + N(0, σ_r[p]²)` |
| Flat (sky=S) | `μ[p] + Poisson(g[p]·S) + N(0, σ_r[p]²)` — exact Poisson, not Gaussian approx |
| Dark flat | `μ[p] + Poisson(λ[p]·t) + A[p] + N(0, σ_r[p]²)` |

### 3. Write FITS and fit the real pipeline

```python
import tempfile
from instrument_model_artifact import InstrumentModel

with tempfile.TemporaryDirectory() as tmpdir:
    gen.write_fits_folder(tmpdir, bias_frames, dark_frames,
                          flat_frames, dflat_frames)
    # Layout written:
    #   tmpdir/bias/bias_0000.fits ...
    #   tmpdir/dark/dark_0000.fits ...
    #   tmpdir/flat/flat_0000.fits ...
    #   tmpdir/dark_flat/dark_flat_0000.fits ...

    model = InstrumentModel.fit_all(tmpdir)

print(model.summary())
```

Optionally with Bayesian priors:

```python
from bayes_calibration import BayesCalibrationState

with tempfile.TemporaryDirectory() as tmpdir:
    gen.write_fits_folder(tmpdir, bias_frames, dark_frames,
                          flat_frames, dflat_frames)
    state = BayesCalibrationState.from_priors(priors, shape=(256, 256))
    model = InstrumentModel.fit_all(tmpdir, bayes_state=state)
```

### 4. Build a synthetic sky scene

```python
from synthetic_scene import SceneParams, SyntheticScene

params = SceneParams.default(shape=(256, 256))
# Or customise:
# params = SceneParams(
#     shape                 = (256, 256),
#     sky_level_adu         = 500.0,    # median sky background [ADU/s]
#     sky_gradient_strength = 0.15,     # ±7.5% top-to-bottom variation
#     vignetting_edge_frac  = 0.60,     # corners at 60% of centre brightness
#     disc_cx               = 0.35,     # disc centre: 35% across, 50% down
#     disc_cy               = 0.50,
#     disc_r_frac           = 0.08,     # radius = 8% of min(H,W)
#     disc_peak_adu         = 2000.0,   # peak above background [ADU/s]
#     gauss_cx              = 0.65,     # Gaussian centre
#     gauss_cy              = 0.50,
#     gauss_sigma_frac      = 0.04,     # sigma = 4% of min(H,W)
#     gauss_peak_adu        = 3000.0,
# )

scene = SyntheticScene(params)
print(scene.summary())
```

```
SyntheticScene
====================================================
  Shape                : 256 × 256 px
  Sky background       : 426–574 ADU/s
  Vignetting range     : 0.600–1.000
  Sky gradient         : ±15% top-to-bottom
  Disc centre          : (90, 128) px
  Disc radius          : 20.5 px
  Disc peak above bg   : 2000 ADU/s
  Disc total flux      : 2640000 ADU/s·px
  Gaussian centre      : (166, 128) px
  Gaussian sigma       : 10.2 px
  Gaussian total flux  : 1960504 ADU/s·px
  true_sky range       : 426–3574 ADU/s
```

**Scene composition:**

```
true_sky[p] = sky_bg[p] + disc[p] + gauss[p]

sky_bg[p]  = sky_level × (1 + gradient × (y/H - 0.5)) × vignetting[p]
vignetting = cos⁴(θ)   where θ scales so corners reach vignetting_edge_frac

disc[p]    = disc_peak_adu  if dist(p, centre) ≤ r  else 0
gauss[p]   = gauss_peak_adu × exp(-r² / 2σ²)
```

The vignetting is baked into `sky_bg` (not into `flat_gain`) because it
represents how many photons from an extended source reach each pixel — it
is what a flat field corrects. The `flat_gain` in `TrueInstrument` captures
only pixel-to-pixel QE variation (Bayer channel offsets + dust), normalised
to median=1.0, which is what `fit_flat` recovers.

### 5. Generate raw light frames and validate

```python
from synthetic_scene import CalibrationValidator

# Generate one raw light frame to inspect
raw = gen.generate_raw_light_frame(scene.true_sky, exposure_s=300.)

# Apply the fitted model
calibrated = model.calibrate_frame(raw, exposure_s=300.)

# Full validation: 50 frames, residual map, noise budget, region stats
validator = CalibrationValidator(model, instr, scene, gen)
report    = validator.run(exposure_s=300., n_repeat=50, rng=rng)

print(report.summary())
report.save_fits("validation_output/")
```

---

## Understanding the report

### χ² median — the headline number

The most important single number. It measures whether the noise in the
calibrated frames matches the theoretical expectation:

```
χ²[p] = σ²_empirical[p] / σ²_expected[p]

σ²_expected[p] = sky_adu[p] / g[p]        signal shot noise
               + dark_rate[p]·t / g[p]²   dark shot noise (after flat division)
               + σ_r[p]² / g[p]²           read noise (after flat division)
```

`σ²_expected` uses the **true** instrument values — it is the irreducible
noise floor that a perfect calibration would achieve.

| χ² value | Interpretation |
|----------|---------------|
| ≈ 1.0 | Correct — calibrated noise matches physical expectation |
| > 1.3 | Excess noise — calibration errors adding signal |
| < 0.7 | Under-noise — something is wrong (impossible with a correct pipeline) |

A well-calibrated run should produce χ² ∈ [0.85, 1.15].

### Parameter recovery table

Each row compares the fitted `InstrumentModel` output against `TrueInstrument`:

```
  bias_mean [ADU]   true= 280.009  est= 280.013  MAE=  0.776 ( 0.28%)  p95=2.30
```

- **MAE** — median absolute error (robust to hot pixel outliers)
- **MAE%** — MAE / true_median — fractional error
- **p95** — 95th percentile of |estimated − true| — tail behaviour

For dark rate, MAE is computed on **normal pixels only** (hot pixels excluded)
because hot pixel dark rates are intentionally not accurately recovered by
linear regression — the `DarkMixtureModel` handles those separately.

Typical good values:

| Parameter | MAE% target |
|-----------|------------|
| bias_mean | < 0.5% |
| read_noise | < 15% (limited by number of bias frames) |
| dark_rate (normal px) | < 150% (very small absolute values; % is misleading) |
| flat_gain | < 3% |

### Spatial region statistics

Three non-overlapping regions:

```
  disc           n=  1319  mean=+183  std=13568  σ_exp= 864  χ²=0.991 ✓
  gaussian       n=  1319  mean=+153  std=10468  σ_exp= 688  χ²=0.974 ✓
  background     n= 62486  mean= +39  std= 2378  σ_exp= 357  χ²=0.985 ✓
```

- **mean** — mean residual (calibrated − true_sky_adu) in the region.
  Should be close to zero (zero means no systematic flux error).
- **std** — standard deviation of residuals. Will always be >> σ_exp because
  `std` is across pixels (spatial), while `σ_exp` is per-pixel noise (temporal).
  These are not comparable directly — use χ² instead.
- **σ_exp** — sqrt(median(σ²_expected)) for this region. Typical noise per pixel per frame.
- **χ²** — median(var_empirical / var_expected) in this region. ✓ = within [0.75, 1.35].

The disc region tests hard-edge flux conservation. The Gaussian region tests
smooth signal recovery. The background tests the flat field and sky gradient
correction.

### FITS output files

| File | Content | What to look for |
|------|---------|-----------------|
| `residual.fits` | `calibrated − true_sky_adu` for one frame | Should look like pure noise — no structure, no pattern |
| `variance_empirical.fits` | Per-pixel variance from N frames | Should follow signal level (brighter regions = more variance) |
| `variance_expected.fits` | Theoretical noise budget | Same structure as empirical |
| `chi2_map.fits` | `var_emp / var_exp` | Should be uniformly ≈ 1.0. Bright spots = miscalibration |

Open in any FITS viewer (e.g. DS9, SIRIL, PixInsight). Stretch `chi2_map.fits`
linearly around 1.0. Any coherent spatial structure in the χ² map indicates a
calibration error — for example, a ring pattern would suggest the flat field is
not being divided out correctly.

---

## Testing specific failure modes

You can deliberately break the calibration and confirm the validator detects it.

### Too few bias frames (noisy read noise estimate)

```python
bias_frames = gen.generate_bias_frames(n=3)   # very few
```
Expected: read_noise MAE increases; χ² may rise slightly above 1.0 in
low-signal regions where read noise dominates.

### Single dark exposure time (unidentifiable slope)

```python
dark_frames = gen.generate_dark_frames(exposures=[300.], repeats=10)
```
Expected: `dark_rate` estimated as zero (degenerate case — `LinearRegressionAccumulator`
returns slope=0 when S_tt=0). Dark residual visible in χ² map for long exposures
where dark current is non-negligible.

### Wrong flat (different session)

```python
# Generate flats with a different rng — simulates dust moving between sessions
gen2 = CalibrationFrameGenerator(instr, rng=np.random.default_rng(999))
flat_frames_wrong = gen2.generate_flat_frames(n=25)
```
Then use the wrong flat in `write_fits_folder`. Expected: χ² rises above 1.0,
residual map shows flat field pattern, flat_gain MAE increases.

### Apply to known scene directly (no pipeline)

If you want to verify the forward model alone (no fitting):

```python
# Apply perfect calibration using true parameters
bias  = instr.bias_mean
dark  = instr.dark_rate
flat  = instr.flat_gain
exp_s = 300.

raw       = gen.generate_raw_light_frame(scene.true_sky, exp_s)
cal_ideal = (raw - bias - dark * exp_s) / flat
residual  = cal_ideal - scene.true_sky * exp_s

print(f"Ideal residual: mean={residual.mean():+.4f}  std={residual.std():.3f}")
# Expected: mean ≈ 0 (just noise), std ≈ sqrt(median(σ²_expected))
```

This tests the forward model only — no fitting error, so χ² should be very
close to 1.0 and the residual mean should be essentially zero.

---

## Notes on scale and runtime

| Shape | n_repeat | Approx runtime |
|-------|----------|---------------|
| 64 × 64 | 50 | < 1 s |
| 256 × 256 | 50 | ~ 5 s |
| 512 × 512 | 50 | ~ 20 s |
| 3008 × 3008 | 50 | ~ 10 min (CPU) |

For the full ASI533 sensor, use `SceneParams.for_asi533()` and reduce
`n_repeat_light` to 20 for a quick sanity check:

```python
report = run_full_validation(
    shape          = (3008, 3008),
    n_repeat_light = 20,
    seed           = 0,
)
```

All frame generation and fitting runs on CPU. No GPU required.

---

## Module API summary

### `synthetic_calibration.py`

| Name | Type | Purpose |
|------|------|---------|
| `TrueInstrument` | dataclass | Ground truth parameter maps + `.summary()` |
| `sample_instrument(state, rng)` | function | Draw from `BayesCalibrationState` posteriors |
| `sample_instrument_from_priors(priors, shape, rng)` | function | Draw from `SensorPriors` cold start |
| `CalibrationFrameGenerator(instr, rng)` | class | Generate frames and raw light frames |
| `→ .generate_bias_frames(n)` | method | |
| `→ .generate_dark_frames(exposures, repeats)` | method | |
| `→ .generate_flat_frames(n, sky_adu)` | method | |
| `→ .generate_dark_flat_frames(n, exposure_s)` | method | |
| `→ .write_fits_folder(dir, bias, dark, flat, dflat)` | method | Write FITS tree for `fit_all` |
| `→ .generate_raw_light_frame(sky, exposure_s)` | method | Apply forward model to scene |

### `synthetic_scene.py`

| Name | Type | Purpose |
|------|------|---------|
| `SceneParams` | dataclass | Scene configuration + `.default()` + `.for_asi533()` |
| `SyntheticScene(params)` | class | Builds `true_sky`, `sky_bg`, `disc_map`, `gauss_map`, masks |
| `CalibrationValidator(model, instr, scene, gen)` | class | Runs validation |
| `→ .run(exposure_s, n_repeat, rng)` | method | Returns `ValidationReport` |
| `ValidationReport` | dataclass | `residual_map`, `chi2_map`, `param_errors`, `region_stats` |
| `→ .summary()` | method | Formatted text report |
| `→ .save_fits(output_dir)` | method | Write four FITS diagnostic files |
| `run_full_validation(...)` | function | One-call end-to-end convenience runner |

---

## Dependencies

```
numpy      >= 1.24
astropy    >= 5.0    FITS I/O for frame writing/reading
h5py       >= 3.0    loading BayesCalibrationState from instrument.h5
```

Project modules (must be on `sys.path`):

```
bayes_calibration.py
instrument_model_artifact.py
```

Install:

```bash
pip install numpy astropy h5py
```
