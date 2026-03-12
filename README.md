# Bayesian Astro Stacker

A principled Bayesian pipeline for deep-sky astrophotography image stacking.
Every processing step is grounded in sensor noise physics. Calibration knowledge
accumulates across sessions via conjugate priors. Stacking, deconvolution, and
super-resolution are solved simultaneously as a single MAP inference problem
rather than three independent post-processing steps.

**Reference hardware:** ZWO ASI533MC-Pro OSC · Esprit 100ED refractor  
All parameters live in `SensorPriors` and `ScopeGeometry` — any camera and
telescope can be used.

---

## Table of contents

1. [Project status](#project-status)
2. [Physical noise model](#physical-noise-model)
3. [Module map](#module-map)
4. [Quick start — full pipeline](#quick-start--full-pipeline)
5. [Phase 0 — Calibration](#phase-0--calibration)
6. [Phase 1 — Optics and WCS](#phase-1--optics-and-wcs)
7. [Phase 2 — Frame characterisation](#phase-2--frame-characterisation)
8. [Phase 3 — Sufficient statistics](#phase-3--sufficient-statistics)
9. [Phase 4 — MAP super-resolution stacker](#phase-4--map-super-resolution-stacker)
10. [Display stretching](#display-stretching)
11. [Validation and ground-truth testing](#validation-and-ground-truth-testing)
12. [HDF5 file layout](#hdf5-file-layout)
13. [Installation](#installation)
14. [Design decisions](#design-decisions)
15. [Bayesian parameter reference](#bayesian-parameter-reference)

---

## Project status

| Phase | Module | Goal | Status |
|-------|--------|------|--------|
| 0a | `instrument_model_artifact.py` | Bias, dark, flat, amp-glow calibration | ✅ Done |
| 0b | `bayes_calibration.py` | Conjugate-prior cross-session learning | ✅ Done |
| 0c | `dark_mixture.py` | Soft hot/cold pixel mixture model (EM) | ✅ Done |
| 1  | `optics.py` | PSF model, WCS geometry, ASTAP plate-solve | ✅ Done |
| 2  | `frame_characterizer.py` | Per-frame seeing PSF, transparency, shift | ✅ Done |
| 3  | `sufficient_statistics.py` | Streaming Bayesian accumulation | ✅ Done |
| 4  | `map_stacker.py` | GPU MAP super-resolution + deconvolution | ✅ Done |
| 5  | `stretch.py` | Display stretch — asinh, linear, log, MTF | ✅ Done |
| —  | `bayesian_astro_stacker.py` | End-to-end combined pipeline | ✅ Done |

---

## Physical noise model

A raw ADU value at pixel *p* in frame *i*:

```
x_i^(p)  ~  Poisson( t_i · (H_i * W_i[λ])^(p) )  +  N(0, σ_r^(p)²)
```

| Symbol | Meaning |
|--------|---------|
| `λ` | True high-resolution sky scene — what the stacker solves for |
| `W_i` | Sub-pixel shift operator for frame *i* (Fourier phase shift from WCS) |
| `H_i = H_diff * H_optics * H_seeing,i` | Per-frame total PSF |
| `t_i` | Per-frame transparency ∈ (0, 1] |
| `σ_r^(p)` | Per-pixel read noise |

Stacking, deconvolution, and super-resolution are a **single joint MAP
inversion** — not three sequential post-processing steps.

---

## Module map

```
instrument_model_artifact.py   Phase 0a — calibration frame fitting and application
bayes_calibration.py           Phase 0b — conjugate priors, cross-session accumulation
dark_mixture.py                Phase 0c — 3-component Gamma EM, soft defect pixel weights
optics.py                      Phase 1  — Airy PSF, WCS geometry, ASTAP plate-solve wrapper
frame_characterizer.py         Phase 2  — per-frame PSF (Moffat), transparency, shift
sufficient_statistics.py       Phase 3  — streaming weighted accumulation, HDF5 checkpoints
map_stacker.py                 Phase 4  — PyTorch MAP solver, TV / KL / Haar regularisation
stretch.py                     Phase 5  — display stretch, debayer, PNG export
bayesian_astro_stacker.py      Combined — runs all phases end-to-end via API / CLI
synthetic_starfield.py         Testing  — synthetic star field and light frame generator
ground_truth_test.py           Testing  — full pipeline ground-truth validation suite
```

---

## Quick start — full pipeline

### CLI (simplest)

```bash
python bayesian_astro_stacker.py  \
    lights/           \           # directory of raw light FITS frames
    results/          \           # output directory (created if absent)
    --bias-dir  calibration/bias/   \
    --dark-dir  calibration/dark/   \
    --flat-dir  calibration/flat/   \
    --scale     2     \           # 2× super-resolution factor
    --alpha-tv  5e-3  \           # Total Variation regularisation strength
    --n-iter    300               # MAP optimisation iterations
```

Outputs written to `results/`:

| File | Contents |
|------|----------|
| `instrument_model.h5` | Fitted calibration model |
| `bayes_state.h5` | Bayesian posteriors — load next session |
| `sufficient_stats.h5` | Per-frame quality table + weighted sums |
| `fast_stack.fits` | Quick weighted-mean stack (no SR) |
| `lambda_hr.fits` | MAP super-resolved result at the requested scale |
| `convergence.png` | Loss curve and gradient norm plot |

### Python API

```python
from bayesian_astro_stacker import BayesianAstroStacker, PipelineConfig

cfg = PipelineConfig(
    aperture_mm     = 100.0,
    focal_length_mm = 550.0,
    pixel_size_um   = 3.76,
    scale_factor    = 2,
    map_mode        = "fast",   # "fast" (default) or "exact"
    map_n_iter      = 300,
    map_alpha_tv    = 5e-3,
)

stacker = BayesianAstroStacker(cfg)
result  = stacker.run(
    light_dir  = "lights/",
    output_dir = "results/",
    bias_dir   = "calibration/bias/",
    dark_dir   = "calibration/dark/",
    flat_dir   = "calibration/flat/",
)
print(result.summary())
```

### Resume an interrupted run

```python
result = BayesianAstroStacker(cfg).resume(
    checkpoint = "results/sufficient_stats.h5",
    model_path = "results/instrument_model.h5",
    light_dir  = "lights/",
    output_dir = "results/",
)
```

---

## Phase 0 — Calibration

### Folder layout

```
calibration/
  bias/        *.fits   — zero-exposure frames; same gain as lights; cover the lens
  dark/        *.fits   — use ≥ 2 different exposure lengths (e.g. 120 s and 300 s)
  flat/        *.fits   — well-exposed twilight or panel flats (target ~30 000 ADU)
  dark_flat/   *.fits   — same exposure as flats, lens covered (optional)
```

Frame type is auto-discovered from subfolder name **or** the `IMAGETYP` FITS header.

### Single-session fitting

```python
from instrument_model_artifact import InstrumentModel

model = InstrumentModel()
model.fit_all("calibration/")
print(model.summary())
model.save("instrument.h5")

# Apply to a light frame
from astropy.io import fits
raw, header = fits.getdata("light_001.fits", header=True)
calibrated  = model.calibrate_frame(raw, exposure_s=float(header["EXPTIME"]))
```

What `fit_all` produces:

| Array | Shape | Description |
|-------|-------|-------------|
| `bias_mean` | `[H, W]` float32 | Per-pixel bias pedestal [ADU] |
| `read_noise` | `[H, W]` float32 | Per-pixel read noise σ [ADU] |
| `dark_rate` | `[H, W]` float32 | Per-pixel dark current [ADU/s] |
| `hot_pixel_mask` | `[H, W]` bool | True where dark_rate > 5 MAD above median |
| `flat_gain` | `[H, W]` float32 | Per-pixel relative throughput (median = 1.0) |
| `flat_uncertainty` | `[H, W]` float32 | Gain estimation uncertainty |
| `amp_glow_profile` | `[H, W]` float32 | Amplifier glow residual |

### Bayesian cross-session mode — session 1

```python
from bayes_calibration import BayesCalibrationState, SensorPriors

# Factory defaults for ZWO ASI533MC-Pro at gain 100, −10 °C
priors = SensorPriors.for_asi533_gain100()

# Or configure for a different camera:
# priors = SensorPriors(
#     bias_mean_adu            = 500.0,
#     bias_mean_std_adu        = 100.0,   # wide → let data dominate quickly
#     read_noise_adu           = 3.5,
#     read_noise_concentration = 2.0,     # ~2 pseudo-observations
#     dark_rate_adu_per_s      = 0.005,
#     dark_rate_concentration  = 2.0,
#     dark_ref_temp_c          = -10.0,
#     flat_gain_std            = 1.0,
# )

state = BayesCalibrationState.from_priors(priors, shape=(3008, 3008))
model = InstrumentModel()
model.fit_all("session_1/calibration/", bayes_state=state)

model.save("instrument.h5")
state.save("instrument.h5")   # appends /bayes/ group to the same file
```

### Bayesian cross-session mode — session 2+

```python
state = BayesCalibrationState.load(
    "instrument.h5",
    new_session = True,    # resets flat gain mean; keeps all other posteriors
    new_temp_c  = -11.5,   # tonight's sensor temp — Arrhenius-corrects dark prior
)
# State now has:
#   bias / read noise posteriors carried forward (tighter than session 1)
#   dark posterior carried forward, scaled to −11.5 °C via Arrhenius
#   flat gain mean reset to 1.0 (dust may have moved)
#   flat gain variance carried forward (stable pixel QE structure)

model = InstrumentModel()
model.fit_all("session_2/calibration/", bayes_state=state)
model.save("instrument.h5")
state.save("instrument.h5")
# Repeat every session — posteriors accumulate monotonically across all nights
```

### Hot / cold pixel soft classification

The default `hot_pixel_mask` is a hard MAD-threshold. For probabilistic weights
run the EM mixture model after `fit_dark`:

```python
from dark_mixture import fit_dark_mixture

mixture = fit_dark_mixture(model.dark_rate)
print(mixture.summary())
# DarkMixtureModel Summary
#   normal : pi=0.994  E[λ]=0.002 ADU/s
#   hot    : pi=0.005  E[λ]=0.036 ADU/s   47 832 pixels
#   cold   : pi=0.001  E[λ]=0.000 ADU/s    6 870 pixels
#   EM converged in 4 iterations

# Continuous weight map for the MAP stacker — near 0 for defective pixels
weight_map = mixture.pixel_weight_map()          # [H, W] float32 ∈ [0, 1]

# Neighbour-interpolated frame for display / fast stacking
calibrated_clean, _ = mixture.calibrate_frame(calibrated)

mixture.save("instrument.h5")   # appends /dark_mixture/ group
```

---

## Phase 1 — Optics and WCS

```python
from optics import ScopeGeometry, get_instrument_psf, ASTAPSolver, compute_frame_shift

geom = ScopeGeometry(
    aperture_mm         = 100.0,
    focal_length_mm     = 550.0,
    pixel_size_um       = 3.76,
    central_obstruction = 0.0,   # refractor; set ~0.35 for SCT
)
print(f"Plate scale : {geom.plate_scale_arcsec_per_px:.3f} arcsec/px")   # ~1.41
print(f"Airy radius : {geom.airy_radius_pixels:.2f} px")

# Broadband Airy PSF kernel used as H_instrument in deconvolution
psf_instrument = get_instrument_psf(geom, kernel_size=31)  # [31, 31] float32, sum=1

# Sub-pixel frame shift via ASTAP plate solving
solver    = ASTAPSolver()          # ASTAP must be installed and on PATH
wcs_ref   = solver.solve("light_001.fits")
wcs_frame = solver.solve("light_002.fits")

shift = compute_frame_shift(wcs_frame, wcs_ref, frame_shape=(3008, 3008))
print(f"dx={shift.dx_px:+.3f} px  dy={shift.dy_px:+.3f} px  rot={shift.rotation_deg:+.4f}°")
```

---

## Phase 2 — Frame characterisation

`FrameCharacterizer` processes one calibrated frame at a time and returns a
`FrameMetadata` object containing everything the MAP stacker needs.

```python
from frame_characterizer import FrameCharacterizer
from optics import ScopeGeometry

scope = ScopeGeometry(aperture_mm=100., focal_length_mm=550., pixel_size_um=3.76)

fc = FrameCharacterizer(
    scope_geometry    = scope,
    snr_threshold     = 20.0,   # minimum star peak SNR for detection
    min_stars_for_psf = 5,      # below this, falls back to prior / Gaussian PSF
    psf_size          = 31,     # output PSF kernel side length [pixels]
    saturation_adu    = 60_000.,
)

# First frame — sets WCS and flux reference
meta_ref = fc.characterize("light_001.fits", model, exposure_s=300., is_reference=True)

# All subsequent frames
meta = fc.characterize("light_002.fits", model, exposure_s=300.)
print(meta.summary())
# FrameMetadata
#   shift        : dx=+1.234 px   dy=−0.876 px   rot=+0.0003°
#   transparency : 0.9731
#   FWHM         : 2.87 arcsec  /  2.03 px
#   n_stars      : 34
#   sky_bg       : median = 312.4 ADU
#   solve_status : wcs
```

`FrameMetadata` fields:

| Field | Type | Description |
|-------|------|-------------|
| `shift` | `FrameShift \| None` | Sub-pixel (dx, dy, rotation, scale) vs reference |
| `psf_total` | `[K, K]` float32 | Empirical total PSF from star stamps |
| `psf_seeing` | `[K, K]` float32 | Atmospheric component (instrument PSF deconvolved) |
| `transparency` | float | Flux ratio vs reference frame ∈ (0, 1] |
| `sky_bg` | `[H, W]` float32 | Smooth sky background map [ADU] |
| `fwhm_arcsec` | float | Moffat-fit FWHM [arcsec] |
| `fwhm_pixels` | float | Moffat-fit FWHM [native pixels] |
| `n_stars_used` | int | Stars contributing to PSF stack |
| `solve_status` | str | `'wcs'` or `'failed'` |

**Graceful degradation:** failed plate solve → `shift = None` (treated as zero offset);
fewer stars than `min_stars_for_psf` → prior PSF from previous frame or Gaussian fallback;
insufficient flux measurements → `transparency = 1.0`.

---

## Phase 3 — Sufficient statistics

Streams all frames in one pass and accumulates exactly what the MAP stacker needs.
Memory usage is `O(H × W)` regardless of frame count.

```python
from sufficient_statistics import SufficientStatsAccumulator

acc = SufficientStatsAccumulator(frame_shape=(3008, 3008))

for path in light_paths:
    raw, header = fits.getdata(path, header=True), fits.getheader(path)
    calibrated  = model.calibrate_frame(raw, float(header["EXPTIME"]))
    meta        = fc.characterize_calibrated(calibrated, header, exposure_s=300.)
    acc.add_calibrated(calibrated, meta)
    acc.save("sufficient_stats.h5")   # checkpoint — safe to interrupt and resume

stats = acc.finalize()
stats.save_fast_stack_fits("fast_stack.fits")   # preview before running MAP

# Select best frames by quality
from sufficient_statistics import select_best_frames
best_paths = select_best_frames(
    stats, all_paths,
    min_transparency = 0.5,
    max_fwhm_arcsec  = 3.5,
    top_n            = 30,
)
```

### Resume after interruption

```python
acc = SufficientStatsAccumulator(frame_shape=(3008, 3008))
acc.resume("sufficient_stats.h5")   # restores exact accumulator state
# continue adding remaining frames — already-processed frames are not re-added
```

---

## Phase 4 — MAP super-resolution stacker

Requires PyTorch: `pip install torch`

### Fast mode (recommended)

Uses Phase 3 sufficient statistics as input — approximately 50× faster than exact mode.

```python
from map_stacker import MapConfig, solve
from sufficient_statistics import SufficientStats

stats = SufficientStats.load("sufficient_stats.h5")

cfg = MapConfig(
    scale_factor = 2,       # output is 2× native sensor resolution
    mode         = "fast",  # uses weighted_sum as data proxy
    n_iter       = 300,
    lr           = 1e-2,
    alpha_tv     = 5e-3,    # Total Variation — smooths background, preserves edges
    alpha_kl     = 0.0,     # KL from Poisson prior — increase for very noisy data
    alpha_wav    = 0.0,     # Haar wavelet L1 — for point-source dominated fields
    device       = None,    # None → auto CUDA if available, else CPU
)

result = solve(stats, cfg)
print(f"Converged: {result.converged}   iterations: {result.n_iter}   "
      f"elapsed: {result.elapsed_s:.1f} s")

result.save_fits("lambda_hr.fits")
result.save_convergence_plot("convergence.png")
```

### Exact mode (unbiased gradient — slower)

```python
cfg = MapConfig(
    mode         = "exact",
    scale_factor = 2,
    n_iter       = 200,
    alpha_tv     = 5e-3,
    batch_size   = 4,       # frames per mini-batch — reduce if GPU runs out of memory
)
result = solve(stats, cfg, fits_paths=light_paths)
```

### Regularisation guide

| Term | Parameter | Increase when | Decrease when |
|------|-----------|---------------|---------------|
| Total Variation | `alpha_tv` | Background is noisy | Fine nebula detail is being smoothed |
| KL from prior | `alpha_kl` | Very few frames (<5) | Enough frames for data to dominate |
| Haar wavelet | `alpha_wav` | Field is mostly point stars | Rich nebulosity is present |

Start with `alpha_tv = 5e-3`, `alpha_kl = alpha_wav = 0`. Increase `alpha_tv` by 2×
until background is smooth without loss of star sharpness.

---

## Display stretching

Raw calibrated and stacked frames are linear float32 with extreme dynamic range
(stars 1 000–100 000× brighter than sky). `stretch.py` converts them to `[0, 1]`
display images for PNG export or browser rendering.

### Stretch modes

| Mode | Formula | When to use |
|------|---------|-------------|
| `asinh` | `arcsinh(β·x) / arcsinh(β)` | **Default** — matches PixInsight auto-STF |
| `linear` | Percentile clip + normalise | Sanity checks, bright / narrow DR scenes |
| `log` | `log(1 + p·x) / log(1 + p)` | Strong highlight compression |
| `midtone` | PixInsight-style MTF S-curve | Fine manual shadow / midtone / highlight control |

### Linked vs unlinked colour

**Linked** (`linked=True`, default) — one stretch derived from the luminance
channel and applied identically to R, G, B. Star colours are preserved.
Use for broadband RGB imaging.

**Unlinked** (`linked=False`) — each channel is stretched independently to fill
the full `[0, 1]` range. Use for narrowband palettes (SHO, HOO) or when channels
have very different signal levels.

### Fully automatic stretch — single call

```python
from stretch import stretch_fits

# Mono stack (calibrated or stacked FITS, no debayer needed)
img = stretch_fits("lambda_hr.fits")

# OSC raw light frame — debayer then auto-stretch, linked colour
img = stretch_fits("light_001.fits", debayer=True)

# Control mode and background target brightness
img = stretch_fits("lambda_hr.fits", mode="asinh", target_bg=0.20)

# Downscale for a fast web preview
img = stretch_fits("lambda_hr.fits", max_size=1024)
```

### Manual control with `auto_params`

```python
from stretch import auto_params, stretch_mono
import numpy as np
from astropy.io import fits

data = fits.getdata("lambda_hr.fits").astype(np.float32)

# Estimate and inspect parameters before applying
params = auto_params(
    data,
    mode          = "asinh",
    target_bg     = 0.18,    # desired mean display brightness for the sky
    bp_sigma      = -2.8,    # black point = sky_median − 2.8σ (PI STF default)
    wp_percentile = 99.9,    # white point clips top 0.1% (saturated star cores)
)
print(params)
# StretchParams(mode='asinh' bp=287.4 wp=14823.2 asinh_strength=847.3 midtone=0.18)

img = stretch_mono(data, params)
```

### Linked RGB auto-stretch

```python
from stretch import stretch_rgb
import numpy as np

# rgb is [H, W, 3] float32 — e.g. from three stacked channels
img = stretch_rgb(rgb, linked=True, mode="asinh", target_bg=0.18)
```

### Unlinked RGB — independent per-channel stretch

```python
img = stretch_rgb(
    rgb,
    linked        = False,
    mode          = "asinh",
    target_bg     = 0.18,
    wp_percentile = 99.8,
)
```

### Explicit per-channel control (narrowband palette)

```python
from stretch import StretchParams, stretch_rgb

params_r = StretchParams(black_point=80.,  white_point=4000., mode="asinh", asinh_strength=600.)
params_g = StretchParams(black_point=60.,  white_point=3000., mode="asinh", asinh_strength=400.)
params_b = StretchParams(black_point=60.,  white_point=3000., mode="asinh", asinh_strength=400.)

img = stretch_rgb(rgb, linked=False,
                  params_r=params_r, params_g=params_g, params_b=params_b)
```

### Midtone Transfer Function (MTF) — fine manual control

```python
from stretch import StretchParams, stretch_mono

# midtone < 0.5 → lift shadows   midtone > 0.5 → push midtones down
params = StretchParams(
    black_point = 250.,
    white_point = 10000.,
    mode        = "midtone",
    midtone     = 0.20,   # background pixels map to ~20% display brightness
)
img = stretch_mono(data, params)
```

### OSC debayer preview

```python
from stretch import debayer_preview, stretch_rgb
from astropy.io import fits

raw = fits.getdata("light_001.fits").astype("float32")   # [H, W] Bayer mosaic

# Bilinear debayer → [H, W, 3] float32  (preview quality, not science)
rgb = debayer_preview(raw, pattern="RGGB")   # RGGB / BGGR / GRBG / GBRG

img = stretch_rgb(rgb, linked=True, mode="asinh")
```

> Do not debayer before stacking. The R / G / B sub-pixel positions within each
> 2×2 super-pixel are the spatial diversity that enables super-resolution.
> Debayering discards this permanently.

### Export to PNG

```python
from stretch import to_png_bytes, save_png

# In-memory bytes — for Flask / HTTP responses
png = to_png_bytes(img)   # bytes, starts with b'\x89PNG'

# Write to disk
save_png(img, "preview.png")
```

### Histograms (for UI display)

```python
from stretch import histogram, rgb_histograms

# Mono — returns (bin_centres, counts) each [n_bins] float32
centres, counts = histogram(img, n_bins=256, log=False)

# RGB — dict with keys 'r', 'g', 'b', 'lum'
hists = rgb_histograms(img_rgb, n_bins=256)
r_centres, r_counts = hists["r"]
```

### Query auto-params without stretching (for UI inspection)

```python
from stretch import get_channel_params

info = get_channel_params(data, linked=True, mode="asinh")
# {'linked': True, 'params': StretchParams(...)}

info_ul = get_channel_params(rgb, linked=False, mode="asinh")
# {'linked': False, 'r': StretchParams(...), 'g': ..., 'b': ...}
```

### `auto_params` parameter reference

| Parameter | Default | Effect |
|-----------|---------|--------|
| `mode` | `'asinh'` | Stretch algorithm |
| `target_bg` | `0.18` | Desired mean display value for sky background ∈ (0, 0.5] |
| `bp_sigma` | `-2.8` | Black point = sky_median + bp_sigma × sky_MAD. Negative places BP below sky (PI STF default). Positive clips background to black. |
| `wp_percentile` | `99.9` | Percentile used for white point — clips saturated star cores |
| `asinh_strength` | auto | Override auto-computed β. Higher = more aggressive highlight compression |
| `midtone` | auto | MTF midtone knob ∈ (0, 1). Only used in `'midtone'` mode |

---

## Validation and ground-truth testing

### Synthetic star field generator

```python
from synthetic_starfield import StarfieldConfig, generate_starfield

cfg   = StarfieldConfig.for_asi533(n_stars=150, n_frames=20)
truth = generate_starfield(cfg, seed=42)

print(truth.summary())
# StarfieldGroundTruth
#   LR shape     : 256 × 256
#   HR shape     : 512 × 512  (2× scale)
#   Stars        : 150
#   Light frames : 20
#   Seeing FWHM  : 3.02 ± 0.29 LR px
#   Transparency : 0.951 ± 0.048

# Ground truth arrays
truth.true_scene_hr       # [512, 512] float32 — true sky at HR
truth.true_shifts         # list of (dx, dy) per frame [LR px]
truth.true_fwhm_lr_px     # list of true seeing FWHM per frame
truth.true_transparency   # list of true t_i per frame
truth.light_fits_paths    # list[Path] — synthetic raw FITS frames

truth.cleanup()           # delete temp dir when done
```

### Ground-truth pipeline validation

```bash
# Quick smoke test (~30 seconds)
python ground_truth_test.py --fast

# Full test including MAP solver
python ground_truth_test.py --full --output-dir validation_results/

# Custom
python ground_truth_test.py \
    --shape 128 128 --n-frames 15 --n-iter 150 \
    --output-dir results/ --verbose
```

Or from Python:

```python
from ground_truth_test import run_ground_truth_test, TestConfig

report = run_ground_truth_test(TestConfig.fast())
print(report.summary())

# ╔══════════════════════════════════════════════════════╗
# ║         BAYESIAN ASTRO STACKER — VALIDATION          ║
# ╠══════════════════════════════════════════════════════╣
# ║  PHASE 0 — Calibration accuracy                      ║
# ║    Bias MAE            0.31%    [thresh <2%]   PASS ║
# ║    Read noise MAE      1.85%    [thresh <5%]   PASS ║
# ║  PHASE 2 — Frame characterisation                    ║
# ║    FWHM median error   0.21 px  [thresh <0.5] PASS  ║
# ║    Transparency error  0.023    [thresh <0.05] PASS ║
# ║  PHASE 3 — Fast stack quality                        ║
# ║    Pearson r           0.983    [thresh >0.90] PASS ║
# ╚══════════════════════════════════════════════════════╝

n_pass, n_total = report.n_passed()
```

Pass / fail thresholds:

| Metric | Threshold |
|--------|-----------|
| Bias MAE | < 2% |
| Read noise MAE | < 5% |
| Dark rate MAE (normal pixels) | < 10% |
| Flat gain MAE | < 3% |
| FWHM recovery error (median) | < 0.5 LR px |
| Transparency error (median) | < 0.05 |
| Shift recovery error (median) | < 0.5 LR px |
| Fast stack Pearson r | > 0.90 |
| MAP Pearson r | > 0.90 |
| SR power above Nyquist | > 1.10× |

---

## HDF5 file layout

All modules share one `.h5` file; each appends its own group without touching others.

```
instrument.h5
  /bias/
    mean                [H, W]  float32   per-pixel bias pedestal [ADU]
    read_noise          [H, W]  float32   per-pixel read noise σ [ADU]
  /dark/
    rate                [H, W]  float32   dark current [ADU/s]
    hot_pixels          [H, W]  bool      hard MAD-threshold mask
  /flat/
    gain                [H, W]  float32   throughput map (median = 1.0)
    uncertainty         [H, W]  float32   gain estimation uncertainty
  /dark_flat/
    amp_glow            [H, W]  float32   amplifier glow profile
  /metadata                               scalar HDF5 attributes
      frame_shape, gain_setting, sensor_temp_c, bayer_pattern, fit_date

  /bayes/                                 present when bayes_state is used
    bias/
      mu_n              [H, W]  float64   posterior mean
      tau2_n            [H, W]  float64   posterior variance
      n                 scalar  int       frames accumulated
    read_noise/
      alpha_n           [H, W]  float64
      beta_n            [H, W]  float64
      n                 scalar  int
    dark/
      alpha_n           [H, W]  float64
      beta_n            [H, W]  float64   (units: seconds)
      ref_temp_c        scalar  float64
      n                 scalar  int
    flat/
      g_n               [H, W]  float64   posterior mean gain
      v_n               [H, W]  float64   posterior variance
      n                 scalar  int

  /dark_mixture/                          present when fit_dark_mixture() called
    class_probs         [H, W, 3]  float32   (p_normal, p_hot, p_cold)
    labels              [H, W]     uint8     hard MAP label
    pi                  [3]        float64   mixing weights
    alpha / beta        [3]        float64   Gamma shape / rate per component

sufficient_stats.h5
  /stats/
    weighted_sum        [H, W]  float64
    weight_sum          [H, W]  float64
    transparency        [N]     float32   per-frame
    fwhm                [N]     float32   per-frame [arcsec]
    psf_kernels/0…N-1   [K, K]  float32   per-frame PSF kernels
    shifts/dx_px        [N]     float32
    shifts/dy_px        [N]     float32
```

Inspect without Python:

```bash
h5ls -r instrument.h5
# or open in HDFView — https://www.hdfgroup.org/downloads/hdfview/
```

---

## Installation

### Python packages

```bash
pip install astropy numpy scipy h5py psutil Pillow
```

For GPU-accelerated MAP stacking:

```bash
pip install torch    # see pytorch.org for GPU-specific wheel selection
```

Full dependency table:

| Package | Min version | Required for |
|---------|-------------|-------------|
| `astropy` | 5.0 | FITS I/O, WCS parsing |
| `numpy` | 1.24 | All array operations |
| `scipy` | 1.9 | PSF (Bessel J₁), dark EM, Wiener filter |
| `h5py` | 3.0 | HDF5 serialisation |
| `psutil` | 5.9 | RAM-aware chunk sizing |
| `Pillow` | 9.0 | PNG export in `stretch.py` |
| `torch` | 2.0 | MAP stacker (Phase 4) |
| `matplotlib` | any | Convergence and validation plots |

### External: ASTAP plate solver

Used for sub-pixel WCS shift extraction. Any WCS-capable solver that writes
standard `CTYPE1/2`, `CRPIX`, `CRVAL`, `CD1_1` FITS headers also works.

Download: https://www.hnsky.org/astap.htm  
ASTAP must be on `PATH`, or pass `astap_path=` to `ASTAPSolver`.

### Recommended acquisition settings (ASI533 + Esprit 100ED)

| Setting | Value | Reason |
|---------|-------|--------|
| Gain | 100 | Unity gain; read noise ~1.7 e⁻ |
| Sensor temperature | −10 °C | Dark current ~0.005 e⁻/s |
| Dither | ±3–4 LR pixels between frames | Required for super-resolution |
| Exposure | 300 s | Shot-noise dominated at typical dark sites |
| Bias frames | 20+ | Covers read noise estimation |
| Dark frames | 5+ at ≥ 2 exposure lengths | Enables slope regression |
| Flat frames | 20+ at ~30 000 ADU | High-SNR gain calibration |

---

## Design decisions

### Why calibrate on the raw Bayer mosaic

The R, G₁, G₂, B sub-pixels sit at half-pixel offsets within each 2×2 super-pixel.
Across multiple dithered frames these sample the scene at different sub-pixel phases —
the spatial diversity the MAP stacker exploits for super-resolution. Debayering before
stacking discards this permanently and replaces it with interpolation artefacts.

### Why MAP rather than sigma-clipping stacking

Traditional pipelines run sequential independent steps (calibrate → align → reject →
combine → deconvolve), each introducing its own approximation. The MAP formulation
inverts the complete forward model simultaneously: PSF, sub-pixel shifts, transparency,
shot noise, and read noise all appear as explicit terms in the same objective. The
solution is consistent with all of them at once.

### Why asinh as the default stretch

Linear stretch clips background to black and burns stars simultaneously.
Log is so aggressive it makes noise look like signal. `arcsinh(β·x) / arcsinh(β)`
is nearly linear near zero (preserving sky noise texture) and nearly logarithmic at
high values (compressing star cores). The single free parameter β is solved
automatically to place the background at a chosen mean display brightness.

The `bp_sigma = -2.8` default places the black point 2.8σ below the sky median —
matching PixInsight's STF — ensuring sky pixels appear above zero in the display.

### Why conjugate priors rather than MCMC

All four calibration noise models have natural exponential-family conjugate pairs
with closed-form posteriors expressible as per-pixel array additions. For a 3008×3008
sensor: ~9 M float64 values per accumulator, updating in under 1 second per frame on
CPU. MCMC would be 100–1000× slower for identical posterior accuracy.

### Why the flat gain mean resets between sessions

Dust migrates across the sensor window and optical surfaces between nights. A
vignetting pattern from a rotated filter or a new dust mote is session-specific.
The *variance* of gain (pixel-to-pixel QE variation intrinsic to the sensor) is a
stable device property and correctly carries forward.

### Why hot pixel classification is probabilistic

A hard binary mask classifies every pixel at maximum certainty — even marginal cases.
The Gamma mixture model produces a continuous `p_normal ∈ [0, 1]` per pixel. A pixel
with `p_normal = 0.6` contributes 60% of its normal weight to the stacker rather than
being fully included or excluded. This is strictly more information than a mask and
avoids the threshold selection problem entirely.

### Memory model

Every `fit_*` and accumulation method streams frames in chunks auto-sized to available
RAM with a 2 GB headroom. Accumulator state is `O(H × W)`. For the ASI533 (3008×3008):
each float64 accumulator array is ~72 MB. Peak memory during calibration fitting:

```
~5 × H × W × 8 bytes  ≈  360 MB   (ASI533 full frame, independent of frame count)
```

---

## Bayesian parameter reference

### Conjugate prior pairs

| Parameter | Likelihood | Prior → Posterior | Update per frame |
|-----------|-----------|-------------------|-----------------|
| Bias mean μ | Gaussian(μ, σ_r²) | Gaussian → Gaussian | precision += 1/σ_r²; weighted sum += x/σ_r² |
| Read noise σ² | Gaussian(μ, σ²) | InvGamma → InvGamma | α += ½; β += ½(x−μ)² |
| Dark rate λ | Poisson(λ·t) | Gamma → Gamma | α += x_counts; β += t |
| Flat gain g | Gaussian(g·S, S+σ_r²) | Gaussian → Gaussian | precision += S²/(S+σ_r²) |

All updates are closed-form per-pixel array additions — no MCMC, no iteration.

### Cross-session carry-over policy

| Parameter | Policy |
|-----------|--------|
| Bias mean | Carry forward — extremely stable (±1 ADU over months) |
| Read noise | Carry forward — stable sensor property |
| Dark rate | Carry forward; Arrhenius-scaled to new temperature |
| Flat gain **mean** | **Reset to 1.0** — dust moves, optics may change |
| Flat gain **variance** | Carry forward — encodes stable pixel QE structure |

### Dark temperature correction (Arrhenius)

```
λ_dark(T) = λ_dark(T_ref) · exp( −E_g / (2k) · (1/T − 1/T_ref) )
```

E_g = 1.12 eV (silicon band gap), k = 8.617 × 10⁻⁵ eV/K.  
Rule of thumb: dark current roughly doubles every 5.5–7 °C.

```python
from bayes_calibration import dark_rate_temperature_correction

corrected = dark_rate_temperature_correction(model.dark_rate, from_c=-10., to_c=-7.)
```
