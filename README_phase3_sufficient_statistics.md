# Phase 3 — Sufficient Statistics Accumulator

`sufficient_statistics.py` streams all light frames through a single
memory-efficient pass and accumulates everything the Phase 4 MAP stacker needs.

The accumulator holds only **six `[H, W]` float64 arrays** in memory at any
time, regardless of how many frames exist.  For a 3008 × 3008 sensor that is
~420 MB.  Individual frames are loaded, processed, and discarded immediately.

---

## What is accumulated

### Per-frame (stored in lists)
| Attribute | Type | Description |
|---|---|---|
| `shift_list` | `list[FrameShift\|None]` | Sub-pixel shift per frame |
| `psf_list` | `list[ndarray K×K]` | Total PSF kernel per frame |
| `transparency_list` | `list[float]` | t_i per frame |
| `fwhm_list` | `list[float]` | Seeing FWHM in arcsec per frame |

### Pixel-grid (running sums)
| Array | Formula | Use |
|---|---|---|
| `weighted_sum` | Σ t_i · (x_i − sky_i) | MAP stacker data term |
| `weight_sum` | Σ t_i | Normalisation |
| `sky_sum` | Σ sky_i | Mean sky model |
| `sq_sum` | Σ t_i · (x_i − sky_i)² | Per-pixel variance estimate |

### Derived on demand (no extra storage)
| Property | Formula | Description |
|---|---|---|
| `weighted_mean` | weighted_sum / weight_sum | **Fast preview stack** — ready immediately |
| `sky_mean` | sky_sum / N | Mean sky background |
| `variance_map` | sq_sum/weight_sum − mean² | Per-pixel noise map |
| `quality_map` | weight_sum / max(weight_sum) | Coverage weight in [0,1] |

---

## Dependencies

```
numpy >= 1.24
h5py  >= 3.0
astropy >= 5.0
frame_characterizer.py   (Phase 2)
instrument_model_artifact.py  (Phase 0)
```

Install:
```bash
pip install numpy h5py astropy
```

---

## Quickstart — full pipeline

```python
import logging
logging.basicConfig(level=logging.INFO)

from pathlib import Path
from optics import ScopeGeometry, ASTAPSolver
from instrument_model_artifact import InstrumentModel
from frame_characterizer import FrameCharacterizer
from sufficient_statistics import SufficientStatsAccumulator

# Load Phase 0 calibration model
model = InstrumentModel.load("instrument.h5")

# Set up Phase 2 characterizer (shared across all frames)
scope  = ScopeGeometry(aperture_mm=100, focal_length_mm=550, pixel_size_um=3.76)
solver = ASTAPSolver()
fc     = FrameCharacterizer(scope, astap_solver=solver)

# Collect all raw light FITS files
light_paths = sorted(Path("lights/").glob("*.fits"))

# Stream through all frames
acc = SufficientStatsAccumulator(frame_shape=(3008, 3008))

for i, path in enumerate(light_paths):
    acc.add_frame(path, model, fc,
                  exposure_s=300.,
                  is_reference=(i == 0))   # first frame is the reference

    # Optional: checkpoint every 20 frames for crash recovery
    if (i + 1) % 20 == 0:
        acc.save(f"checkpoint_{i+1:04d}.h5")
        print(f"  checkpoint saved at frame {i+1}")

# Finalise
stats = acc.finalize()
print(stats.summary())

# Save complete stats for Phase 4 MAP stacker
stats.save("sufficient_stats.h5")

# Write fast-preview stack as FITS (no GPU, no deconvolution)
stats.save_fast_stack_fits("fast_stack.fits")
```

---

## Quickstart — minimal (in-memory, no FITS files)

Useful for testing with synthetic data:

```python
import numpy as np
from frame_characterizer import FrameMetadata, _gaussian_psf
from sufficient_statistics import SufficientStatsAccumulator

H, W = 3008, 3008
rng  = np.random.default_rng(0)

def make_meta(t=1.0, fwhm=2.5):
    psf = _gaussian_psf(31, fwhm_px=fwhm)
    sky = np.full((H, W), 120.0, dtype=np.float32)
    return FrameMetadata(
        shift=None, psf_total=psf, psf_seeing=psf,
        transparency=t, sky_bg=sky,
        fwhm_arcsec=fwhm*1.41, fwhm_pixels=fwhm,
        n_stars_used=15, solve_status='failed',
    )

acc = SufficientStatsAccumulator(frame_shape=(H, W))

for i in range(20):
    frame = rng.poisson(150, (H, W)).astype(np.float32)
    meta  = make_meta(t=0.9 + 0.1*rng.random(), fwhm=2.5 + rng.random())
    acc.add_calibrated(frame, meta)

stats = acc.finalize()
print(stats.summary())
fast_stack = stats.weighted_mean   # [H, W] float32 — ready to display
```

---

## Crash recovery and resume

```python
# Session A — accumulate 60 frames, save checkpoint
acc = SufficientStatsAccumulator()
for i, path in enumerate(light_paths[:60]):
    acc.add_frame(path, model, fc, exposure_s=300., is_reference=(i==0))
acc.save("session_checkpoint.h5")

# Session B — resume and add the remaining frames
acc = SufficientStatsAccumulator.resume("session_checkpoint.h5")
for path in light_paths[60:]:
    acc.add_frame(path, model, fc, exposure_s=300.)
stats = acc.finalize()
stats.save("sufficient_stats_final.h5")
```

---

## Selecting the best frames

Two utility functions help quality-filter before Phase 4:

```python
from sufficient_statistics import select_best_frames, rebuild_stats_from_subset

# Keep best 50% by seeing FWHM
best_idx = select_best_frames(stats, top_frac=0.5, key='fwhm')

# Or keep best 75% by transparency
best_idx = select_best_frames(stats, top_frac=0.75, key='transparency')

# Build a new SufficientStats for the selected subset
stats_best = rebuild_stats_from_subset(stats, best_idx)
print(f"Using {stats_best.frame_count}/{stats.frame_count} frames")
stats_best.save("stats_best.h5")
```

> **Note:** `rebuild_stats_from_subset` rescales the pixel arrays
> proportionally to the subset weight.  This is an approximation.
> For exact pixel-level filtering, run `add_frame()` only on the
> selected frames from the beginning.

---

## HDF5 file layout

```
sufficient_stats.h5
└── stats/
    ├── weighted_sum      [H, W] float32  gzip-4
    ├── weight_sum        [H, W] float32  gzip-4
    ├── sky_sum           [H, W] float32  gzip-4
    ├── sq_sum            [H, W] float32  gzip-4
    ├── transparency      [N]    float32
    ├── fwhm              [N]    float32
    ├── psf_kernels/
    │   ├── 0             [K, K] float32  gzip-4
    │   ├── 1             [K, K] float32
    │   └── …
    └── shifts/
        ├── dx_px         [N]    float32   (-9999 = solve failed)
        ├── dy_px         [N]    float32
        ├── rot_deg       [N]    float32
        └── scale         [N]    float32
```

Frame-count and shape are stored as HDF5 group attributes:
```
stats.attrs["frame_count"]   int
stats.attrs["frame_shape"]   [H, W]
```

---

## API reference

### `SufficientStatsAccumulator`

```python
acc = SufficientStatsAccumulator(
    frame_shape   = None,   # (H, W) or None (inferred from first frame)
    outlier_sigma = 3.0,    # warn if transparency < median - σ×MAD
)
```

| Method | Description |
|---|---|
| `add_frame(fits_path, model, fc, exposure_s, is_reference)` | Load → calibrate → characterise → accumulate |
| `add_calibrated(calibrated, meta)` | Accumulate a pre-characterised frame |
| `finalize()` | Return `SufficientStats` (non-destructive) |
| `save(path)` | Checkpoint to HDF5 |
| `resume(path)` | Class method — restore from checkpoint |
| `transparency_stats()` | dict of min/median/max/std |
| `fwhm_stats()` | dict of min/median/max/best_n |

### `SufficientStats`

```python
stats = SufficientStats.load("sufficient_stats.h5")
```

| Property / Method | Description |
|---|---|
| `weighted_mean` | Fast preview stack [H, W] float32 |
| `sky_mean` | Mean sky model [H, W] float32 |
| `variance_map` | Per-pixel empirical variance [H, W] float32 |
| `quality_map` | Coverage weight ∈ [0,1] [H, W] float32 |
| `mean_fwhm_arcsec` | Median seeing across frames |
| `mean_transparency` | Median transparency |
| `summary()` | Human-readable summary string |
| `save(path)` | Write to HDF5 |
| `save_fast_stack_fits(path)` | Write weighted_mean as FITS |

### Utility functions

```python
# Top-fraction frame selection
indices = select_best_frames(stats, top_frac=0.5, key='fwhm')
# key: 'fwhm' (lower=better) or 'transparency' (higher=better)

# Rebuild pixel stats for a subset
stats_sub = rebuild_stats_from_subset(stats, indices)
```

---

## Running the unit tests

```bash
cd /path/to/project
python3 -m pytest test_sufficient_statistics.py -v
```

Or run inline:

```bash
python3 - << 'EOF'
import sys, numpy as np, tempfile, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.WARNING)

from sufficient_statistics import SufficientStats, SufficientStatsAccumulator, select_best_frames
from frame_characterizer import FrameMetadata, _gaussian_psf
from pathlib import Path

rng = np.random.default_rng(0)
H, W, N = 64, 64, 6

def make_meta(t=1.0, fwhm=2.5):
    psf = _gaussian_psf(15, fwhm)
    sky = np.full((H,W), 100., dtype='f4')
    return FrameMetadata(None, psf, psf, t, sky, fwhm*1.41, fwhm, 8, 'failed')

def make_frame():
    f = np.full((H,W), 100., dtype='f4')
    f[28:34,28:34] += 200.
    return f + rng.normal(0,5,(H,W)).astype('f4')

acc = SufficientStatsAccumulator(frame_shape=(H,W))
for _ in range(N):
    acc.add_calibrated(make_frame(), make_meta(0.9+0.1*rng.random()))

stats = acc.finalize()
assert stats.frame_count == N
assert float(stats.weighted_mean[30,30]) > float(stats.weighted_mean[5,5]) + 100

with tempfile.TemporaryDirectory() as td:
    p = Path(td)/'s.h5'
    stats.save(p)
    s2 = SufficientStats.load(p)
    assert s2.frame_count == N
    assert np.allclose(stats.weighted_sum, s2.weighted_sum, atol=1e-3)

idx = select_best_frames(stats, 0.5, 'fwhm')
assert len(idx) == N//2

print(f"All tests passed. weighted_mean source={stats.weighted_mean[30,30]:.1f} ADU")
EOF
```

---

## Memory and performance

| Sensor | Float64 accumulator RAM | Expected time / frame (CPU) |
|---|---|---|
| 1024 × 1024 | ~50 MB | ~0.3 s |
| 2048 × 2048 | ~200 MB | ~1.0 s |
| 3008 × 3008 | ~430 MB | ~2.5 s |

Time per frame is dominated by Phase 2 characterisation (sky background fit
and star extraction).  The accumulation step itself is O(H×W) and negligible.

Checkpointing every 20 frames adds ~10 s for gzip-compressed HDF5 I/O on a
3008 × 3008 sensor.

---

## Integration with Phase 4 (MAP stacker)

The Phase 4 stacker reads `SufficientStats` directly:

```python
from sufficient_statistics import SufficientStats

stats = SufficientStats.load("sufficient_stats.h5")

# What Phase 4 uses:
shifts         = stats.shift_list          # list[FrameShift|None]
psfs           = stats.psf_list            # list[ndarray K×K]
transparencies = stats.transparency_list   # list[float]
weighted_sum   = stats.weighted_sum        # [H,W] float32 — stacker data term
weight_sum     = stats.weight_sum          # [H,W] float32 — stacker weight
quality        = stats.quality_map         # [H,W] float32 — prior weight
sky_model      = stats.sky_mean            # [H,W] float32 — for residual check
```

The `weighted_mean` serves as the initialisation point for the MAP
optimisation — starting the solver near the Gamma-Poisson posterior mean
dramatically reduces the number of iterations needed.
