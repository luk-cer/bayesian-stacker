# Phase 2 — Frame Characterizer

`frame_characterizer.py` characterises each calibrated light frame and
produces all per-frame quantities the Phase 4 MAP stacker needs:

| Output | Symbol | Description |
|---|---|---|
| `shift` | W_i | Sub-pixel (dx, dy, rotation, scale) vs reference frame |
| `psf_total` | H_total,i | Empirical full-system PSF from star stamps |
| `psf_seeing` | H_seeing,i | Atmospheric component (Wiener-deconvolved) |
| `transparency` | t_i | Per-frame throughput ∈ (0, 1] |
| `sky_bg` | — | Smooth 2-D sky background model [ADU] |
| `fwhm_arcsec` | — | Seeing FWHM from Moffat fit |

---

## Dependencies

```
numpy >= 1.24
scipy >= 1.9
astropy >= 5.0
optics.py          (project module — Phase 1)
```

Install:
```bash
pip install numpy scipy astropy
```

---

## Pipeline (one frame)

```
raw FITS
  │
  ▼  model.calibrate_frame()          Phase 0 InstrumentModel
calibrated [H, W] float32
  │
  ├─▶  estimate_sky_background()      sigma-clip + 2-D Legendre polynomial
  │      sky_bg [H, W]
  │
  ├─▶  extract_stars()                local-max detection + stamp extraction
  │      positions [N,2]  stamps [N×K×K]
  │
  ├─▶  estimate_psf_from_stamps()     normalise → align → median stack
  │      psf_total [K,K]  fwhm_px  beta
  │
  ├─▶  deconvolve_instrument_psf()    Wiener deconvolution of H_instrument
  │      psf_seeing [K,K]
  │
  ├─▶  estimate_transparency()        aperture photometry vs reference fluxes
  │      t_i float
  │
  └─▶  extract_wcs_geometry()         WCS from FITS header → FrameShift
         shift (dx, dy, rotation, scale)
```

---

## Quickstart

```python
import logging
logging.basicConfig(level=logging.INFO)

from optics import ScopeGeometry, ASTAPSolver
from instrument_model_artifact import InstrumentModel
from frame_characterizer import FrameCharacterizer

# 1. Load calibration model (built in Phase 0)
model = InstrumentModel.load("instrument.h5")

# 2. Set up the characterizer (once per session)
scope  = ScopeGeometry(aperture_mm=100, focal_length_mm=550, pixel_size_um=3.76)
solver = ASTAPSolver()                        # optional, needs ASTAP installed
fc     = FrameCharacterizer(scope, astap_solver=solver)

# 3. Characterise the reference frame (first or best-seeing frame)
from pathlib import Path
light_paths = sorted(Path("lights/").glob("*.fits"))

ref_meta = fc.characterize(light_paths[0], model, exposure_s=300.,
                            is_reference=True)
print(ref_meta.summary())

# 4. Characterise all remaining frames
metas = [ref_meta]
for path in light_paths[1:]:
    meta = fc.characterize(path, model, exposure_s=300.)
    metas.append(meta)
    print(f"{path.name}: t={meta.transparency:.3f}  FWHM={meta.fwhm_arcsec:.2f}\"")
```

### Using pre-calibrated arrays (for integration with Phase 3)

```python
from astropy.io import fits

with fits.open("light_001.fits") as hdul:
    raw    = hdul[0].data.astype("float32").squeeze()
    header = hdul[0].header

cal  = model.calibrate_frame(raw, exposure_s=300.)
meta = fc.characterize_calibrated(cal, header, exposure_s=300.,
                                   is_reference=True)
```

---

## Constructor parameters

| Parameter | Default | Description |
|---|---|---|
| `scope_geometry` | required | `ScopeGeometry` instance |
| `astap_solver` | `None` | `ASTAPSolver` instance; `None` disables shift estimation |
| `psf_size` | 31 | PSF kernel side length (pixels, forced odd) |
| `stamp_size` | 41 | Star cutout side length (should be ≥ psf_size, forced odd) |
| `snr_threshold` | 20.0 | Minimum star peak SNR for extraction |
| `saturation_adu` | 60 000 | Reject stars with any pixel above this |
| `poly_degree` | 2 | Sky background polynomial degree (2=quadratic, 3=cubic) |
| `min_stars_for_psf` | 5 | Fall back to prior PSF if fewer stars found |
| `wiener_snr` | 10.0 | Wiener regularisation SNR (higher = sharper deconvolution) |
| `aperture_r_px` | 8.0 | Photometry aperture radius [pixels] |

---

## Tuning guide

### Too few stars detected
Lower `snr_threshold` (try 10) or increase `saturation_adu` if most bright
stars are being rejected.  Check `n_stars_used` in `FrameMetadata.summary()`.

### PSF looks noisy / ringing
Increase `psf_size` to 63 for better oversampling.  Lower `wiener_snr` to
5–8 if H_seeing has ringing artefacts.

### Sky background follows nebulosity
Lower `poly_degree` to 1 (linear), increase `box_size` in
`estimate_sky_background()` directly, or lower `sigma_clip` to 2.5.

### Transparency estimates unstable
Ensure the reference frame is truly photometric.  Increase `aperture_r_px`
for brighter, more isolated stars.

### Plate solving fails
Verify ASTAP is installed and index files are downloaded.  Supply a closer
RA/Dec hint via `ASTAPSolver(search_radius_deg=2.0)`.  Pre-solved FITS files
(WCS already in header) bypass ASTAP entirely.

---

## Public functions

### `estimate_sky_background(frame, poly_degree, sigma_clip, n_iter, box_size)`
Estimates smooth 2-D sky background via tiled sigma-clipping and a 2-D
Legendre polynomial fit.  Returns `[H, W] float32`.

### `extract_stars(frame, sky_bg, snr_threshold, saturation_adu, min_sep_px, stamp_size, max_stars)`
Detects isolated, unsaturated stars.  Returns `(positions [N,2], stamps list-of-arrays)`.

### `fit_moffat(stamp)`
Fits a 2-D Moffat profile to a background-subtracted stamp.
Returns `(fwhm_px, beta)`.

### `estimate_psf_from_stamps(stamps, psf_size)`
Median-stacks Fourier-aligned, normalised stamps.
Returns `(psf [K,K], fwhm_px, beta)`.

### `deconvolve_instrument_psf(psf_total, psf_instrument, wiener_snr)`
Wiener deconvolution: recovers H_seeing from H_total and H_instrument.
Returns `[K,K] float32`.

### `estimate_transparency(positions, calibrated, sky_bg, ref_fluxes, aperture_r)`
Aperture photometry comparison vs reference frame.
Returns `float ∈ (0, 1]`.

---

## Running the unit tests

```bash
cd /path/to/project
python3 -m pytest test_frame_characterizer.py -v
```

Or run the self-contained test block directly:

```bash
python3 - << 'EOF'
import sys; sys.path.insert(0, '.')
import numpy as np, logging
logging.basicConfig(level=logging.WARNING)
from frame_characterizer import (
    estimate_sky_background, extract_stars, fit_moffat,
    estimate_psf_from_stamps, deconvolve_instrument_psf,
    estimate_transparency, _gaussian_psf, _fourier_shift,
)

rng = np.random.default_rng(0)
H, W = 128, 128

# Sky background on flat field
flat = np.full((H,W), 500., dtype=np.float32) + rng.normal(0,5,(H,W)).astype('f4')
sky = estimate_sky_background(flat)
assert abs(float(sky.mean()) - 500.) < 5., f"Sky mean off: {sky.mean()}"
print("✓ sky_background")

# Star extraction
frame = flat.copy()
for cy,cx in [(40,40),(80,80),(30,90)]:
    yg,xg = np.mgrid[0:H,0:W]
    frame += (3000*np.exp(-((xg-cx)**2+(yg-cy)**2)/8)).astype('f4')
sky2 = estimate_sky_background(frame)
pos, stamps = extract_stars(frame, sky2, snr_threshold=10., stamp_size=21)
assert len(pos) >= 2, f"Only {len(pos)} stars"
print(f"✓ extract_stars: {len(pos)} stars")

# PSF estimation
fake = [_gaussian_psf(21,3.5)*1000. + rng.normal(0,.01,(21,21)).astype('f4')
        for _ in range(8)]
psf, fwhm, beta = estimate_psf_from_stamps(fake, psf_size=21)
assert abs(psf.sum()-1.)<1e-4 and 2.<fwhm<6., f"PSF check failed fwhm={fwhm}"
print(f"✓ estimate_psf_from_stamps: FWHM={fwhm:.2f} px")

# Fourier shift round-trip
arr = _gaussian_psf(31, 3.0).astype('f8')
rt  = _fourier_shift(_fourier_shift(arr, 1.5, -0.8), -1.5, 0.8)
assert np.allclose(arr, rt, atol=1e-9)
print("✓ Fourier shift round-trip")

# Wiener deconvolution
from scipy.ndimage import convolve
Hi = _gaussian_psf(31,1.2).astype('f8')
Hs = _gaussian_psf(31,3.0).astype('f8')
Ht = convolve(Hi,Hs,mode='wrap'); Ht/=Ht.sum()
Hs_rec = deconvolve_instrument_psf(Ht.astype('f4'), Hi.astype('f4'), 20.)
assert abs(Hs_rec.sum()-1.)<1e-4
print("✓ deconvolve_instrument_psf")

print("\nAll tests passed.")
EOF
```

---

## Integration with Phase 3

`FrameCharacterizer.characterize_calibrated()` is designed to be called from
the `SufficientStatsAccumulator` in Phase 3, which handles the calibration
loop.  Pass the same `FrameCharacterizer` instance for every frame so the
internal reference WCS and reference stellar fluxes persist across calls:

```python
from sufficient_statistics import SufficientStatsAccumulator

acc = SufficientStatsAccumulator(frame_shape=(3008, 3008))
for i, path in enumerate(light_paths):
    acc.add_frame(path, model, fc, exposure_s=300., is_reference=(i==0))

stats = acc.finalize()
```

---

## Degradation behaviour

| Failure | Behaviour |
|---|---|
| Plate solve fails | `shift = None`; stacker treats as (0, 0) |
| < `min_stars_for_psf` stars | Use PSF from previous frame; Gaussian if no prior |
| < 2 valid star fluxes | `transparency = 1.0` |
| Sky fit fails | Return global frame median |
| Moffat fit fails | Fall back to second-moment FWHM estimate |

Every failure is logged at WARNING level.
