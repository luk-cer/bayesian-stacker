"""
bayesian_astro_stacker.py
=========================
Combined Bayesian astrophotography stacking pipeline — Phases 0–4.

This module wires together all pipeline phases into a single, easy-to-use
entry point.  Each phase can also be run independently via its own module.

Pipeline overview
-----------------

  Phase 0a  InstrumentModel calibration
            ├── fit_bias(bias_frames)
            ├── fit_dark(dark_frames)
            └── fit_flat(flat_frames)

  Phase 0b  BayesCalibrationState
            ├── Conjugate prior accumulators for all sensor parameters
            └── Cross-session posterior carry-over (with Arrhenius correction)

  Phase 0c  DarkMixtureModel
            └── EM Gamma mixture for hot/cold/normal pixel classification

  Phase 1   Optics + WCS
            ├── ScopeGeometry → diffraction PSF
            └── ASTAPSolver / extract_wcs_geometry → per-frame shift

  Phase 2   FrameCharacterizer (per-frame)
            ├── Sky background estimation
            ├── Star extraction + PSF estimation
            ├── Seeing FWHM (Moffat fit)
            ├── Transparency (aperture photometry)
            └── Sub-pixel shift (WCS or phase correlation)

  Phase 3   SufficientStatsAccumulator (streaming, limited RAM)
            ├── Calibrate + characterise each frame
            ├── Accumulate weighted_sum, weight_sum, sky_sum, sq_sum
            └── Persist to HDF5 for crash recovery

  Phase 4   MAP super-resolution stacker (GPU, PyTorch)
            ├── Solve λ_hr on S×H × S×W grid
            ├── Regularised Poisson NLL (TV + KL + wavelet)
            └── Output: super-resolved FITS + convergence plot

Usage
-----

Full pipeline from raw FITS files:

    from bayesian_astro_stacker import BayesianAstroStacker, PipelineConfig

    cfg = PipelineConfig(
        scale_factor   = 2,
        map_n_iter     = 200,
        map_alpha_tv   = 1e-2,
        map_mode       = 'fast',
    )
    stacker = BayesianAstroStacker(cfg)
    result  = stacker.run(
        bias_dir  = "calibration/bias/",
        dark_dir  = "calibration/dark/",
        flat_dir  = "calibration/flat/",
        light_dir = "lights/",
        output_dir = "results/",
    )

Resume an interrupted run:

    stacker = BayesianAstroStacker.resume(
        checkpoint = "results/stats_checkpoint.h5",
        model_path = "results/instrument_model.h5",
        light_dir  = "lights/",
        output_dir = "results/",
    )

Dependencies
------------
    numpy astropy scipy h5py
    torch >= 2.0   (for Phase 4 GPU optimisation)
    matplotlib     (optional — convergence plots)
    astap          (optional — plate solving for exact shifts)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── Phase 0a / 0b
from instrument_model_artifact import InstrumentModel
from bayes_calibration import (
    BayesCalibrationState,
    SensorPriors,
)

# ── Phase 0c
from dark_mixture import DarkMixtureModel

# ── Phase 1
from optics import ScopeGeometry

# ── Phase 2
from frame_characterizer import FrameCharacterizer

# ── Phase 3
from sufficient_statistics import SufficientStatsAccumulator, SufficientStats

# ── Phase 4
from map_stacker import MapConfig, MapResult, solve, _TORCH_OK

logger = logging.getLogger(__name__)


# ============================================================================
# PipelineConfig
# ============================================================================

@dataclass
class PipelineConfig:
    """
    Top-level configuration for the combined pipeline.

    Instrument
    ----------
    aperture_mm : float
        Telescope aperture in millimetres.  Default: 100 (Esprit 100ED).
    focal_length_mm : float
        Telescope focal length in millimetres.  Default: 550.
    pixel_size_um : float
        Camera pixel size in micrometres.  Default: 3.76 (ZWO ASI533).
    gain_setting : float
        Sensor gain setting (ADU/e-).  Default: 100.
    sensor_temp_c : float
        Sensor temperature during acquisition, °C.  Default: -10.

    Calibration
    -----------
    use_bayes_calibration : bool
        If True, use Bayesian conjugate-prior accumulators for calibration
        frame fitting.  Allows cross-session posterior carry-over.
        Default: True.
    bayes_state_path : str | None
        Path to an existing BayesCalibrationState HDF5 file to resume from.
        If None, a fresh state is built from default priors.
    new_session : bool
        If True (default), reset flat-gain mean to 1.0 when loading a prior
        state (correct behaviour: dust moves between sessions).

    Frame selection
    ---------------
    exposure_s : float
        Light frame exposure time in seconds.  If None, read from FITS header.
    min_transparency : float
        Reject frames with transparency below this threshold.  Default: 0.3.
    max_fwhm_arcsec : float
        Reject frames with seeing FWHM above this threshold.  Default: 5.0.
    checkpoint_every : int
        Save a stats checkpoint every N frames.  Default: 20.

    Super-resolution (Phase 4)
    --------------------------
    scale_factor : int
        Super-resolution upscaling factor.  Default: 2.
    map_mode : str
        'fast' (default) or 'exact'.
    map_n_iter : int
        Maximum MAP optimisation iterations.  Default: 300.
    map_alpha_tv : float
        Total variation regularisation weight.  Default: 1e-2.
    map_alpha_kl : float
        KL-from-prior weight.  Default: 0.
    map_alpha_wav : float
        Haar wavelet L1 weight.  Default: 0.
    map_batch_size : int
        Mini-batch size (exact mode only).  Default: 8.
    map_device : str | None
        PyTorch device.  None = auto-detect (CUDA > MPS > CPU).

    Output
    ------
    save_fast_stack : bool
        Also write a fast (Gamma-Poisson mean) stack FITS.  Default: True.
    save_convergence_plot : bool
        Write convergence loss-curve PNG.  Default: True.
    overwrite : bool
        Overwrite existing output files.  Default: True.
    """
    # Instrument
    aperture_mm:      float = 100.0
    focal_length_mm:  float = 550.0
    pixel_size_um:    float = 3.76
    gain_setting:     float = 100.0
    sensor_temp_c:    float = -10.0

    # Calibration
    use_bayes_calibration: bool         = True
    bayes_state_path:      Optional[str] = None
    new_session:           bool         = True

    # Frame selection
    exposure_s:         Optional[float] = None
    min_transparency:   float           = 0.3
    max_fwhm_arcsec:    float           = 5.0
    checkpoint_every:   int             = 20

    # Phase 4
    scale_factor:   int            = 2
    map_mode:       str            = 'fast'
    map_n_iter:     int            = 300
    map_alpha_tv:   float          = 1e-2
    map_alpha_kl:   float          = 0.0
    map_alpha_wav:  float          = 0.0
    map_batch_size: int            = 8
    map_device:     Optional[str]  = None

    # Output
    save_fast_stack:       bool = True
    save_convergence_plot: bool = True
    overwrite:             bool = True


# ============================================================================
# PipelineResult
# ============================================================================

@dataclass
class PipelineResult:
    """
    Final result of the combined pipeline.

    Attributes
    ----------
    map_result : MapResult | None
        Phase 4 MAP super-resolution result.  None if torch unavailable.
    stats : SufficientStats
        Phase 3 sufficient statistics.
    model : InstrumentModel
        Fitted instrument calibration model.
    output_dir : Path
        Directory where all output files were written.
    n_frames_total : int
        Total frames found in light directory.
    n_frames_used : int
        Frames that passed quality filtering.
    elapsed_s : float
        Total wall-clock time.

    Output files
    ------------
    instrument_model.h5         Phase 0a calibration model
    bayes_state.h5              Phase 0b Bayesian calibration state
    sufficient_stats.h5         Phase 3 accumulated statistics
    fast_stack.fits             Phase 3 Gamma-Poisson mean stack
    lambda_hr.fits              Phase 4 super-resolved scene
    convergence.png             Phase 4 loss curve
    """
    map_result:     Optional[MapResult]
    stats:          SufficientStats
    model:          InstrumentModel
    output_dir:     Path
    n_frames_total: int
    n_frames_used:  int
    elapsed_s:      float

    def summary(self) -> str:
        lines = [
            "BayesianAstroStacker — Pipeline Result",
            "=" * 56,
            f"  Output dir      : {self.output_dir}",
            f"  Frames          : {self.n_frames_used}/{self.n_frames_total} used",
            f"  Elapsed         : {self.elapsed_s:.1f} s",
            "",
            "  Phase 3 statistics:",
            f"    Frame shape   : {self.stats.frame_shape}",
            f"    Median FWHM   : {self.stats.mean_fwhm_arcsec:.2f}\"",
            f"    Median transp : {self.stats.mean_transparency:.3f}",
            f"    PSF kernels   : {len(self.stats.psf_list)} stored",
        ]
        if self.map_result is not None:
            r = self.map_result
            lines += [
                "",
                "  Phase 4 MAP result:",
                f"    HR shape      : {r.lambda_hr.shape}",
                f"    Scale factor  : {r.config.scale_factor}x",
                f"    Iterations    : {r.n_iter} / {r.config.n_iter}",
                f"    Converged     : {r.converged}",
                f"    Final loss    : {r.loss_history[-1]:.4e}"
                              if r.loss_history else "    Final loss    : n/a",
                f"    Device        : {r.device}",
                f"    lambda range  : [{r.lambda_hr.min():.1f},"
                                    f" {r.lambda_hr.max():.1f}] ADU",
            ]
        else:
            lines.append("\n  Phase 4: skipped (PyTorch not available)")
        lines += ["=" * 56]
        return "\n".join(lines)

    def outputs(self) -> dict:
        """Return dict of {label: Path} for all output files."""
        d = self.output_dir
        out = {
            "instrument_model" : d / "instrument_model.h5",
            "sufficient_stats" : d / "sufficient_stats.h5",
        }
        if (d / "bayes_state.h5").exists():
            out["bayes_state"] = d / "bayes_state.h5"
        if (d / "fast_stack.fits").exists():
            out["fast_stack"] = d / "fast_stack.fits"
        if (d / "lambda_hr.fits").exists():
            out["lambda_hr"] = d / "lambda_hr.fits"
        if (d / "convergence.png").exists():
            out["convergence_plot"] = d / "convergence.png"
        return out


# ============================================================================
# BayesianAstroStacker
# ============================================================================

class BayesianAstroStacker:
    """
    Orchestrates the full Bayesian astrophotography stacking pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline hyperparameters (see PipelineConfig docstring).
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------ #
    # Public: run full pipeline                                            #
    # ------------------------------------------------------------------ #

    def run(
        self,
        light_dir:  str | Path,
        output_dir: str | Path,
        bias_dir:   Optional[str | Path] = None,
        dark_dir:   Optional[str | Path] = None,
        flat_dir:   Optional[str | Path] = None,
        dflat_dir:  Optional[str | Path] = None,
        model_path: Optional[str | Path] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline end-to-end.

        Parameters
        ----------
        light_dir   : directory containing calibrated or raw light FITS frames
        output_dir  : directory for all output files (created if absent)
        bias_dir    : calibration bias frames  (None → skip bias)
        dark_dir    : calibration dark frames  (None → skip dark)
        flat_dir    : calibration flat frames  (None → skip flat)
        dflat_dir   : dark flats               (None → skip)
        model_path  : pre-built InstrumentModel HDF5 (skips Phase 0)

        Returns
        -------
        PipelineResult
        """
        cfg = self.config
        t0  = time.time()

        light_dir  = Path(light_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Phase 0: calibration ─────────────────────────────────────────
        model = self._phase0_calibration(
            bias_dir, dark_dir, flat_dir, dflat_dir, model_path, output_dir
        )

        # ── Phase 1: optics ──────────────────────────────────────────────
        scope, fc = self._phase1_optics()

        # ── Phase 2+3: characterise + accumulate ─────────────────────────
        light_paths = self._collect_lights(light_dir)
        n_total     = len(light_paths)
        logger.info("Found %d light frames in %s", n_total, light_dir)

        stats, n_used = self._phase2_3_accumulate(
            light_paths, model, scope, fc, output_dir
        )

        # ── Phase 4: MAP stacker ─────────────────────────────────────────
        map_result = self._phase4_map(stats, light_paths, output_dir)

        # ── Write outputs ────────────────────────────────────────────────
        self._write_outputs(stats, map_result, model, output_dir)

        elapsed = time.time() - t0
        result  = PipelineResult(
            map_result     = map_result,
            stats          = stats,
            model          = model,
            output_dir     = output_dir,
            n_frames_total = n_total,
            n_frames_used  = n_used,
            elapsed_s      = elapsed,
        )
        logger.info(result.summary())
        return result

    # ------------------------------------------------------------------ #
    # Public: resume from checkpoint                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def resume(
        cls,
        checkpoint:  str | Path,
        model_path:  str | Path,
        light_dir:   str | Path,
        output_dir:  str | Path,
        config:      Optional[PipelineConfig] = None,
    ) -> PipelineResult:
        """
        Resume an interrupted pipeline run from a Phase 3 checkpoint.

        Parameters
        ----------
        checkpoint  : path to stats_checkpoint.h5 (Phase 3 HDF5)
        model_path  : path to instrument_model.h5  (Phase 0 HDF5)
        light_dir   : directory of light FITS frames
        output_dir  : output directory
        config      : PipelineConfig (uses defaults if None)
        """
        stacker    = cls(config)
        cfg        = stacker.config
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        t0         = time.time()

        logger.info("Resuming from checkpoint: %s", checkpoint)

        # Load existing model + state
        model = InstrumentModel.load(model_path)
        scope, fc = stacker._phase1_optics()

        # Resume accumulation
        light_paths = stacker._collect_lights(light_dir)
        acc = SufficientStatsAccumulator.resume(checkpoint)
        n_done = acc.frame_count
        logger.info("Checkpoint has %d frames; %d remaining",
                    n_done, len(light_paths) - n_done)

        remaining = light_paths[n_done:]
        for i, path in enumerate(remaining):
            exp_s = cfg.exposure_s or _read_exptime(path)
            try:
                acc.add_frame(path, model, fc, exposure_s=exp_s,
                              is_reference=(i == 0 and n_done == 0))
            except Exception as exc:
                logger.warning("Frame %s failed: %s — skipping", path.name, exc)
                continue
            if (n_done + i + 1) % cfg.checkpoint_every == 0:
                acc.save(output_dir / "stats_checkpoint.h5")

        stats     = acc.finalize()
        n_used    = stats.frame_count
        map_result = stacker._phase4_map(stats, light_paths, output_dir)
        stacker._write_outputs(stats, map_result, model, output_dir)

        result = PipelineResult(
            map_result     = map_result,
            stats          = stats,
            model          = model,
            output_dir     = output_dir,
            n_frames_total = len(light_paths),
            n_frames_used  = n_used,
            elapsed_s      = time.time() - t0,
        )
        logger.info(result.summary())
        return result

    # ------------------------------------------------------------------ #
    # Internal: Phase 0 — calibration                                     #
    # ------------------------------------------------------------------ #

    def _phase0_calibration(
        self,
        bias_dir:   Optional[Path],
        dark_dir:   Optional[Path],
        flat_dir:   Optional[Path],
        dflat_dir:  Optional[Path],
        model_path: Optional[Path],
        output_dir: Path,
    ) -> InstrumentModel:
        cfg = self.config

        # If a pre-built model is provided, load it
        if model_path is not None:
            logger.info("Loading pre-built InstrumentModel from %s", model_path)
            return InstrumentModel.load(model_path)

        # Build Bayesian calibration state
        bayes_state = None
        if cfg.use_bayes_calibration:
            bayes_state = self._load_or_create_bayes_state(output_dir)

        # Fit from calibration frames
        model = InstrumentModel()

        if bias_dir and Path(bias_dir).exists():
            logger.info("Phase 0a: fitting bias from %s", bias_dir)
            paths = _collect_fits(bias_dir)
            model.fit_bias(paths, bayes_state=bayes_state)

        if dark_dir and Path(dark_dir).exists():
            logger.info("Phase 0a: fitting dark from %s", dark_dir)
            paths = _collect_fits(dark_dir)
            model.fit_dark(paths, bayes_state=bayes_state)

        if flat_dir and Path(flat_dir).exists():
            logger.info("Phase 0a: fitting flat from %s", flat_dir)
            paths = _collect_fits(flat_dir)
            dflat_paths = _collect_fits(dflat_dir) if dflat_dir and \
                          Path(dflat_dir).exists() else None
            model.fit_flat(paths, bayes_state=bayes_state)
            if dflat_paths:
                model.fit_dark_flat(dflat_paths)

        # Fit dark mixture (Phase 0c) if dark was fitted
        if model.dark_rate is not None:
            logger.info("Phase 0c: fitting dark pixel mixture model")
            sensor_mean = float(np.median(model.dark_rate))
            dmm = DarkMixtureModel()
            dmm.fit(model.dark_rate, sensor_mean_rate=sensor_mean)
            # Attach mixture weights to model metadata
            model.metadata["dark_mixture_normal_weight"] = \
                float(dmm.pixel_weight_map(model.dark_rate, sensor_mean).mean())

        # Save
        out_path = output_dir / "instrument_model.h5"
        model.save(out_path)
        logger.info("InstrumentModel saved to %s", out_path)

        if bayes_state is not None:
            bs_path = output_dir / "bayes_state.h5"
            bayes_state.save(bs_path)
            logger.info("BayesCalibrationState saved to %s", bs_path)

        return model

    def _load_or_create_bayes_state(
        self, output_dir: Path
    ) -> BayesCalibrationState:
        cfg       = self.config
        bs_path   = cfg.bayes_state_path
        new_temp  = cfg.sensor_temp_c

        if bs_path and Path(bs_path).exists():
            logger.info("Loading BayesCalibrationState from %s", bs_path)
            return BayesCalibrationState.load(
                bs_path,
                new_session  = cfg.new_session,
                new_temp_c   = new_temp,
            )
        logger.info("Creating fresh BayesCalibrationState from default priors")
        priors = SensorPriors.for_asi533_gain100()
        # Shape will be set lazily when first frame is processed
        return BayesCalibrationState.from_priors(priors, shape=None)

    # ------------------------------------------------------------------ #
    # Internal: Phase 1 — optics                                          #
    # ------------------------------------------------------------------ #

    def _phase1_optics(self):
        cfg = self.config
        scope = ScopeGeometry(
            aperture_mm     = cfg.aperture_mm,
            focal_length_mm = cfg.focal_length_mm,
            pixel_size_um   = cfg.pixel_size_um,
        )
        fc = FrameCharacterizer(scope)
        return scope, fc

    # ------------------------------------------------------------------ #
    # Internal: Phase 2+3 — characterise + accumulate                     #
    # ------------------------------------------------------------------ #

    def _phase2_3_accumulate(
        self,
        light_paths: List[Path],
        model:       InstrumentModel,
        scope:       ScopeGeometry,
        fc:          FrameCharacterizer,
        output_dir:  Path,
    ):
        cfg        = self.config
        acc        = SufficientStatsAccumulator(
            frame_shape=model.frame_shape if model.frame_shape else None
        )
        n_used     = 0
        n_rejected = 0

        for i, path in enumerate(light_paths):
            is_ref = (n_used == 0)
            exp_s  = cfg.exposure_s or _read_exptime(path)

            try:
                meta = fc.characterize(path, model, exposure_s=exp_s,
                                       is_reference=is_ref)
            except Exception as exc:
                logger.warning("Phase 2 failed for %s: %s — skipping",
                               path.name, exc)
                n_rejected += 1
                continue

            # Quality filter
            if meta.transparency < cfg.min_transparency:
                logger.debug("Reject %s: transparency=%.3f < %.3f",
                             path.name, meta.transparency, cfg.min_transparency)
                n_rejected += 1
                continue
            if (meta.fwhm_arcsec or 0.) > cfg.max_fwhm_arcsec:
                logger.debug("Reject %s: FWHM=%.2f\" > %.2f\"",
                             path.name, meta.fwhm_arcsec, cfg.max_fwhm_arcsec)
                n_rejected += 1
                continue

            acc.add_calibrated(meta.calibrated, meta)
            n_used += 1

            if n_used % cfg.checkpoint_every == 0:
                ckpt = output_dir / "stats_checkpoint.h5"
                acc.save(ckpt)
                logger.info("Checkpoint saved (%d frames) → %s", n_used, ckpt)

            logger.info(
                "Frame %3d/%d  t=%.3f  FWHM=%.2f\"  accepted=%d  rejected=%d",
                i+1, len(light_paths),
                meta.transparency, meta.fwhm_arcsec or 0.,
                n_used, n_rejected,
            )

        if n_used == 0:
            raise RuntimeError(
                "No frames passed quality filtering.  "
                "Lower min_transparency or max_fwhm_arcsec."
            )

        stats = acc.finalize()
        logger.info("Phase 3 complete: %d/%d frames  %s",
                    n_used, len(light_paths), stats.summary())
        return stats, n_used

    # ------------------------------------------------------------------ #
    # Internal: Phase 4 — MAP stacker                                     #
    # ------------------------------------------------------------------ #

    def _phase4_map(
        self,
        stats:       SufficientStats,
        light_paths: List[Path],
        output_dir:  Path,
    ) -> Optional[MapResult]:
        if not _TORCH_OK:
            logger.warning(
                "PyTorch not available — Phase 4 MAP optimisation skipped.\n"
                "Install PyTorch: pip install torch\n"
                "Phase 3 fast_stack.fits is still written as usable output."
            )
            return None

        cfg = self.config
        map_cfg = MapConfig(
            scale_factor = cfg.scale_factor,
            mode         = cfg.map_mode,
            n_iter       = cfg.map_n_iter,
            alpha_tv     = cfg.map_alpha_tv,
            alpha_kl     = cfg.map_alpha_kl,
            alpha_wav    = cfg.map_alpha_wav,
            batch_size   = cfg.map_batch_size,
            device       = cfg.map_device,
        )

        fits_paths = light_paths if cfg.map_mode == 'exact' else None
        logger.info("Phase 4: MAP stacker  mode=%s  scale=%d×  device=%s",
                    cfg.map_mode, cfg.scale_factor, cfg.map_device or 'auto')

        return solve(stats, config=map_cfg, fits_paths=fits_paths)

    # ------------------------------------------------------------------ #
    # Internal: write output files                                         #
    # ------------------------------------------------------------------ #

    def _write_outputs(
        self,
        stats:      SufficientStats,
        map_result: Optional[MapResult],
        model:      InstrumentModel,
        output_dir: Path,
    ) -> None:
        cfg = self.config

        # Phase 3 stats
        stats_path = output_dir / "sufficient_stats.h5"
        stats.save(stats_path)
        logger.info("SufficientStats saved → %s", stats_path)

        bayer = model.bayer_pattern   # None for mono, e.g. 'RGGB' for OSC

        # Fast stack (Gamma-Poisson mean)
        if cfg.save_fast_stack:
            fast_path = output_dir / "fast_stack.fits"
            stats.save_fast_stack_fits(fast_path, bayer_pattern=bayer)
            logger.info("Fast stack saved → %s", fast_path)

        # Phase 4 outputs
        if map_result is not None:
            lambda_path = output_dir / "lambda_hr.fits"
            map_result.save_fits(lambda_path, quality_map=stats.quality_map,
                                 bayer_pattern=bayer)
            logger.info("Super-resolved scene saved → %s", lambda_path)

            if cfg.save_convergence_plot:
                conv_path = output_dir / "convergence.png"
                map_result.save_convergence_plot(conv_path)

    # ------------------------------------------------------------------ #
    # Internal: helpers                                                    #
    # ------------------------------------------------------------------ #

    def _collect_lights(self, light_dir: Path) -> List[Path]:
        paths = _collect_fits(light_dir)
        if not paths:
            raise FileNotFoundError(
                f"No FITS files found in {light_dir}\n"
                "Check the path and ensure files have .fits / .fit / .fts extension."
            )
        paths.sort()
        return paths


# ============================================================================
# Standalone helpers
# ============================================================================

def _collect_fits(directory: str | Path) -> List[Path]:
    """Return sorted list of all FITS files in a directory."""
    d = Path(directory)
    paths: List[Path] = []
    for ext in ("*.fits", "*.fit", "*.fts", "*.FITS", "*.FIT", "*.FTS"):
        paths.extend(d.glob(ext))
    return sorted(set(paths))


def _read_exptime(path: Path) -> float:
    """Read EXPTIME from a FITS header; default 300 s if absent."""
    try:
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdul:
            return float(hdul[0].header.get("EXPTIME", 300.0))
    except Exception:
        return 300.0


# ============================================================================
# CLI entry point
# ============================================================================

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Bayesian Astro Stacker — full pipeline (Phases 0–4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("light_dir",   help="Directory of light FITS frames")
    parser.add_argument("output_dir",  help="Output directory")
    parser.add_argument("--bias-dir",  default=None)
    parser.add_argument("--dark-dir",  default=None)
    parser.add_argument("--flat-dir",  default=None)
    parser.add_argument("--model",     default=None,
                        help="Pre-built instrument model .h5 (skips Phase 0)")
    parser.add_argument("--scale",     type=int,   default=2,
                        help="Super-resolution scale factor")
    parser.add_argument("--mode",      choices=["fast","exact"], default="fast",
                        help="MAP stacker mode")
    parser.add_argument("--n-iter",    type=int,   default=300)
    parser.add_argument("--alpha-tv",  type=float, default=1e-2)
    parser.add_argument("--alpha-kl",  type=float, default=0.0)
    parser.add_argument("--alpha-wav", type=float, default=0.0)
    parser.add_argument("--device",    default=None,
                        help="PyTorch device: cuda / mps / cpu")
    parser.add_argument("--exposure",  type=float, default=None,
                        help="Exposure time in seconds (reads FITS header if omitted)")
    parser.add_argument("--min-transparency", type=float, default=0.3)
    parser.add_argument("--max-fwhm",  type=float, default=5.0)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level   = logging.DEBUG if args.verbose else logging.INFO,
        format  = "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    cfg = PipelineConfig(
        scale_factor       = args.scale,
        map_mode           = args.mode,
        map_n_iter         = args.n_iter,
        map_alpha_tv       = args.alpha_tv,
        map_alpha_kl       = args.alpha_kl,
        map_alpha_wav      = args.alpha_wav,
        map_device         = args.device,
        exposure_s         = args.exposure,
        min_transparency   = args.min_transparency,
        max_fwhm_arcsec    = args.max_fwhm,
        checkpoint_every   = args.checkpoint_every,
    )

    stacker = BayesianAstroStacker(cfg)
    result  = stacker.run(
        light_dir  = args.light_dir,
        output_dir = args.output_dir,
        bias_dir   = args.bias_dir,
        dark_dir   = args.dark_dir,
        flat_dir   = args.flat_dir,
        model_path = args.model,
    )

    print(result.summary())
    print("\nOutputs written:")
    for label, path in result.outputs().items():
        print(f"  {label:<22} {path}")


if __name__ == "__main__":
    _cli()
