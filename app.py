from bayesian_astro_stacker import BayesianAstroStacker, PipelineConfig

cfg = PipelineConfig(
    aperture_mm     = 100.0,
    focal_length_mm = 553.0,
    pixel_size_um   = 3.76,
    scale_factor    = 2,
    map_mode        = "fast",   # "fast" (default) or "exact"
    map_n_iter      = 300,
    map_alpha_tv    = 5e-3,
)

path = "C:/Users/lukas/Downloads/stacker_test/NGC6888-sv220/"
calibration_dir = f"{path}calibration/"
stacker = BayesianAstroStacker(cfg)
result  = stacker.run(
    light_dir  = f"{path}lights/",
    flat_dir   = f"{path}flats/",
    output_dir = f"{path}results/",
    bias_dir   = f"{calibration_dir}bias/",
    dark_dir   = f"{calibration_dir}darks/",
)
print(result.summary())