data:
  X_pool: point_space_2d_scaled.npy
  out_dir: test_2D_synthetic
  overwrite: true
  y_pool: proptest2d_2.npy
experiment:
  acquisition_parameters:
    acquisition_mode: expected_improvement
  init_sampling_mode:
  - 1292
  - 6480
  - 0
  - 6546
  n_batch: 4
  n_cycles: 10
  percentile: 90
  sampling_mode: voronoi
ml:
  kernel_recipe:
    length_scale: 1.0
  ml_model: GPR
  ml_model_param:
    n_restarts_optimizer: 10
