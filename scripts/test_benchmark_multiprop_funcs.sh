#!/usr/bin/env bash

python scripts/benchmark_functions.py \
  -bc scripts/general_config/benchmark_config_multiprop.yaml \
  -mc scripts/mlmodel_config/gpr_config.yaml \
  -acqmodes scripts/general_config/acquisition_mode_settings_multiprop.yaml \
  -tfc scripts/general_config/target_function_config_multiprop.yaml \
  --rerun