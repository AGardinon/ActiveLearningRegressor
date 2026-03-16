# Bash script to run the active learning regression experiment with specified configuration files

# Variables
PY_SCRIPT="benchmark_functions.py"
BENCHMARK_CONFIG="general_config/benchmark_config.yaml"
ACQ_MODE_CONFIG="general_config/acquisition_mode_settings.yaml"
TGT_FUNC_CONFIG="general_config/target_function_config.yaml"
ML_MODEL_CONFIG="mlmodel_config/gpr_config.yaml"

# Set number of repetions for the experiment as an input argument
if [ -z "$1" ]; then
    echo "No number of repetitions provided. Using default value of 3."
    N_REPS=3
else
    N_REPS=$1
fi

# Optional rerun flag
RERUN_FLAG=""
if [ "$2" = "--rerun" ]; then
    RERUN_FLAG="--rerun"
fi

# Output run file
LOG_FILE="benchmark_func_output.log"
# if [ -f "$LOG_FILE" ]; then
#     echo "Log file $LOG_FILE already exists. It will be overwritten."
#     > $LOG_FILE
# fi

echo "Experiment started with the following configs:
    - $PY_SCRIPT -> Python script to run the benchmark experiment
    - $BENCHMARK_CONFIG -> general benchmark settings 
    - $ML_MODEL_CONFIG -> ML model settings
    - $ACQ_MODE_CONFIG -> acquisition mode settings
    - $TGT_FUNC_CONFIG -> target function settings
    - Number of repetitions: $N_REPS
    - Rerun flag: $RERUN_FLAG
"

# Run
python $PY_SCRIPT \
    --benchmark_config $BENCHMARK_CONFIG \
    --model_config $ML_MODEL_CONFIG \
    --acquisition_mode_settings $ACQ_MODE_CONFIG \
    --target_function_config $TGT_FUNC_CONFIG \
    --repetitions $N_REPS \
    $RERUN_FLAG \
    > $LOG_FILE