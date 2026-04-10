#!/usr/bin/env bash

set -e  # exit if a command fails

#######################################
# Default values
#######################################

N_REPS=5
RERUN=false
LOG_FILE="benchmark_functions_output.log"

PY_SCRIPT="benchmark_functions.py"
BENCHMARK_CONFIG="general_config/benchmark_config.yaml"
ACQ_MODE_CONFIG="general_config/acquisition_mode_settings.yaml"
TGT_FUNC_CONFIG="general_config/target_function_config.yaml"
ML_MODEL_CONFIG="mlmodel_config/gpr_config.yaml"

#######################################
# Help function
#######################################

usage() {
    echo "Usage: ./run_benchmark_funcs.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --reps N           Number of repetitions (default: 5)"
    echo "  --rerun            Force rerun of experiments"
    echo "  --log FILE         Log file name"
    echo "  -h, --help         Show this help message"
    echo ""
}

#######################################
# Argument parsing
#######################################

while [[ $# -gt 0 ]]; do
    case $1 in

        --reps)
            N_REPS="$2"
            shift 2
            ;;

        --rerun)
            RERUN=true
            shift
            ;;

        --log)
            LOG_FILE="$2"
            shift 2
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;

    esac
done

#######################################
# Prepare log file
#######################################

# if [[ -f "$LOG_FILE" ]]; then
#     echo "Log file $LOG_FILE already exists. Overwriting."
#     > "$LOG_FILE"
# fi

#######################################
# Display configuration
#######################################

echo "Running benchmark with:"
echo "  repetitions: $N_REPS"
echo "  rerun:       $RERUN"
echo "  log file:    $LOG_FILE"
echo ""

#######################################
# Build optional flags
#######################################

PY_FLAGS=""

if [[ "$RERUN" = true ]]; then
    PY_FLAGS="$PY_FLAGS --rerun"
fi

#######################################
# Run experiment
#######################################

# echo the full command for transparency
echo "Executing command:"
echo "python $PY_SCRIPT --benchmark_config $BENCHMARK_CONFIG --model_config $ML_MODEL_CONFIG --acquisition_mode_settings $ACQ_MODE_CONFIG --target_function_config $TGT_FUNC_CONFIG --repetitions $N_REPS $PY_FLAGS > $LOG_FILE"
echo ""

python $PY_SCRIPT \
    --benchmark_config $BENCHMARK_CONFIG \
    --model_config $ML_MODEL_CONFIG \
    --acquisition_mode_settings $ACQ_MODE_CONFIG \
    --target_function_config $TGT_FUNC_CONFIG \
    --repetitions $N_REPS \
    $PY_FLAGS \
    > "$LOG_FILE"
