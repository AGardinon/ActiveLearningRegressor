#!/bin/bash
# usage: PY=/path/to/python ./run.sh <variant> [repetitions] [seed] [sampling_method]
#   variant:         norefine | history | window30 | window10 | history_bestfirst | window30_bestfirst (configs/bc_<variant>.yaml)
#   repetitions:     default 5
#   seed:            default 13 (overrides SEED; sets the starting pool)
#   sampling_method: default fps (overrides batch_selection.method_params.sampling_method)
# Output: runs/<sampling_method>_s<seed>/<variant>/ and runs/<sampling_method>_s<seed>/<variant>.log
S=$(dirname "$(readlink -f "$0")"); C=$S/configs
V=$1; R=${2:-5}; SEED=${3:-13}; SM=${4:-fps}
[ -f "$C/bc_$V.yaml" ] || { echo "usage: $0 <norefine|history|window30|window10|history_bestfirst|window30_bestfirst> [reps] [seed] [fps|voronoi]"; exit 1; }
PY=${PY:-python}
REPO=$(git -C "$S" rev-parse --show-toplevel)
OUT=$S/runs/${SM}_s${SEED}; mkdir -p "$OUT"
BC=$OUT/bc_$V.yaml
sed -e "s|^experiment_name:.*|experiment_name: $OUT/$V|" \
    -e "s|^SEED:.*|SEED: $SEED|" \
    -e "s|^    sampling_method:.*|    sampling_method: $SM|" "$C/bc_$V.yaml" > "$BC"
grep -q "^    sampling_method: $SM" "$BC" || { echo "sampling_method not set in $BC"; exit 1; }
cd "$REPO" && "$PY" scripts/benchmark_functions.py -bc "$BC" -mc "$C/gpr.yaml" \
    -acqmodes "$C/acq.yaml" -tfc "$C/tfc.yaml" -r "$R" --rerun > "$OUT/$V.log" 2>&1 \
  || echo "FAILED $V seed $SEED $SM (see $OUT/$V.log)"
