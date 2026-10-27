#!/bin/bash
# usage: PY=/path/to/python ./run.sh <variant> [repetitions] [seed]
#   variant:     maximin | bestfirst | maximin_fixed | bestfirst_fixed (configs/bc_<variant>.yaml)
#   repetitions: default 5
#   seed:        default 13 (overrides SEED; sets the 5-point starting pool)
# Output: runs/s<seed>/<variant>/ and runs/s<seed>/<variant>.log
S=$(dirname "$(readlink -f "$0")"); C=$S/configs
V=$1; R=${2:-5}; SEED=${3:-13}
[ -f "$C/bc_$V.yaml" ] || { echo "usage: $0 <maximin|bestfirst|maximin_fixed|bestfirst_fixed> [reps] [seed]"; exit 1; }
PY=${PY:-python}
REPO=$(git -C "$S" rev-parse --show-toplevel)
OUT=$S/runs/s${SEED}; mkdir -p "$OUT"
BC=$OUT/bc_$V.yaml
sed -e "s|^experiment_name:.*|experiment_name: $OUT/$V|" \
    -e "s|^SEED:.*|SEED: $SEED|" "$C/bc_$V.yaml" > "$BC"
cd "$REPO" && "$PY" scripts/benchmark_functions.py -bc "$BC" -mc "$C/gpr.yaml" \
    -acqmodes "$C/acq.yaml" -tfc "$C/tfc.yaml" -r "$R" --rerun > "$OUT/$V.log" 2>&1 \
  || echo "FAILED $V seed $SEED (see $OUT/$V.log)"
