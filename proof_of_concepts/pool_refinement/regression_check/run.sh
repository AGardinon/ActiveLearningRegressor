#!/bin/bash
# usage: PY=/path/to/python ./run.sh <label>
# Runs configs A, B, C of scripts/benchmark_functions.py into runs/<label>_{A,B,C}
# (logs in runs/<label>_{A,B,C}.log). PY defaults to `python`; it needs botorch.
S=$(dirname "$(readlink -f "$0")"); C=$S/configs; L=$1
[ -z "$L" ] && { echo "usage: $0 <label>"; exit 1; }
PY=${PY:-python}
REPO=$(git -C "$S" rev-parse --show-toplevel)
mkdir -p "$S/runs"
for X in A B C; do
  case $X in
    A) TFC=$C/tfc_A.yaml;          ACQ=$C/acq_single.yaml;;
    B) TFC=$C/tfc_hartmann3d.yaml; ACQ=$C/acq_single.yaml;;
    C) TFC=$C/tfc_C.yaml;          ACQ=$C/acq_multi.yaml;;
  esac
  # point the output folder of this run to runs/<label>_X
  BC=$S/runs/bc_${L}_$X.yaml
  sed "s|^experiment_name:.*|experiment_name: $S/runs/${L}_$X|" "$C/bc_$X.yaml" > "$BC"
  (cd "$REPO" && "$PY" scripts/benchmark_functions.py -bc "$BC" -mc "$C/gpr_fast.yaml" \
      -acqmodes "$ACQ" -tfc "$TFC" -r 2 --rerun) > "$S/runs/${L}_$X.log" 2>&1 \
    || echo "FAILED $X (see runs/${L}_$X.log)"
  echo "$X refinement steps: $(grep -c 'Adaptive refinement step' "$S/runs/${L}_$X.log")"
done
