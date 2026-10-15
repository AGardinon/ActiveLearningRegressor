# Regression check for adaptive refinement

Before/after check for changes to `activereg/adaptiveRefinement.py` and the
refinement code in `scripts/benchmark_functions.py`. It runs three small
benchmark campaigns with a fixed seed, so two runs of the same code give
identical CSVs (verified).

| Config | Function | What it exercises |
|---|---|---|
| A | Hartmann6D, 5k LHS pool, 20 cycles, N_r 250 | main path; refined points get acquired (~17 per rep) |
| B | Hartmann3D, `init_batch_size` 15 > `refinement_step` 10 | catch-up trigger (fires after only 2 new points) |
| C | BraninCurrin, ParEGO | multi-property `y` |

All use a fast GPR (`configs/gpr_fast.yaml`, no grid search) and 2 repetitions.

## Usage

```bash
# PY must be an environment with activereg (editable) and botorch
PY=/path/to/env/bin/python ./run.sh before   # on the old code
PY=/path/to/env/bin/python ./run.sh after    # on the new code
for X in A B C; do echo "== $X"; python compare.py runs/before_$X runs/after_$X; done
```

- **Pure refactor:** every file must be `identical`.
- **Intended behaviour change:** list in advance which files/configs should
  change and why, then check that only those do.

`runs/` is git-ignored.
