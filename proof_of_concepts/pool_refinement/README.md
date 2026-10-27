# Proof of concept: adaptive pool refinement

Does adaptive refinement (`activereg/adaptiveRefinement.py`) work, and does it let a campaign go
beyond the best point of a sparse starting pool? This folder holds the checks and experiments
used to debug and validate it (branch `POC/pool_refinement`, October 2026).

| Folder | What it is |
|---|---|
| `regression_check/` | Before/after harness for refinement code changes: 3 small fixed-seed campaigns, compared file by file |
| `phase1_6d/` | 6D Ackley, the paper setup (50k pool, 40 cycles × 5 points): centroid windows, centroid rules, fps vs voronoi batch selection |
| `phase2_2d/` | 2D Styblinski-Tang from a 5-point pool: how far refinement can travel, and whether best first gets trapped in a decoy basin |

Each phase folder has a README (setup, how to run) and a `build_notebook.py` that writes
`analysis.ipynb`; execute it after the runs. The notebooks (`*.ipynb`, repo-wide) and the `runs/`
folders are git-ignored; every run is reproducible from its seed.

## Fixes made along the way

| # | Problem | Fix |
|---|---|---|
| F13 | Centroid selection drew from the global numpy RNG, coupling it to the rest of the campaign | `AdaptiveRefinement` owns a generator seeded from the `DatasetGenerator` |
| F15 | `fps` treated `start_idx=0` as "no start index" | `if start_idx is None` |
| F16 | `fps` early stop dropped the point just selected, and returned a tuple regardless of `return_distD` | keep `[:i + 1]`, normal return |
| F5 | Cube size used `half_side * scale_`, which is only right for StandardScaler | `scaled_to_original_length()`: StandardScaler and MinMaxScaler, error otherwise |
| F6 | `max_half_side` comment assumed [0, 1] scaling | comment and docstring explain the scaled units |
| F12 | `plot_visual_benchmark_experiment.py` failed on current run folders | reads `config/benchmark_config.yaml` (falls back to `config.yaml`); filters on `set` only when present |
| F17 | Batch selection crashed with fewer than 50 candidates and 2 or more points per mode | `min_points` capped at the number of candidates |
| F18 | Custom functions with bounds in YAML failed (lists, not arrays) | bounds converted with `np.asarray` |

Also added: a per-cube log (`refinement_cubes_rep*.csv`) and the `best_first` centroid rule
(`refinement_centroid_selection: best_first`). Behaviour-preserving changes were checked to be
bit-identical with `regression_check/`.

## Results

**Phase 1, 6D Ackley** (3 seeds × 5 repetitions per variant). Repetitions ending above the best
point of the starting pool (out of 15):

| Centroid rule, window | fps batch selection | voronoi batch selection |
|---|---|---|
| no refinement | 0 | 0 |
| maximin, whole history | 8 | 4 |
| maximin, last 30 points | 12 | 5 |
| maximin, last 10 points | 7 | 8 |
| best first, whole history | **15** | – |
| best first, last 30 points | **15** | – |

- **Without refinement** the campaign always stops at the best point of the starting pool.
- **Refinement never made the final result worse.**
- **Best first ends much closer to the optimum:** mean best y -3.0 to -3.7, against -6.8 to -8.7
  for maximin (optimum 0).
  - 53–75% of its centroids lie in the central basin, against 1–6% for maximin.
  - Maximin picked mostly exploration points as centroids.
- **fps batch selection** (argmax of the acquisition, then greedy maximin) is better than voronoi
  or equal to it.

**Phase 2, 2D Styblinski-Tang, 5-point pool** (5 seeds × 5 repetitions). Repetitions reaching the
global basin:

| Starting pool | maximin | best first | maximin, constant cubes | best first, constant cubes |
|---|---|---|---|---|
| a pool point near the global optimum (2 seeds) | 7/10 | **10/10** | 9/10 | **10/10** |
| no pool point near it (3 seeds) | 0/15 | 0/15 | 2/15 | 0/15 |

- **Refinement is a local search.** New candidates only appear around acquired points.
- **The cubes collapse:** with the `density` half-side they shrink about fivefold per step after
  the first, from about 1.0 down to 0.01.
- **Best first converges fastest when the global basin is reachable,** and stays in the basin of
  the best point when it is not. Maximin rarely escapes either.

## Conclusions

1. The tool works and does what it was designed for. In high dimensions, refinement is what takes
   a campaign past the limit of its starting pool.
2. Default kept: `maximin`, so the paper runs stay reproducible.
3. `best_first` is recommended when the starting pool covers the space, as a large LHS pool does.
   It supports a single target property only.
4. Refinement does not explore. Basins the starting pool does not touch stay out of reach.
5. Use `fps` for batch selection (the code default).

## Open items (not on this branch)

- **Lab campaigns:** refining at will is not possible in a lab.
  - Variables have an instrument resolution (a step) or a fixed list of values.
  - The proposed design: typed variables, refinement on the allowed grid, a floor of one
    instrument step, and saturation when a cube is fully resolved. It is described in the
    (local) design notes, §4 and §8.
  - Phase 2 shows why the floor matters: unconstrained cubes shrink far below any instrument
    step.
- **A global component:** a few samples from the whole domain at each refinement step, to reach
  basins the pool does not touch.
- **Cube-size schedule:** the `density` rule already shrinks the cubes, maybe too fast in sparse
  pools. A floor or a TuRBO-style expand/shrink rule are the alternatives.
- **F7:** the `density` cap compares a local mean k-NN distance with a global median 1-NN distance.
  Minor, with no effect seen in these runs.
- **`best_first` for multiple targets** needs a scalar score per point, e.g. via `scalarize()`.
