# Lab Scripts — Design Document

Branch: `refactor/simulation-scripts`

---

## Motivation

The benchmark pipeline (`benchmark_functions.py`) is the canonical, fully-automated path.
Two additional use cases are not covered:

1. **Real-lab single-cycle**: one AL step at a time, with pauses for physical measurement.
   The user runs the script, takes the output CSV to the lab, fills in measured values,
   and re-runs for the next cycle.

2. **In-silico lab simulation**: automated multi-cycle run using a CSV ground truth as
   an oracle — functionally like the benchmark, but for arbitrary CSV datasets rather
   than synthetic functions.

Both cases share the same inner logic; what differs is the outer loop and state management.

---

## Core abstraction: `run_single_al_cycle()`

A new function added to `activereg/experiment.py`. It represents one complete AL step
with no file I/O — purely in-memory.

### Signature

```python
def run_single_al_cycle(
    pool_df: pd.DataFrame,                          # full search space, features only
    evidence_df: pd.DataFrame,                      # labeled points so far (features + targets)
    scaler: BaseEstimator,                          # pre-fit on pool; caller owns and persists it
    ml_model: IndependentMultiPropertyModel,        # untrained; trained inside this call
    search_vars: list[str],
    target_vars: list[str],
    acquisition_params: list[dict],
    batch_selection_method: str,
    batch_selection_params: dict,
    penalization_params: tuple[float, float] | None = (0.25, 1.0),
    rng: np.random.Generator | None = None,
) -> dict
```

### Return dict keys

| Key | Type | Description |
|---|---|---|
| `next_batch_df` | `pd.DataFrame` | Selected candidates (features + NaN targets) |
| `trained_model` | `IndependentMultiPropertyModel` | Model after training on evidence |
| `y_pred` | `np.ndarray` | Mean predictions over full pool |
| `y_unc` | `np.ndarray` | Uncertainty over full pool |
| `landscapes` | `np.ndarray` | Acquisition landscapes, shape `(n_entries, M_candidates)` |
| `y_best` | `dict[str, float]` | Per-property best observed value |
| `sampled_idx` | `list[int]` | Indices into candidates selected this cycle |
| `candidates_df` | `pd.DataFrame` | Pool minus evidence at this cycle |
| `meta` | `list[dict \| None]` | Per-entry metadata from `sampling_block` |

### Internal flow

1. `prepare_training_data()` → derive `X_train`, `Y_train`, `X_candidates`, `candidates_df`
2. `ml_model.train(X_train, Y_train)`
3. `ml_model.predict(X_pool)` → `y_pred`, `y_unc` (for logging/visualization)
4. `sampling_block(...)` → `sampled_idx`, `landscapes`, `meta`
5. Build `next_batch_df` from `candidates_df.iloc[sampled_idx]`

---

## Helper: `prepare_training_data()`

Migrated from `cycle.py` (deleted), adapted to current API. Added to `experiment.py`.

```python
def prepare_training_data(
    pool_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    search_vars: list[str],
    target_vars: list[str],
    scaler: BaseEstimator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]
```

Returns `X_train`, `Y_train` (2-D), `X_candidates`, `candidates_df`.
Uses `remove_evidence_from_gt()` for the pool-minus-evidence split.

---

## Scaler ownership

The scaler is **fit once on the full pool** at experiment initialization and persisted to
`dataset/scaler.joblib`. Every cycle reloads and reuses it. This guarantees the feature
scale never drifts as evidence grows.

The caller (script or user) is responsible for fitting and passing the scaler.
`run_single_al_cycle()` never fits — only transforms.

---

## Real-lab script: `scripts/run_lab_cycle.py`

### Experiment folder layout

```
{output_dir}/{experiment_name}/
  dataset/
    POOL.csv            # immutable: full search space (features only)
    EVIDENCE.csv        # grows each cycle (features + targets)
    CANDIDATES.csv      # shrinks each cycle (features only)
    scaler.joblib       # fit once on POOL at init
  cycle_0/
    output_sampled.csv  # script output — points to measure (target=NaN)
    predictions.csv     # model mean + uncertainty over full pool
    landscapes.csv      # acquisition landscape per entry
    model_snapshot.pkl
    log.json
    validated.csv       # USER fills this in after measuring (triggers cycle 1)
  cycle_1/
    ...
```

### Output path resolution

Default: `LAB_AL_REPO / experiment_name` where `LAB_AL_REPO` is defined in `format.py`.
Override: set `output_dir` in the YAML config; the experiment folder becomes
`Path(output_dir) / experiment_name`.

### Cycle detection

On startup the script counts how many `cycle_N/` folders exist in the experiment directory.
That count is the current cycle index.

### State update between cycles

Before running cycle N (N > 0) the script:
1. Reads `cycle_{N-1}/validated.csv`
2. Validates it (see below)
3. Appends rows to `EVIDENCE.csv`
4. Removes those rows from `CANDIDATES.csv`

### Validation of `validated.csv`

Checked before any state update:
- File exists
- Has all `search_vars` columns
- Has all `target_vars` columns
- No NaN values in `target_vars`
- Row count matches `output_sampled.csv`
- Feature values match `output_sampled.csv` exactly (point-by-point)

Any failure raises a clear error with instructions for the user.

### YAML config keys

```yaml
experiment_name: "my_lab_experiment"
experiment_notes: ""
output_dir: null              # null → use LAB_AL_REPO from format.py

search_space_variables: [...]
target_variables: [...]

data_scaler: "StandardScaler" # or "MinMaxScaler"

ml_model: "GPR"
model_parameters:
  kernel: "MATERN_W"
  n_restarts_optimizer: 20
  alpha: 1.0e-10
  normalize_y: true

batch_selection_method: "highest_landscape"
batch_selection_params:
  percentile: 95
  sampling_method: "voronoi"

penalization_params:
  radius: 0.25
  strength: 1.0

acquisition_parameters:
  - acquisition_mode: "expected_improvement"
    n_points: 5
    xi: 0.01

ground_truth_file: null       # if set: auto-validate each cycle (in-silico mode)
```

### In-silico mode

If `ground_truth_file` is set in the config, after writing `output_sampled.csv` the
script automatically looks up target values from the GT CSV and writes `validated.csv`
itself. This turns the single-cycle script into a simple N-cycle simulation without
requiring the benchmark machinery.

---

## In-silico simulation script: `scripts/insilico_lab_al_simulation.py`

This script already exists but has **broken API calls** against the current codebase:

| Broken call | Fix |
|---|---|
| `setup_ml_model(config)` | `setup_multi_property_ml_model(config, target_vars)` |
| `sampling_block(..., sampling_mode=CYCLE_SAMPLING)` | new signature: `batch_selection_method`, `batch_selection_params` |
| Single-property `Y_train` 1-D | must be 2-D `(N, P)` |

The YAML config also needs `batch_selection_method` and `batch_selection_params` added.

---

## Deletions

| Path | Reason |
|---|---|
| `activereg/cycle.py` | All logic superseded by `experiment.py`; `prepare_training_data` migrated |
| `run_labAL_cycle.py` (root) | Replaced by `scripts/run_lab_cycle.py` |

---

## `format.py` addition

```python
LAB_AL_REPO = REPO_ROOT / 'lab_al_experiments'
```

---

## README updates

- Package Structure: remove `cycle.py` entry
- Quick Start: replace `lab_al_cycle` with `run_single_al_cycle` usage pattern
- Add **Lab Use** section: single-cycle workflow + insilico simulation
- Update Configuration section: add `batch_selection_method`/`batch_selection_params`
