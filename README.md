# ActiveLearningRegressor

**`activereg`** — An active learning framework for supervised regression problems.

Active learning reduces labeling cost by iteratively selecting the most informative candidates from an unlabeled pool, training on them, and refining the model. This repository provides the full infrastructure: acquisition functions, sampling strategies, ML model backends, evaluation metrics, and benchmark scripts.

---

## Table of Contents

- [Overview](#overview)
- [Package Structure](#package-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Lab Use](#lab-use)
- [ML Models](#ml-models)
- [Acquisition Functions](#acquisition-functions)
- [Sampling Strategies](#sampling-strategies)
- [Configuration](#configuration)
- [Running Benchmarks](#running-benchmarks)
- [License](#license)

---

## Overview

The active learning loop in `activereg` follows a standard pool-based protocol:

```
Initial labeled set
        │
        ▼
   Train ML model
        │
        ▼
  Predict on pool  ──►  Acquisition landscape
        │
        ▼
  Select batch  ──►  FPS / Voronoi / random
        │
        ▼
   Label batch  ──►  add to training set
        │
        └──── repeat ◄────────────────────┘
```

Each iteration expands the training set with the most informative points according to the chosen acquisition function, converging toward an accurate model with fewer labels than random or grid sampling.

---

## Package Structure

```
activereg/
├── mlmodel/                # ML model backends (subpackage)
│   ├── _base.py            # MLModel protocol definition
│   ├── _gpr.py             # Gaussian Process Regressor
│   ├── _knn.py             # k-Nearest Neighbours regressor
│   ├── _mlp.py             # MLP and Anchored Ensemble MLP
│   ├── _bnn.py             # Bayesian Neural Network (PyTorch + Pyro)
│   └── _kernels.py         # Kernel factory for GPs
├── acquisition.py          # Acquisition functions and batch selection
├── adaptiveRefinement.py   # Adaptive spatial refinement strategies
├── benchmarkFunctions.py   # Benchmark test functions (Hartmann, Ackley, …)
├── beauty.py               # Visualization utilities
├── data.py                 # Dataset generation (LHS, Sobol, random)
├── experiment.py           # Experiment setup, model factory, and AL cycle core
├── hyperparams.py          # Hyperparameter grids and grid search utilities
├── metrics.py              # Evaluation metrics (RMSE, NLL, PICP, MPIW, …)
├── sampling.py             # Sampling methods (FPS, Voronoi, random)
├── utils.py                # Miscellaneous utilities
└── format.py               # Repository path constants
```

All ML models expose the same three-method protocol (see `MLModel` in `activereg/mlmodel/_base.py`):

| Method | Signature | Description |
|---|---|---|
| `train` | `(X, y) → None` | Fit model to labeled data |
| `predict` | `(X) → (ŷ, mean, uncertainty)` | Predict with uncertainty estimate |
| `__repr__` | `() → str` | Human-readable model description |

---

## Installation

### Requirements

- Python 3.10+
- PyTorch (CPU or GPU)

### Install from source

```bash
git clone https://github.com/AGardinon/ActiveLearningRegressor.git
cd ActiveLearningRegressor
pip install -e .
```

This installs `activereg` in editable mode together with all required dependencies.

### Optional: GPU-accelerated PyTorch

The default install pulls the CPU-only PyTorch wheel. For a CUDA build, install PyTorch separately **before** running `pip install -e .` — follow the [official PyTorch install guide](https://pytorch.org/get-started/locally/).

---

## Quick Start

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from activereg.experiment import (
    setup_multi_property_ml_model,
    setup_data_pool,
    run_single_al_cycle,
)

# --- data ---
# pool_df: search space (features only)
# evidence_df: initial labeled samples (features + targets)
pool_df     = pd.read_csv("datasets/my_pool.csv")
evidence_df = pd.read_csv("datasets/my_evidence.csv")

search_vars = ["x1", "x2", "x3"]
target_vars = ["y"]

# --- scaler: fit once on the full pool, reuse every cycle ---
scaler = StandardScaler()
_, scaler = setup_data_pool(df=pool_df, search_var=search_vars, scaler=scaler)

# --- model config ---
config = {
    "ml_model": "GPR",
    "model_parameters": {"kernel": "MATERN_W", "n_restarts_optimizer": 10, "alpha": 1e-10},
}
acquisition_params = [{"acquisition_mode": "expected_improvement", "n_points": 5, "xi": 0.01}]

# --- active learning cycle ---
for cycle in range(10):
    model = setup_multi_property_ml_model(config, target_vars)
    result = run_single_al_cycle(
        pool_df=pool_df,
        evidence_df=evidence_df,
        scaler=scaler,
        ml_model=model,
        search_vars=search_vars,
        target_vars=target_vars,
        acquisition_params=acquisition_params,
        batch_selection_method="highest_landscape",
        batch_selection_params={"percentile": 95, "sampling_method": "voronoi"},
    )
    # next_batch_df contains the selected candidates with NaN targets
    # measure them and add to evidence_df for the next cycle
    new_points = result["next_batch_df"]
    print(f"Cycle {cycle+1}: selected {len(new_points)} candidates")
    # evidence_df = pd.concat([evidence_df, measured_points])
```

---

## Lab Use

`activereg` provides two additional workflows for experiments that cannot be run as a single automated benchmark.

### Real-lab single-cycle workflow

One AL step at a time, with physical measurement between cycles:

```
Cycle 0: python scripts/run_lab_cycle.py -c config.yaml
         → writes cycle_0/output_sampled.csv
              (fill in measured values → cycle_0/validated.csv)
Cycle 1: python scripts/run_lab_cycle.py -c config.yaml
         → reads validated.csv, updates evidence, writes cycle_1/output_sampled.csv
         ...
```

**Folder layout** created automatically on the first run:

```
lab_al_experiments/{experiment_name}/
  dataset/
    POOL.csv          # immutable: full search space (features only)
    EVIDENCE.csv      # grows each cycle (features + targets)
    CANDIDATES.csv    # shrinks each cycle (features only)
    scaler.joblib     # fit once on POOL
  cycle_0/
    output_sampled.csv   # points to measure (targets = NaN)
    predictions.csv      # model predictions over pool
    landscapes.csv       # acquisition landscape per entry
    model_snapshot.pkl
    log.json
    validated.csv        # fill in measurements, then re-run
  cycle_1/
    ...
```

Copy and edit `scripts/lab_cycle_config_template.yaml` to configure your experiment.

### In-silico simulation

Set `ground_truth_file` in the config to run the full multi-cycle loop automatically — the script looks up target values from the CSV after each cycle, so no manual measurement step is needed:

```yaml
ground_truth_file: "ackley3d_10000pts.csv"   # in datasets/
```

```bash
python scripts/insilico_lab_al_simulation.py -c scripts/insilico_lab_al_simulation_config.yaml
```

---

## ML Models

| Model | Class | Backend | Uncertainty source |
|---|---|---|---|
| Gaussian Process Regressor | `GPR` | scikit-learn | Posterior variance |
| k-Nearest Neighbours | `kNNRegressorAL` | scikit-learn | Neighbourhood spread |
| Multi-Layer Perceptron | `MLP` | PyTorch | Dropout / ensemble |
| Anchored Ensemble MLP | `AnchoredEnsembleMLP` | PyTorch | Ensemble disagreement |
| Bayesian Neural Network | `BayesianNN` | PyTorch + Pyro | Variational posterior |

Models are configured via YAML files in `scripts/mlmodel_config/` and instantiated through the factory:

```python
from activereg.experiment import setup_ml_model

model = setup_ml_model(
    ml_model_type="GPR",
    ml_model_params={"kernel": "matern", "nu": 2.5},
)
```

### GPR kernels

The `KernelFactory` in `activereg/mlmodel/_kernels.py` supports composing standard scikit-learn kernels (RBF, Matérn, WhiteKernel, ConstantKernel). Pre-defined kernel recipes are available in `hyperparams.py`.

---

## Acquisition Functions

Acquisition functions map the model's predictions over the unlabeled pool to a scalar *informativeness* score for each candidate point.

| Mode | Description |
|---|---|
| `UCB` | Upper Confidence Bound — balances mean and uncertainty |
| `EI` | Expected Improvement over current best |
| `TEI` | Target Expected Improvement — steers toward a target value |
| `uncertainty` | Pure uncertainty sampling |
| `max_pred` | Maximum predicted value |

```python
from activereg.acquisition import AcquisitionFunction

acq = AcquisitionFunction(mode="UCB", kappa=2.0)
landscape = acq.compute(y_mean, y_uncertainty)
batch_idx = acq.select_batch(landscape, n=5)
```

Batch selection applies a Gaussian penalty around already-selected points to encourage diversity within a single batch (`penalize_landscape_fast`).

---

## Sampling Strategies

Once the acquisition landscape is computed, a batch of candidates is drawn using one of three strategies:

| Strategy | Function | Description |
|---|---|---|
| Farthest Point Sampling | `fps` | Maximises pairwise distance in feature space |
| Voronoi | `voronoi` | Cluster-aware spatial coverage |
| Random | `rnd` | Uniform random draw from top percentile |

```python
from activereg.sampling import sample_landscape

batch = sample_landscape(
    X_pool, landscape, n=5, method="fps"
)
```

---

## Configuration

Experiments are controlled by YAML configuration files under `scripts/`:

```
scripts/
├── general_config/
│   ├── benchmark_config.yaml          # experiment-level settings
│   ├── acquisition_mode_settings.yaml # acquisition function parameters
│   └── target_function_config.yaml    # benchmark function / dataset settings
├── mlmodel_config/
│   └── model_config.yaml              # ML model type and hyperparameters
└── lab_cycle_config_template.yaml     # template for run_lab_cycle.py
```

Key configuration fields shared across benchmark and lab scripts:

```yaml
ml_model: "GPR"
model_parameters:
  kernel: "MATERN_W"          # kernel recipe; see hyperparams.py
  n_restarts_optimizer: 20
  alpha: 1.0e-10
  normalize_y: true

batch_selection_method: "highest_landscape"   # highest_landscape | constant_liar | …
batch_selection_params:
  percentile: 95
  sampling_method: "voronoi"  # voronoi | fps | random

acquisition_parameters:
  - acquisition_mode: "expected_improvement"
    n_points: 5
    xi: 0.01
```

---

## Running Benchmarks

Benchmark experiments compare acquisition strategies on synthetic test functions (Hartmann3, Hartmann6, Ackley, Styblinski-Tang) across dimensions.

```bash
# Run benchmark with synthetic functions
bash scripts/run_benchmark_funcs.sh

# Or directly with Python
python scripts/benchmark_functions.py \
    --benchmark_config  scripts/general_config/benchmark_config.yaml \
    --model_config      scripts/mlmodel_config/model_config.yaml \
    --acq_config        scripts/general_config/acquisition_mode_settings.yaml \
    --target_config     scripts/general_config/target_function_config.yaml
```

Results (training CSVs, metric logs, figures) are written to `benchmarks/`.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
