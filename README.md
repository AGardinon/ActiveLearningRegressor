# ActiveLearningRegressor

**`activereg`** — An active learning framework for supervised regression problems.

Active learning reduces labeling cost by iteratively selecting the most informative candidates from an unlabeled pool, training on them, and refining the model. This repository provides the full infrastructure: acquisition functions, sampling strategies, ML model backends, evaluation metrics, and benchmark scripts.

---

## Table of Contents

- [Overview](#overview)
- [Package Structure](#package-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
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
├── cycle.py                # Active learning cycle orchestration
├── data.py                 # Dataset generation (LHS, Sobol, random)
├── experiment.py           # Experiment setup and model factory
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
from activereg.experiment import setup_ml_model, setup_data_pool
from activereg.cycle import lab_al_cycle
from activereg.acquisition import AcquisitionFunction

# --- data ---
# gt_df: ground-truth DataFrame with features + target column
# evidence_df: initial labeled samples
gt_df = pd.read_csv("datasets/my_experiment.csv")
evidence_df = gt_df.sample(5, random_state=0)

feature_cols = ["x1", "x2", "x3"]
target_col   = "y"

# --- model ---
model_params = {"kernel": "matern", "nu": 2.5, "n_restarts": 5}
model = setup_ml_model("GPR", model_params)

# --- pool ---
X_pool, scaler = setup_data_pool(gt_df[feature_cols].values)

# --- acquisition ---
acq = AcquisitionFunction(mode="UCB", kappa=2.0)

# --- active learning cycle ---
for cycle in range(10):
    X_train = evidence_df[feature_cols].values
    y_train = evidence_df[target_col].values

    new_batch = lab_al_cycle(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_pool=X_pool,
        acq=acq,
        batch_size=3,
        scaler=scaler,
    )
    evidence_df = pd.concat([evidence_df, gt_df.loc[new_batch]])
    print(f"Cycle {cycle+1}: {len(evidence_df)} labeled points")
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
└── mlmodel_config/
    └── model_config.yaml              # ML model type and hyperparameters
```

Example `model_config.yaml`:

```yaml
ml_model_type: GPR
ml_model_params:
  kernel: matern
  nu: 2.5
  n_restarts: 5
  alpha: 1.0e-6
```

Example `acquisition_mode_settings.yaml`:

```yaml
mode: UCB
kappa: 2.0
batch_size: 5
sampling_method: fps
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
