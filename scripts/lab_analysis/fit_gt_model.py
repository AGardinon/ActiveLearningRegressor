#!/usr/bin/env python3
"""
Fit a GT surrogate model on the ground truth data and produce dense predictions over the full design space.

The ground truth covers ~495/1000 design space points.  This script trains an
activereg GPR (optionally log-transformed, optionally grid-searched) and predicts
all 1000 points, providing a dense reference for RMSE computation comparable to
the benchmark rmse_vs_gt_pool metric.

--log-transform is recommended: it enforces non-negative predictions and reduces
the peak-attenuation bias that arises when a linear-scale GPR smooths already-averaged
GT measurements.

Inputs (from --analysis-dir):
  ground_truth.csv   — sparse GT with row_idx
  design_space.csv   — full 1000-point design space

Outputs (in --analysis-dir):
  gt_predictions.npy     — mean predictions for all 1000 design space points
  gt_predictions_std.npy — prediction std for all 1000 points
  gt_model.pkl           — fitted model + scaler

Usage:
    python scripts/lab_analysis/fit_gt_model.py
    python scripts/lab_analysis/fit_gt_model.py --log-transform
    python scripts/lab_analysis/fit_gt_model.py --log-transform --grid-search
    python scripts/lab_analysis/fit_gt_model.py --kernel RBF_W --n-restarts 50
"""

import argparse
import pickle
from functools import partial

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from activereg.mlmodel import GPR
from activereg.hyperparams import (
    get_gp_kernel,
    grid_search_cv,
    GPR_MATERN_PARAM_GRID,
    GPR_RBF_PARAM_GRID,
)

COORD_COLS = [
    "Conc_AP_Lys_100_2_(mM)",
    "Conc_AP_Asp_100_2_(mM)",
    "Conc_NaCl_(mM)",
]

KERNEL_GRIDS = {
    'MATERN_W': GPR_MATERN_PARAM_GRID,
    'RBF_W':    GPR_RBF_PARAM_GRID,
}


def fit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str,
    n_restarts: int,
    log_transform: bool,
    grid_search: bool,
    cv: int,
) -> GPR:
    if grid_search:
        print(f"Running grid search (cv={cv}) over {kernel} parameter grid...")
        model_factory = partial(GPR, log_transform=log_transform)
        result = grid_search_cv(
            model_factory=model_factory,
            param_grid=KERNEL_GRIDS[kernel],
            X=X_train,
            y=y_train,
            cv=cv,
            scoring='neg_mean_squared_error',
            verbose=1,
        )
        best_params = result['best_params']
        print(f"  Best params: {best_params}")
        model = GPR(log_transform=log_transform, **best_params)
    else:
        model = GPR(
            log_transform=log_transform,
            kernel=get_gp_kernel(kernel),
            normalize_y=True,
            n_restarts_optimizer=n_restarts,
            random_state=13,
        )

    print(f"Fitting GPR (log_transform={log_transform}) on {len(X_train)} GT points...")
    model.train(X_train, y_train)
    print(f"  Fitted kernel: {model.model.kernel_}")
    print(f"  Log-marginal-likelihood: {model.model.log_marginal_likelihood_value_:.3f}")
    return model


def main(
    analysis_dir: Path,
    kernel: str,
    n_restarts: int,
    log_transform: bool,
    grid_search: bool,
    cv: int,
) -> None:
    gt = pd.read_csv(analysis_dir / "ground_truth.csv")
    ds = pd.read_csv(analysis_dir / "design_space.csv", index_col="row_idx")

    X_train = gt[COORD_COLS].values
    y_train = gt["Mean_Diameter_mean"].values
    X_pool = ds[COORD_COLS].values

    if log_transform and np.any(y_train <= 0):
        import warnings
        n_nonpos = int(np.sum(y_train <= 0))
        warnings.warn(
            f"--log-transform requested but {n_nonpos} non-positive values found in "
            f"ground_truth.csv (zeros = valid 'no particles' measurements). "
            f"Falling back to linear-scale GPR with post-hoc clipping at 0.",
            UserWarning,
            stacklevel=2,
        )
        log_transform = False

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_pool_s = scaler.transform(X_pool)

    model = fit_model(X_train_s, y_train, kernel, n_restarts, log_transform, grid_search, cv)

    print("Predicting all 1000 design space points...")
    _, y_pred, y_std = model.predict(X_pool_s)

    # Clip at 0: GPR posterior can go slightly negative near zero-boundary regions.
    # Zero is the physically meaningful floor (no particles formed).
    y_pred = np.maximum(y_pred, 0.0)

    np.save(analysis_dir / "gt_predictions.npy", y_pred)
    np.save(analysis_dir / "gt_predictions_std.npy", y_std)

    with open(analysis_dir / "gt_model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    print("\nGT prediction summary:")
    print(f"  Mean_Diameter range: {y_pred.min():.3f} - {y_pred.max():.3f}")
    print(f"  Mean uncertainty:    {y_std.mean():.3f}")
    print(f"\nSaved -> {analysis_dir / 'gt_predictions.npy'}")
    print(f"         {analysis_dir / 'gt_predictions_std.npy'}")
    print(f"         {analysis_dir / 'gt_model.pkl'}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    default_analysis = repo_root / "lab_experiments" / "analysis"

    parser = argparse.ArgumentParser(
        description="Fit GT surrogate model and predict full design space"
    )
    parser.add_argument("--analysis-dir", type=Path, default=default_analysis)
    parser.add_argument(
        "--kernel", choices=list(KERNEL_GRIDS), default="MATERN_W",
        help="Kernel recipe (default: MATERN_W). Ignored when --grid-search is set.",
    )
    parser.add_argument(
        "--n-restarts", type=int, default=20,
        help="GPR optimizer restarts (default: 20). Ignored when --grid-search is set.",
    )
    parser.add_argument(
        "--log-transform", action="store_true",
        help="Fit in log10 space — enforces non-negative predictions and reduces peak attenuation.",
    )
    parser.add_argument(
        "--grid-search", action="store_true",
        help="Run cross-validated grid search over kernel hyperparameters.",
    )
    parser.add_argument(
        "--cv", type=int, default=5,
        help="CV folds for grid search (default: 5).",
    )
    args = parser.parse_args()

    main(
        args.analysis_dir,
        args.kernel,
        args.n_restarts,
        args.log_transform,
        args.grid_search,
        args.cv,
    )
