#!

import numpy as np
from typing import Union, Dict, List, Callable, Any

from activereg.mlmodel import KernelFactory
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================================
# Kernel recipes for GPR
# ============================================================================

GPR_KERNELS = {
    'RBF_W': ConstantKernel() * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.),
    'MATERN_W': ConstantKernel() * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.),
}

# Backward compatibility alias
GPR = GPR_KERNELS


def get_gp_kernel(kernel_recipe: Union[str, list]):
    """Gets the Gaussian Process kernel from a recipe.

    Args:
        kernel_recipe (str or list): Name of the kernel or list of kernel components.

    Returns:
        _type_: The corresponding Gaussian Process kernel.
    """
    if isinstance(kernel_recipe, str):
        return get_default_gp_kernel(kernel_recipe)

    elif isinstance(kernel_recipe, list):
        return get_custom_gp_kernel(kernel_recipe)

    else:
        raise ValueError("kernel_recipe must be either a string or a list.")


def get_default_gp_kernel(kernel_recipe: str):
    """Gets the Gaussian Process kernel by name.

    Args:
        name (str): Name of the kernel.

    Returns:
        _type_: The corresponding Gaussian Process kernel.
    """
    if kernel_recipe not in GPR_KERNELS:
        raise ValueError(f"Kernel '{kernel_recipe}' is not defined. Choose from {list(GPR_KERNELS.keys())}.")
    return GPR_KERNELS[kernel_recipe]


def get_custom_gp_kernel(kernel_recipe: list):
    """Gets a custom Gaussian Process kernel from a recipe.

    Args:
        kernel_recipe (list): List of kernel components.

    Returns:
        _type_: The corresponding custom Gaussian Process kernel.
    """
    kernel_factory = KernelFactory(kernel_recipe)
    return kernel_factory.get_kernel()


# ============================================================================
# Default hyperparameter grids for grid search
# ============================================================================

GPR_PARAM_GRID = {
    'kernel': [
        ConstantKernel(1.0) * RBF(lc) + WhiteKernel(noise_level=noise)
        for lc in [0.01, 0.1, 1.0, 10.0]
        for noise in [0.75, 0.5, 0.25, 0.1]
    ],
    'alpha': [1e-12, 1e-10, 1e-8, 1e-6, 1e-4],
    'normalize_y': [True, False],
    'n_restarts_optimizer': [50],
}

KNN_PARAM_GRID = {
    'k': [3, 5, 10, 15, 20],
    'length_scale': [0.1, 0.5, 1.0, 5.0, 10.0],
}

MLP_PARAM_GRID = {
    'hidden_layers': [[32], [64], [32, 32], [64, 64]],
    'lr': [1e-4, 1e-3, 1e-2],
    'lambda_anchor': [1e-5, 1e-4, 1e-3],
}

BNN_PARAM_GRID = {
    'hidden_layers': [[32, 32], [64, 64], [32, 64]],
    'lr': [1e-4, 1e-3],
    'weight_sigma': [0.5, 1.0, 2.0],
}


# ============================================================================
# Scoring functions
# ============================================================================

def _compute_score(y_true: np.ndarray, y_pred: np.ndarray, scoring: str) -> float:
    """Compute a scoring metric. Returns values where higher is better (negated for errors)."""
    if scoring == 'neg_mean_squared_error':
        return -mean_squared_error(y_true, y_pred)
    elif scoring in ('neg_rmse', 'neg_root_mean_squared_error'):
        return -np.sqrt(mean_squared_error(y_true, y_pred))
    elif scoring == 'r2':
        return r2_score(y_true, y_pred)
    else:
        raise ValueError(f"Unknown scoring metric: '{scoring}'. "
                         f"Choose from: 'neg_mean_squared_error', 'neg_rmse', 'r2'.")


# ============================================================================
# Grid search cross-validation
# ============================================================================

def grid_search_cv(
    model_factory: Callable[..., Any],
    param_grid: Dict[str, list],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'neg_mean_squared_error',
    verbose: int = 0,
) -> Dict[str, Any]:
    """Grid search cross-validation for any MLModel-compatible class.

    Exhaustively searches over the parameter grid, evaluating each combination
    using k-fold cross-validation with the model's train() and predict() interface.

    Args:
        model_factory: Callable that accepts **params and returns an MLModel instance.
                       Can be a class (e.g. GPR), a lambda, or functools.partial.
        param_grid: Dict mapping parameter names to lists of values to search.
        X: Training features, shape (n_samples, n_features).
        y: Training targets, shape (n_samples,).
        cv: Number of cross-validation folds.
        scoring: Scoring metric. One of 'neg_mean_squared_error', 'neg_rmse', 'r2'.
                 For error metrics, values are negated so that higher is always better.
        verbose: Verbosity level. 0=silent, 1=progress summary, 2=per-combination detail.

    Returns:
        Dict with keys:
            - 'best_params': dict of the best parameter combination
            - 'best_score': float, best mean CV score (higher is better)
            - 'cv_results': list of dicts, one per parameter combination, each with
              'params', 'mean_score', 'std_score', 'fold_scores'

    Example:
        >>> from activereg.mlmodel import GPR
        >>> from activereg.hyperparams import grid_search_cv, GPR_PARAM_GRID
        >>> result = grid_search_cv(GPR, GPR_PARAM_GRID, X_train, y_train)
        >>> best_model = GPR(**result['best_params'])

        >>> # With fixed params using functools.partial
        >>> from functools import partial
        >>> factory = partial(AnchoredEnsembleMLP, n_models=5, in_feats=6, out_feats=1)
        >>> result = grid_search_cv(factory, MLP_PARAM_GRID, X_train, y_train)
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    param_combinations = list(ParameterGrid(param_grid))
    n_combinations = len(param_combinations)
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

    best_score = -np.inf
    best_params = None
    cv_results = []

    for i, params in enumerate(param_combinations):
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Instantiate and train model
            model = model_factory(**params)
            model.train(X_train, y_train)

            # Predict and score — predict returns (y_hat, y_mean, y_std)
            _, y_pred, _ = model.predict(X_val)
            score = _compute_score(y_val, y_pred, scoring)
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        cv_results.append({
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score,
            'fold_scores': fold_scores,
        })

        if verbose >= 2:
            print(f"[{i+1}/{n_combinations}] {params} -> {scoring}={mean_score:.4f} (+/- {std_score:.4f})")

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    if verbose >= 1:
        print(f"Best {scoring}: {best_score:.4f} with params: {best_params}")

    return {
        'best_params': best_params,
        'best_score': best_score,
        'cv_results': cv_results,
    }
