#!

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Literal, Tuple, Dict

# --------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------
EPS = 1e-12  # small constant for numerical stability
CONFIDENCE = 0.95  # default confidence level

# --------------------------------------------------------------
# Per-Cycle evaluation function
# --------------------------------------------------------------

def evaluate_cycle_metrics(
    y_pred_pool: np.ndarray,
    y_pred_val: np.ndarray | None,
    y_uncertainty_val: np.ndarray | None,
    y_true_pool: np.ndarray,
    y_true_val: np.ndarray | None,
) -> dict:
    """Evaluate cycle metrics.

    Args:
        y_pred_pool (np.ndarray): Predicted values for the pool.
        y_pred_val (np.ndarray | None): Predicted values for the validation set.
        y_uncertainty_val (np.ndarray | None): Uncertainty estimates for the validation set.
        y_true_pool (np.ndarray): True values for the pool.
        y_true_val (np.ndarray | None): True values for the validation set.

    Returns:
        dict: Dictionary with evaluation metrics.
    """
    return {
        "y_best_predicted_pool": np.max(y_pred_pool),
        "y_best_predicted_val": np.max(y_pred_val) if y_pred_val is not None else np.nan,
        "rmse_vs_gt_pool": np.sqrt(mean_squared_error(y_true=y_true_pool, y_pred=y_pred_pool)),
        "mae_vs_gt_pool": mean_absolute_error(y_true=y_true_pool, y_pred=y_pred_pool),
        "rmse_vs_gt_val": np.sqrt(mean_squared_error(y_true=y_true_val, y_pred=y_pred_val)) if y_pred_val is not None else np.nan,
        "mae_vs_gt_val": mean_absolute_error(y_true=y_true_val, y_pred=y_pred_val) if y_pred_val is not None else np.nan,
        "nll_val": nll_gauss(y_true=y_true_val, y_mean=y_pred_val, y_std=y_uncertainty_val) if (y_pred_val is not None and y_uncertainty_val is not None) else np.nan,
        "picp95_val": picp(y_true=y_true_val, y_mean=y_pred_val, y_std=y_uncertainty_val) if (y_pred_val is not None and y_uncertainty_val is not None) else np.nan,
        "mpiw95_val": mpiw(y_std=y_uncertainty_val) if (y_pred_val is not None and y_uncertainty_val is not None) else np.nan,
    }

# --------------------------------------------------------------------------------
# Uncertainty & regression metrics
# --------------------------------------------------------------

def nll_gauss(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, eps=EPS) -> float:
    """Negative log-likelihood under Gaussian assumption.

    Args:
        y_true (np.ndarray): True values.
        y_mean (np.ndarray): Predicted mean values.
        y_std (np.ndarray): Predicted standard deviations.
        eps (float, optional): Small constant for numerical stability. Defaults to EPS=1e-12.

    Returns:
        float: Negative log-likelihood.
    """
    y_std = np.clip(y_std, eps, None)
    return 0.5 * np.mean(np.log(2 * np.pi * y_std**2) + ((y_true - y_mean)**2) / (y_std**2))
           

def picp(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, confidence=CONFIDENCE) -> float:
    """Prediction Interval Coverage Probability (PICP).

    Args:
        y_true (np.ndarray): True values.
        y_mean (np.ndarray): Predicted mean values.
        y_std (np.ndarray): Predicted standard deviations.
        confidence (float, optional): Confidence level. Defaults to CONFIDENCE=0.95.

    Returns:
        float: Prediction Interval Coverage Probability (PICP).
    """
    z = norm.ppf(0.5 + confidence / 2.0)
    lower = y_mean - z * y_std
    upper = y_mean + z * y_std
    inside = (y_true >= lower) & (y_true <= upper)
    return np.mean(inside)


def mpiw(y_std: np.ndarray, confidence=CONFIDENCE) -> float:
    """Mean Prediction Interval Width (MPIW).

    Args:
        y_mean (np.ndarray): Predicted mean values.
        y_std (np.ndarray): Predicted standard deviations.
        confidence (float, optional): Confidence level. Defaults to CONFIDENCE=0.95.

    Returns:
        float: Mean Prediction Interval Width (MPIW).
    """
    z = norm.ppf(0.5 + confidence / 2.0)
    interval_widths = 2 * z * y_std
    return np.mean(interval_widths)


# --------------------------------------------------------------
# Composite convergence metrics
# --------------------------------------------------------------

def compute_experiment_scores(
    experiments_data: pd.DataFrame,
    X: str,
    Y: str,
    method: Literal['euclidean_improvement', 'manhattan_improvement', 'pareto_dominance', 
                    'average_descent', 'efficiency_ratio', 'weighted_improvement'] = 'euclidean_improvement',
    weight_x: float = 1.0,
    weight_y: float = 1.0,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Compute performance scores for experiments based on their trajectory in metric space.
    """
    
    scores = {}
    ideal_x, ideal_y = 0.0, 0.0  # Ideal point for both metrics
    
    for exp in experiments_data['experiment'].unique():
        # Extract trajectory
        exp_data = experiments_data[experiments_data['experiment'] == exp]
        x_data = exp_data[X].values
        y_data = exp_data[Y].values
        
        if len(x_data) < 2:
            scores[exp] = 0.0
            continue
        
        # Compute trajectory derivatives
        dx = np.diff(x_data)  # Δx_t = x_t - x_{t-1}
        dy = np.diff(y_data)  # Δy_t = y_t - y_{t-1}
        T = len(dx)  # Number of steps
        
        # Compute score based on method
        if method == 'euclidean_improvement':
            # L2 distance improvement
            d_start = np.sqrt((x_data[0] - ideal_x)**2 + (y_data[0] - ideal_y)**2)
            d_end = np.sqrt((x_data[-1] - ideal_x)**2 + (y_data[-1] - ideal_y)**2)
            score = d_start - d_end
            
        elif method == 'manhattan_improvement':
            # L1 distance improvement
            d_start = np.abs(x_data[0] - ideal_x) + np.abs(y_data[0] - ideal_y)
            d_end = np.abs(x_data[-1] - ideal_x) + np.abs(y_data[-1] - ideal_y)
            score = d_start - d_end
            
        elif method == 'pareto_dominance':
            # Pareto-improving steps with penalties
            pareto_steps = (dx < 0) & (dy < 0)
            pareto_count = np.sum(pareto_steps)
            
            # Forward improvement (when metrics decrease)
            forward_x = np.sum(-dx[dx < 0])
            forward_y = np.sum(-dy[dy < 0])
            forward_improvement = forward_x + forward_y
            
            # Backward penalty (when metrics increase)
            backward_x = np.sum(dx[dx > 0])
            backward_y = np.sum(dy[dy > 0])
            backward_penalty = backward_x + backward_y
            
            # Combined score: weight Pareto steps highly, add improvements, subtract penalties
            pareto_weight = 1.0  # Weight for each Pareto-improving step
            score = pareto_weight * pareto_count + forward_improvement - backward_penalty
            
        elif method == 'average_descent':
            # Mean per-step improvement (negative gradient toward origin)
            # Higher negative (dx + dy) means better descent
            per_step_improvement = -(dx + dy)
            score = np.mean(per_step_improvement)
            normalize = False  # Already normalized by taking mean
            
        elif method == 'efficiency_ratio':
            # Path straightness: direct distance / actual path length
            direct_distance = np.sqrt((x_data[-1] - x_data[0])**2 + 
                                     (y_data[-1] - y_data[0])**2)
            
            step_distances = np.sqrt(dx**2 + dy**2)
            path_length = np.sum(step_distances)
            
            if path_length > 0:
                score = direct_distance / path_length
            else:
                score = 0.0
            
            # Also consider if we're moving toward origin (not away)
            if x_data[-1] > x_data[0] or y_data[-1] > y_data[0]:
                score *= -1  # Penalize if ending farther from origin
                
            normalize = False  # Ratio is already normalized
            
        elif method == 'weighted_improvement':
            # Weighted distance improvement
            d_start = np.sqrt(weight_x * (x_data[0] - ideal_x)**2 + 
                            weight_y * (y_data[0] - ideal_y)**2)
            d_end = np.sqrt(weight_x * (x_data[-1] - ideal_x)**2 + 
                          weight_y * (y_data[-1] - ideal_y)**2)
            score = d_start - d_end
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize by number of steps if requested
        # This makes experiments with different lengths comparable
        if normalize and T > 0:
            score = score / T
        
        scores[exp] = score
    
    return scores
