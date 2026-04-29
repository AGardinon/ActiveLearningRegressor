#!

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Literal, Tuple, Dict, List

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
) -> Dict[str, float]:
    """Evaluate cycle metrics.

    Args:
        y_pred_pool (np.ndarray): Predicted values for the pool.
        y_pred_val (np.ndarray | None): Predicted values for the validation set.
        y_uncertainty_val (np.ndarray | None): Uncertainty estimates for the validation set.
        y_true_pool (np.ndarray): True values for the pool.
        y_true_val (np.ndarray | None): True values for the validation set.

    Returns:
        Dict[str, float]: Dictionary with evaluation metrics.
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

def compare_across_metrics(
    experiments_data: pd.DataFrame,
    X: str,
    Y: str,
    top_k: int = 3,
    weight_x: float = 1.0,
    weight_y: float = 1.0,
    normalize: bool = True,
    methods: List[Literal['euclidean_improvement', 'manhattan_improvement', 'pareto_dominance', 
                    'average_descent', 'efficiency_ratio', 'weighted_improvement']] = None
) -> Dict[str, Tuple[List[str], Dict[str, float]]]:
    """Compare experiments across multiple composite metrics.

    Args:
        experiments_data (pd.DataFrame): DataFrame containing experiment trajectories with columns for 'experiment', X, and Y.
        X (str): Column name for the X metric.
        Y (str): Column name for the Y metric.
        top_k (int, optional): Number of top experiments to return per method. Defaults to 3.
        weight_x (float, optional): Weight for the X metric in weighted methods. Defaults to 1.0.
        weight_y (float, optional): Weight for the Y metric in weighted methods. Defaults to 1.0.
        normalize (bool, optional): Whether to normalize scores by trajectory length. Defaults to True.
        methods (List[Literal[&#39;euclidean_improvement&#39;, &#39;manhattan_improvement&#39;, &#39;pareto_dominance&#39;, &#39;average_descent&#39;, &#39;efficiency_ratio&#39;, &#39;weighted_improvement&#39;], optional): List of scoring methods to compare. If None, defaults to all methods.

    Returns:
        Dict[str, Tuple[List[str], Dict[str, float]]]: Dictionary mapping method names to tuples of (top-k experiment names, all experiment scores).
    """
    if methods is None:
        methods = ['euclidean_improvement', 'manhattan_improvement', 'pareto_dominance', 
                   'average_descent', 'efficiency_ratio', 'weighted_improvement']
    
    print(f"Top {top_k} Experiments by Method:\n" + "="*60)

    results = {}
    for method in methods:
        top_experiments, scores = get_top_k_experiments(
            experiments_data=experiments_data,
            X=X,
            Y=Y,
            method=method,
            weight_x=weight_x,
            weight_y=weight_y,
            normalize=normalize,
            top_k=top_k
        )
        results[method] = (top_experiments, scores)

        print(f"\n{method.replace('_', ' ').title()}:")
        for rank, exp in enumerate(top_experiments, start=1):
            print(f"  {rank}. {exp:30s} (score: {scores[exp]:8.4f})")
    
    # Find consensus top experiments across methods
    all_top_experiments = [exp for method_results in results.values() for exp in method_results[0]]
    consensus_counts = pd.Series(all_top_experiments).value_counts()
    consensus_top = consensus_counts[consensus_counts == consensus_counts.max()].index.tolist()

    print("\nConsensus Top Experiments:")
    for exp in consensus_top:
        print(f"  {exp}")

    return results


def get_top_k_experiments(
    experiments_data: pd.DataFrame,
    X: str,
    Y: str,
    method: Literal['euclidean_improvement', 'manhattan_improvement', 'pareto_dominance', 
                    'average_descent', 'efficiency_ratio', 'weighted_improvement'] = 'euclidean_improvement',
    weight_x: float = 1.0,
    weight_y: float = 1.0,
    normalize: bool = True,
    top_k: int = 3
) -> Tuple[List[str], Dict[str, float]]:
    """Get top-k experiments based on computed scores.

    Args:
        experiments_data (pd.DataFrame): DataFrame containing experiment trajectories with columns for 'experiment', X, and Y.
        X (str): Column name for the X metric.
        Y (str): Column name for the Y metric.
        method (Literal[&#39;euclidean_improvement&#39;, &#39;manhattan_improvement&#39;, &#39;pareto_dominance&#39;, &#39;average_descent&#39;, &#39;efficiency_ratio&#39;, &#39;weighted_improvement&#39;], optional): Scoring method. Defaults to 'euclidean_improvement'.
        weight_x (float, optional): Weight for the X metric. Defaults to 1.0.
        weight_y (float, optional): Weight for the Y metric. Defaults to 1.0.
        normalize (bool, optional): Whether to normalize the scores. Defaults to True.
        top_k (int, optional): Number of top experiments to return. Defaults to 3.

    Returns:
        Tuple[List[str], Dict[str, float]]: A tuple containing a list of top-k experiment names and a dictionary of all experiment scores.
    """
    scores = compute_experiment_scores(
        experiments_data=experiments_data,
        X=X,
        Y=Y,
        method=method,
        weight_x=weight_x,
        weight_y=weight_y,
        normalize=normalize
    )
    
    # Sort experiments by score in descending order and return top-k keys
    sorted_experiments = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_experiments = [exp for exp, score in sorted_experiments[:top_k]]
    
    return top_experiments, scores


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


# --------------------------------------------------------------
# Multi-property Pareto metrics
# --------------------------------------------------------------

def compute_pareto_front(Y: np.ndarray, maximize: bool = True) -> np.ndarray:
    """Return a boolean mask of non-dominated (Pareto-optimal) rows in Y.

    Args:
        Y: Array of shape (N, P) — N points, P objectives.
        maximize: If True (default), higher is better for all objectives.

    Returns:
        Boolean mask of shape (N,) — True where the row is non-dominated.
    """
    Y = np.asarray(Y, dtype=float)
    n = len(Y)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        if maximize:
            dominated = np.all(Y >= Y[i], axis=1) & np.any(Y > Y[i], axis=1)
        else:
            dominated = np.all(Y <= Y[i], axis=1) & np.any(Y < Y[i], axis=1)
        dominated[i] = False
        if np.any(dominated):
            is_pareto[i] = False
    return is_pareto


def compute_hypervolume(
    Y_pareto: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """Compute the hypervolume indicator for a set of Pareto-front points.

    Closed-form sweep for P=2, O(N log N).
    Raises NotImplementedError for P > 2 (deferred to Phase 3).

    The hypervolume is the area of the objective-space region dominated by at
    least one point in Y_pareto and that itself dominates the reference point.
    For maximization the reference must satisfy r_j < min(Y_pareto[:, j]).

    Args:
        Y_pareto: Array of shape (N, 2) — non-dominated points only.
        reference_point: Array of shape (2,) dominated by all Pareto points.

    Returns:
        Hypervolume indicator as a float.
    """
    Y_pareto = np.asarray(Y_pareto, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)
    if Y_pareto.ndim == 1:
        Y_pareto = Y_pareto.reshape(1, -1)
    P = Y_pareto.shape[1]
    if P != 2:
        raise NotImplementedError(
            f"compute_hypervolume supports P=2 only (got P={P}). "
            "P>2 is deferred to Phase 3."
        )
    if len(Y_pareto) == 0:
        return 0.0
    # Sort by first objective descending; for a maximisation Pareto front
    # this yields ascending order in the second objective.
    order = np.argsort(Y_pareto[:, 0])[::-1]
    Y_s = Y_pareto[order]
    hv = 0.0
    prev_y2 = reference_point[1]
    for pt in Y_s:
        if pt[0] > reference_point[0] and pt[1] > prev_y2:
            hv += (pt[0] - reference_point[0]) * (pt[1] - prev_y2)
            prev_y2 = pt[1]
    return hv


def compute_pareto_attribution(
    points_df: pd.DataFrame,
    target_names: List[str],
    reference_point: np.ndarray,
    maximize: bool = True,
) -> pd.DataFrame:
    """Compute per-acquisition-source attribution of Pareto front contributions.

    For each acquisition source returns:
    - n_points: total points sampled from that source.
    - n_pareto_points: points on the *final* Pareto front.
    - pareto_hit_rate: n_pareto_points / n_points.
    - total_delta_hv: cumulative marginal HV gain attributed to that source
      (P=2 only; NaN otherwise).
    - delta_hv_fraction: total_delta_hv / final_total_hv (P=2 only).

    Points are processed in chronological order (by cycle). Marginal HV gain
    per point is attributed to the source that added it. Within-cycle ordering
    follows row order in points_df — marginal attribution within a cycle is
    order-dependent, which is expected for simultaneous batch acquisitions.

    Args:
        points_df: train_points_data DataFrame. Required columns:
                   [*target_names, 'acquisition_source', 'cycle'].
                   Pass a single-repetition subset when multiple reps exist.
        target_names: Objective column names, e.g. ['y1', 'y2'].
        reference_point: Array of shape (P,) for hypervolume computation.
        maximize: If True (default), higher objective values are better.

    Returns:
        DataFrame sorted by pareto_hit_rate descending.
    """
    P = len(target_names)
    compute_hv = (P == 2)
    reference_point = np.asarray(reference_point, dtype=float)

    df = points_df.sort_values('cycle').reset_index(drop=True)
    Y_all = df[target_names].to_numpy(dtype=float)
    sources = df['acquisition_source'].values

    current_Y = np.empty((0, P), dtype=float)
    current_hv = 0.0
    delta_hv_by_source: Dict[str, float] = {}
    count_by_source: Dict[str, int] = {}

    for y, source in zip(Y_all, sources):
        current_Y = np.vstack([current_Y, y.reshape(1, -1)])
        count_by_source[source] = count_by_source.get(source, 0) + 1
        if compute_hv:
            pf = compute_pareto_front(current_Y, maximize=maximize)
            new_hv = compute_hypervolume(current_Y[pf], reference_point)
            delta_hv_by_source[source] = (
                delta_hv_by_source.get(source, 0.0) + new_hv - current_hv
            )
            current_hv = new_hv

    pareto_mask = compute_pareto_front(Y_all, maximize=maximize)
    pareto_hits: Dict[str, int] = {}
    for on_pareto, source in zip(pareto_mask, sources):
        if on_pareto:
            pareto_hits[source] = pareto_hits.get(source, 0) + 1

    rows = []
    for source in count_by_source:
        total = count_by_source[source]
        hits = pareto_hits.get(source, 0)
        dhv = delta_hv_by_source.get(source, np.nan)
        rows.append({
            'acquisition_source': source,
            'n_points': total,
            'n_pareto_points': hits,
            'pareto_hit_rate': hits / total if total > 0 else 0.0,
            'total_delta_hv': dhv if compute_hv else np.nan,
            'delta_hv_fraction': (
                dhv / current_hv
                if (compute_hv and current_hv > 0) else np.nan
            ),
        })

    return pd.DataFrame(rows).sort_values(
        'pareto_hit_rate', ascending=False
    ).reset_index(drop=True)


def interpolate_simple_metric(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    value_col: str,
    experiment_labels: List[str] = None,
    gt_optimum_value: float = None,
    normalize_by_initial: bool = True,
    regret: bool = False,
    percentiles: List[float] = [0.25, 0.5, 0.75, 1.0],
) -> Tuple[Dict, pd.DataFrame]:
    """Interpolates a metric over cumulative samples for multiple experiments.

    Trajectories are interpolated per repetition onto a common sample grid before
    averaging, so that different batch sizes (and hence different numbers of AL
    cycles) are compared on a fair common axis of screened points.

    Normalization is applied per repetition by dividing by the value at t=0
    (the first trained-model evaluation), consistent with the normalized
    convergence scores defined in the SI. This must be done before averaging
    to avoid bias from repetitions with favorable or unfavorable initializations.

    Args:
        experiments: Mapping from experiment name to (points_df, metrics_df).
        value_col: Column name of the metric to interpolate in metrics_df.
        experiment_labels: Optional list of labels to rename experiment keys.
        gt_optimum_value: Reference optimum used to compute regret when
            regret=True. Must be provided if regret=True.
        normalize_by_initial: If True, divide each repetition's trajectory by
            its value at t=0 (per-repetition normalization). Default True.
        regret: If True, compute r_t = gt_optimum_value - value before
            normalization.
        percentiles: Fractions of max_samples at which to snapshot metrics.

    Returns:
        interpolated_results: Dict keyed by experiment name with entries:
            - 'common_samples': shared sample grid (length max_samples)
            - 'interpolated_mean': mean of normalized interpolated trajectories
            - 'interpolated_std': std of normalized interpolated trajectories
            - 'raw_mean': mean of raw (non-interpolated) trajectories per cycle
            - 'raw_std': std of raw trajectories per cycle
        percentiles_df: DataFrame with mean and std at each percentile checkpoint.
    """
    if experiment_labels is not None:
        experiments = dict(zip(experiment_labels, experiments.values()))

    if regret and gt_optimum_value is None:
        raise ValueError("gt_optimum_value must be provided when regret=True.")

    interpolated_results = {}
    percentiles_csv_rows = []

    for exp_name, (points_df, metrics_df) in experiments.items():
        all_curves_samples = []
        all_curves_values = []
        all_interp_values = []

        repetitions = metrics_df['repetition'].unique()

        for rep in repetitions:
            rep_metrics = metrics_df[metrics_df['repetition'] == rep].sort_values('cycle')
            rep_points = points_df[points_df['repetition'] == rep].sort_values('cycle')

            # Cumulative samples at each cycle, excluding cycle-0 initialization
            # so that cycle 0 maps to n=0 (first surrogate evaluation point)
            cumulative_samples = []
            for cycle in rep_metrics['cycle'].values:
                n_samples = (
                    len(rep_points[rep_points['cycle'] <= cycle])
                    - len(rep_points[rep_points['cycle'] == 0])
                )
                cumulative_samples.append(n_samples)
            cumulative_samples = np.array(cumulative_samples)

            values = rep_metrics[value_col].values.astype(float)

            # Compute regret before normalization
            if regret:
                values = gt_optimum_value - values

            # Per-repetition normalization by the initial value (t=0)
            # This corresponds to tilde_r_t = r_t / r_0^(s) in the SI notation
            if normalize_by_initial:
                v0 = values[0]
                if v0 == 0.0:
                    raise ValueError(
                        f"Initial value for repetition {rep} of experiment "
                        f"'{exp_name}' is zero; cannot normalize by initial value."
                    )
                values = values / v0

            all_curves_samples.append(cumulative_samples)
            all_curves_values.append(values)

        # Common grid: integer steps from 0 to max observed sample count
        max_samples = max(s[-1] for s in all_curves_samples)
        common_samples = np.arange(0, max_samples + 1, dtype=float)

        for samples, values in zip(all_curves_samples, all_curves_values):
            # np.interp: flat extrapolation at left boundary (below first sample)
            # is only safe if cycle 0 is present and maps to common_samples[0]=0.
            # Verified by the cumulative_samples construction above.
            interp_vals = np.interp(common_samples, samples, values)
            all_interp_values.append(interp_vals)

        all_interp_values = np.array(all_interp_values)  # (n_reps, max_samples+1)

        interpolated_results[exp_name] = {
            'common_samples': common_samples,
            'interpolated_mean': all_interp_values.mean(axis=0),
            'interpolated_std': all_interp_values.std(axis=0),
            # Raw (non-interpolated) mean/std per cycle — only valid within one
            # experiment where all repetitions share the same number of cycles
            'raw_mean': np.mean(all_curves_values, axis=0),
            'raw_std': np.std(all_curves_values, axis=0),
        }

        # Snapshot at requested percentile checkpoints
        for p in percentiles:
            target_sample = p * max_samples
            closest_idx = int(np.abs(common_samples - target_sample).argmin())
            percentiles_csv_rows.append({
                'experiment': exp_name,
                f'{value_col}_mean': all_interp_values[:, closest_idx].mean(),
                f'{value_col}_std': all_interp_values[:, closest_idx].std(),
                'sample_percentile': p,
                'samples_at_percentile': common_samples[closest_idx],
            })

    percentiles_df = pd.DataFrame(percentiles_csv_rows)
    return interpolated_results, percentiles_df

