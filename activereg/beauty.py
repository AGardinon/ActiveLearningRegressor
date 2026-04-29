#!

import ast
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# --- PLOT FUNC GENERAL

def plot_predicted_landscape(X_pool: np.ndarray, pred_array: np.ndarray, columns: int=3, save_path: Path=None):
    """Plot the predicted landscape.

    Args:
        X_pool (np.ndarray): The input features for the pool.
        pred_array (np.ndarray): The predicted values, must be 2D (N_cycles, N_samples).
        save_path (Path, optional): The path to save the plot. Defaults to None.
    """

    cycles, _ = pred_array.shape
    fig, ax = get_axes(cycles, columns)

    for i,pred in enumerate(pred_array):
        sc = ax[i].scatter(
            *X_pool.T, c=pred, cmap='coolwarm', s=5
        )
        ax[i].set_title(f'Cycle {i+1}')
        _ = fig.colorbar(sc, ax=ax[i])
        ax[i].set_aspect('equal')
        fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        
    return fig, ax

# ---------------------------------------------------------------------------
# --- PLOT FUNC BENCHMARKS


def plot_best_value_over_time(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    true_optimum: float,
    total_points: int = None,
    y_scaling_value: float = 1.0,
    value_col: str = 'y_best_screened',
    palette: str = 'colorblind',
    experiment_labels: List[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the best value over time.

    Args:
        experiments (Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]): The experimental data.
        true_optimum (float): The true optimum value.
        total_points (int, optional): The total number of points sampled. Defaults to None.
        y_scaling_value (float, optional): Scaling factor for the y-axis values. Defaults to 1.0.
        value_col (str, optional): The column name for the best value. Defaults to 'y_best_screened'.
        palette (str, optional): The color palette to use. Defaults to 'colorblind'.
        experiment_labels (List[str], optional): The labels for the experiments. Defaults to None.
        figsize (Tuple[int, int], optional): The figure size. Defaults to (10, 6).

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    colors = sns.color_palette(palette, n_colors=len(experiments))

    if experiment_labels is not None:
        experiments = dict(zip(experiment_labels, experiments.values()))

    for exp_name, (points_df, metrics_df) in experiments.items():
        # Group by repetition and get values per cycle
        all_curves = []
        
        for rep in metrics_df['repetition'].unique():
            rep_data = metrics_df[metrics_df['repetition'] == rep].sort_values('cycle')
            values = rep_data[value_col].values
            all_curves.append(values)
        
        # Convert to array (repetitions x cycles)
        all_curves = np.array(all_curves)
        all_curves *= y_scaling_value
        mean_curve = all_curves.mean(axis=0)
        std_curve = all_curves.std(axis=0)
        
        cycles = metrics_df[metrics_df['repetition'] == metrics_df['repetition'].unique()[0]]['cycle'].values
        # if total points is given, show behavior vs points per cycle
        if total_points is not None:
            points_per_cycle = total_points // len(cycles)
            cycles = cycles * points_per_cycle

        ax.plot(cycles, mean_curve, label=exp_name, linewidth=2, color=colors[len(ax.lines)])
        ax.fill_between(cycles, 
                        mean_curve - std_curve, 
                        mean_curve + std_curve, 
                        alpha=0.15, color=colors[len(ax.lines)-1], zorder=-1
                        )

    if true_optimum is not None:
        ax.axhline(y=true_optimum, color='.5', linestyle=':', label='Max.', zorder=-1)
    
    return fig, ax


def plot_simple_regret(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    true_optimum: float,
    total_points: int = None,
    y_scaling_value: float = 1.0,
    value_col: str = 'y_best_screened',
    log_scale: bool = False,
    palette: str = 'colorblind',
    experiment_labels: List[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the simple regret over time.

    Args:
        experiments (Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]): The experimental data.
        true_optimum (float): The true optimum value.
        total_points (int, optional): The total number of points sampled. Defaults to None.
        y_scaling_value (float, optional): Scaling factor for the y-axis values. Defaults to 1.0.
        value_col (str, optional): The column name for the best value. Defaults to 'y_best_screened'.
        log_scale (bool, optional): Whether to use a logarithmic scale for the y-axis. Defaults to False.
        palette (str, optional): The color palette to use. Defaults to 'colorblind'.
        experiment_labels (List[str], optional): The labels for the experiments. Defaults to None.
        figsize (Tuple[int, int], optional): The figure size. Defaults to (10, 6).

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    colors = sns.color_palette(palette, n_colors=len(experiments))

    if experiment_labels is not None:
        experiments = dict(zip(experiment_labels, experiments.values()))
    
    for exp_name, (points_df, metrics_df) in experiments.items():
        all_regrets = []
        
        for rep in metrics_df['repetition'].unique():
            rep_data = metrics_df[metrics_df['repetition'] == rep].sort_values('cycle')
            values = rep_data[value_col].values * y_scaling_value
            regret = true_optimum - values
            # Ensure non-negative and positive for log scale
            regret = np.maximum(regret, 1e-10)
            all_regrets.append(regret)
        
        all_regrets = np.array(all_regrets)
        mean_regret = all_regrets.mean(axis=0)
        std_regret = all_regrets.std(axis=0)
        
        cycles = metrics_df[metrics_df['repetition'] == metrics_df['repetition'].unique()[0]]['cycle'].values
        if total_points is not None:
            points_per_cycle = total_points // len(cycles)
            cycles = cycles * points_per_cycle
        
        if log_scale:
            ax.set_yscale('log')
        ax.plot(cycles, mean_regret, label=exp_name, linewidth=2, color=colors[len(ax.lines)])
        ax.fill_between(cycles, 
                        np.maximum(mean_regret - std_regret, 1e-10),
                        mean_regret + std_regret, 
                        alpha=0.15, color=colors[len(ax.lines)-1]
                        )

    return fig, ax


def plot_sample_efficiency(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], 
    value_col: str = 'y_best_screened',
    y_scaling_value: float = 1.0,
    palette: str = 'colorblind',
    experiment_labels: List[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot sample efficiency: Best Value vs Total Samples.

    Args:
        experiments (Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]): The experimental data.
        value_col (str, optional): The column name for the best value. Defaults to 'y_best_screened'.
        y_scaling_value (float, optional): Scaling factor for the y-axis values. Defaults to 1.0.
        palette (str, optional): The color palette to use. Defaults to 'colorblind'.
        experiment_labels (List[str], optional): The labels for the experiments. Defaults to None.
        figsize (Tuple[int, int], optional): The figure size. Defaults to (10, 6).

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    colors = sns.color_palette(palette, n_colors=len(experiments))

    if experiment_labels is not None:
        experiments = dict(zip(experiment_labels, experiments.values()))

    for exp_name, (points_df, metrics_df) in experiments.items():
        all_curves_samples = []
        all_curves_values = []
        
        for rep in metrics_df['repetition'].unique():
            rep_metrics = metrics_df[metrics_df['repetition'] == rep].sort_values('cycle')
            rep_points = points_df[points_df['repetition'] == rep].sort_values('cycle')
            
            # Count cumulative samples per cycle for only cycle >= 1
            cumulative_samples = []
            for cycle in rep_metrics['cycle'].values:
                n_samples = len(rep_points[rep_points['cycle'] <= cycle]) - len(rep_points[rep_points['cycle'] == 0])
                cumulative_samples.append(n_samples)

            values = rep_metrics[value_col].values * y_scaling_value

            all_curves_samples.append(cumulative_samples)
            all_curves_values.append(values)
        
        # Average across repetitions (need to interpolate to common sample counts)
        max_samples = max([max(s) for s in all_curves_samples])
        common_samples = np.linspace(0, max_samples, 50)
        
        interpolated_values = []
        for samples, values in zip(all_curves_samples, all_curves_values):
            interp_vals = np.interp(common_samples, samples, values)
            interpolated_values.append(interp_vals)
        
        interpolated_values = np.array(interpolated_values)
        mean_values = interpolated_values.mean(axis=0)
        std_values = interpolated_values.std(axis=0)
        
        ax.plot(common_samples, mean_values, label=exp_name, linewidth=2, color=colors[len(ax.lines)])
        ax.fill_between(common_samples, 
                        mean_values - std_values, 
                        mean_values + std_values, 
                        alpha=0.15, color=colors[len(ax.lines)-1])

    return fig, ax


def plot_time_to_threshold(
    threshold_df: pd.DataFrame, 
    thresholds: List[float] = [0.9, 0.95, 0.99],
    total_points_ref: int = None,
    plot_line: bool = True,
    palette: str = 'colorblind',
    experiment_labels: List[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot time to threshold as grouped bar chart.

    Args:
        threshold_df (pd.DataFrame): DataFrame containing threshold data.
        thresholds (List[float], optional): List of thresholds to plot. Defaults to [0.9, 0.95, 0.99].
        total_points_ref (int, optional): Reference value for total points. Defaults to None.
        plot_line (bool, optional): Whether to plot a line connecting the bars. Defaults to True.
        palette (str, optional): Color palette to use. Defaults to 'colorblind'.
        experiment_labels (List[str], optional): Labels for the experiments. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    colors = sns.color_palette(palette, n_colors=len(thresholds))
    
    x = np.arange(len(threshold_df))  # the label locations
    total_width = 0.8
    bar_width = total_width / len(thresholds)
    offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, len(thresholds))
    for i, thresh in enumerate(thresholds):
        means = threshold_df[f"{int(thresh*100)}%_mean"].values
        stds = threshold_df[f"{int(thresh*100)}%_std"].values

        ax.bar(x + offsets[i], means, width=bar_width, label=f"{int(thresh*100)}%", color=colors[i], yerr=stds, capsize=3., alpha=0.8)

        if plot_line:
            ax.scatter(x + offsets[i], means, color=colors[i], marker='o', edgecolor='black')
            ax.plot(x + offsets[i], means, color=colors[i], linestyle='--', linewidth=1, zorder=-1)

    if total_points_ref is not None:
        ax.set_ylim(0, total_points_ref)
        ax.set_ylabel('Points sampled')
        ax.set_title('Points to performance thresholds')
    else:
        ax.set_ylabel('Cycles')
        ax.set_title('Cycles to performance thresholds')

    if experiment_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(experiment_labels, ha='center')
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(threshold_df['Experiment'], ha='center')

    return fig, ax


def plot_batch_diversity_over_time(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], 
    feature_cols: List[str] = ['x1', 'x2', 'x3'],
    n_points: int = 100,
    experiment_labels: List[str] = None,
    palette: str = 'colorblind',
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot batch diversity over time.

    Args:
        experiments (Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]): Dictionary of experiments with their data.
        feature_cols (List[str], optional): List of feature columns to use for diversity calculation. Defaults to ['x1', 'x2', 'x3'].
        n_points (int, optional): Number of points to interpolate for the x-axis. Defaults to 100.
        experiment_labels (List[str], optional): Labels for the experiments. Defaults to None.
        palette (str, optional): Color palette to use. Defaults to 'colorblind'.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes objects.
    """
    from scipy.spatial.distance import pdist
    from scipy.interpolate import interp1d

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    colors = sns.color_palette(palette, n_colors=len(experiments))
    if experiment_labels is not None:
        experiments = dict(zip(experiment_labels, experiments.values()))
    
    # Common x-axis: 0% to 100% of experiment completion
    normalized_x = np.linspace(0, 100, n_points)
    
    for exp_name, (points_df, metrics_df) in experiments.items():
        all_diversity = []
        
        for rep in points_df['repetition'].unique():
            rep_points = points_df[points_df['repetition'] == rep].sort_values('cycle')
            
            # Get diversity for each cycle
            cycles = sorted(rep_points['cycle'].unique())
            diversity_per_cycle = []
            
            for cycle in cycles:
                # Get points from THIS cycle only (not cumulative)
                cycle_batch = rep_points[rep_points['cycle'] == cycle]
                X_batch = cycle_batch[feature_cols].values
                
                if len(X_batch) > 1:
                    diversity = pdist(X_batch).mean()
                    diversity_per_cycle.append(diversity)
                else:
                    diversity_per_cycle.append(0)
            
            # Normalize cycle numbers to percentage (0-100%)
            if len(cycles) > 1:
                cycle_percentage = np.linspace(0, 100, len(cycles))
                
                # Interpolate to common x-axis
                if len(diversity_per_cycle) > 1:
                    interp_func = interp1d(cycle_percentage, diversity_per_cycle, 
                                          kind='linear', 
                                          bounds_error=False, 
                                          fill_value='extrapolate'
                                          )
                    diversity_interpolated = interp_func(normalized_x)
                    all_diversity.append(diversity_interpolated)
            elif len(cycles) == 1:
                # Edge case: only one cycle
                all_diversity.append(np.full(n_points, diversity_per_cycle[0]))
        
        if len(all_diversity) == 0:
            continue
            
        # Convert to array and compute statistics
        all_diversity = np.array(all_diversity)
        mean_div = all_diversity.mean(axis=0)
        std_div = all_diversity.std(axis=0)

        ax.plot(normalized_x, mean_div, label=exp_name, linewidth=2, color=colors[len(ax.lines)])
        ax.fill_between(normalized_x,
                        mean_div - std_div,
                        mean_div + std_div,
                        alpha=0.15, color=colors[len(ax.lines)-1]
                        )

    return fig, ax


ACQFUNC_ACRONYMS = {
    'exploration_mutual_info' : 'MI',
    'uncertainty_landscape' : 'UL',
    'upper_confidence_bound' : 'UCB',
    'expected_improvement' : 'EI',
    'target_expected_improvement' : 'TEI',
    'percentage_target_expected_improvement' : '%TEI',
    'random' : 'RND'
}


def analyze_acquisition_source_distribution(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], 
    experiment_labels: List[str] = None,
    acquisition_function_acronyms: Dict[str, str] = ACQFUNC_ACRONYMS,
    palette: str = 'colorblind',
    n_ticks: int = 3,
    subplotsize: Tuple[float, float] = (3., 3.),
    n_subplots_cols: int = 3
) -> Tuple[plt.Figure, plt.Axes]:
    """Analyze the distribution of acquisition sources over time.

    Args:
        experiments (Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]): Dictionary of experiments with their data.
        experiment_labels (List[str], optional): Labels for the experiments. Defaults to None.
        acquisition_function_acronyms (Dict[str, str], optional): Mapping of acquisition function names to acronyms. Defaults to ACQFUNC_ACRONYMS.
        palette (str, optional): Color palette to use. Defaults to 'colorblind'.
        n_ticks (int, optional): Number of ticks for the x-axis. Defaults to 3.
        figsize (Tuple[float, float], optional): Figure size. Defaults to (3.3, 3).

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes objects.
    """
    # defined a dict of acronyms for the acquisition sources
    source_acronyms = acquisition_function_acronyms.copy()

    # Assign to each source acronym a unique color based on the palette
    palette_colors = sns.color_palette(palette, n_colors=len(source_acronyms))
    source_colors = dict(zip(source_acronyms.values(), palette_colors))

    fig, axes = get_axes(
        len(experiments), 
        len(experiments) if len(experiments) < n_subplots_cols else n_subplots_cols, 
        fig_frame=subplotsize, res=300)
    if len(experiments) == 1:
        axes = [axes]

    if experiment_labels is not None:
        experiments = dict(zip(experiment_labels, experiments.values()))

    # Substitue acquisition source names with acronyms
    for exp_name, (points_df, metrics_df) in experiments.items():
        points_df['acquisition_source'] = points_df['acquisition_source'].map(
            lambda x: source_acronyms.get(x, x)
        )
    
    for idx, (exp_name, (points_df, metrics_df)) in enumerate(experiments.items()):
        ax = axes[idx]
        
        # Get all unique cycles and sources
        cycles = sorted(points_df['cycle'].unique())
        sources = sorted(points_df['acquisition_source'].unique())
        
        # Store percentages for each repetition
        all_reps_percentages = {source: [] for source in sources}
        
        for rep in points_df['repetition'].unique():
            rep_points = points_df[points_df['repetition'] == rep]
            
            # For this repetition, get percentage of each source per cycle
            rep_percentages = {source: [] for source in sources}
            
            for cycle in cycles:
                cycle_data = rep_points[rep_points['cycle'] == cycle]
                total = len(cycle_data)
                
                if total > 0:
                    for source in sources:
                        count = len(cycle_data[cycle_data['acquisition_source'] == source])
                        rep_percentages[source].append(count / total * 100)
                else:
                    # No data for this cycle in this repetition
                    for source in sources:
                        rep_percentages[source].append(0)
            
            # Collect this repetition's data
            for source in sources:
                all_reps_percentages[source].append(rep_percentages[source])
        
        # Average across repetitions
        mean_percentages = {}
        for source in sources:
            # Convert to array: (n_reps, n_cycles)
            source_array = np.array(all_reps_percentages[source])
            mean_percentages[source] = source_array.mean(axis=0)
        
        # Create stacked area plot
        bottom = np.zeros(len(cycles))
        # Get colors for sources
        colors = [source_colors[source] for source in sources]
        
        for source, color in zip(sources, colors):
            ax.fill_between(cycles, bottom, bottom + mean_percentages[source], 
                           label=source, alpha=1., color=color)
            bottom += mean_percentages[source]
        
        # Customize ticks and labels
        tick_indices = np.linspace(0, len(cycles) - 1, n_ticks, dtype=int)
        tick_values = [cycles[i] for i in tick_indices]
        ax.set_xticks(tick_values)
        ax.set_xticklabels([str(int(c)) for c in tick_values])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 100])
    
    # Hide unused subplots
    for idx in range(len(experiments), len(axes)):
        axes[idx].axis('off')
    
    return fig, axes


def plot_model_metrics_over_time(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], 
    metrics: List[str] = ['rmse_vs_gt_val', 'mae_vs_gt_val', 'nll_val'],
    total_points: int = None,
    palette: str = 'colorblind',
    experiment_labels: List[str] = None,
    subplotsize: Tuple[int, int] = (7, 4),
    n_subplots_cols: int = 2
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot model metrics over time.

    Args:
        experiments (Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]): Dictionary of experiments with their data.
        metrics (List[str], optional): List of metrics to plot. Defaults to ['rmse_vs_gt_val', 'mae_vs_gt_val', 'nll_val'].
        total_points (int, optional): Total number of points to plot. Defaults to None.
        palette (str, optional): Color palette to use. Defaults to 'colorblind'.
        experiment_labels (List[str], optional): Labels for the experiments. Defaults to None.
        subplotsize (Tuple[int, int], optional): Figure size. Defaults to (7, 4).
        n_subplots_cols (int, optional): Number of subplot columns. Defaults to 2.

    Returns:
        Tuple[plt.Figure, List[plt.Axes]]: The figure and axes objects.
    """
    fig, axes = get_axes(
        len(metrics), 
        len(metrics) if len(metrics) < n_subplots_cols else n_subplots_cols, 
        fig_frame=subplotsize, res=300)
    if len(metrics) == 1:
        axes = [axes]
    
    colors = sns.color_palette(palette, n_colors=len(experiments))

    if experiment_labels is not None:
        experiments = dict(zip(experiment_labels, experiments.values()))

    for ax, metric in zip(axes, metrics):
        for exp_name, (points_df, metrics_df) in experiments.items():
            all_curves = []
            
            for rep in metrics_df['repetition'].unique():
                rep_data = metrics_df[metrics_df['repetition'] == rep].sort_values('cycle')
                values = rep_data[metric].values
                all_curves.append(values)
            
            all_curves = np.array(all_curves)
            mean_curve = all_curves.mean(axis=0)
            std_curve = all_curves.std(axis=0)
            
            if total_points is not None:
                points_per_cycle = total_points // len(metrics_df['cycle'].unique())
                cycles = metrics_df[metrics_df['repetition'] == metrics_df['repetition'].unique()[0]]['cycle'].values * points_per_cycle
            else:
                cycles = metrics_df[metrics_df['repetition'] == metrics_df['repetition'].unique()[0]]['cycle'].values
            
            ax.plot(cycles, mean_curve, label=exp_name, linewidth=2, color=colors[len(ax.lines)])
            ax.fill_between(cycles, 
                            mean_curve - std_curve, 
                            mean_curve + std_curve, 
                            alpha=0.15, color=colors[len(ax.lines)-1]
                            )
        
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


def plot_metric_at_milestones(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    metric: str = 'rmse_vs_gt_val',
    screened_points: List[int] = [25, 50, 100],
    palette: str = 'colorblind',
    figsize: Tuple[int, int] = (12, 6),
    experiment_labels: List[str] = None,
    box_width: float = 1.2,
    group_spacing: float = 1.5,
    ylabel: str = None,
    patched_xticks: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a metric at specific milestones across experiments.

    Args:
        experiments (Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]): The experimental data.
        metric (str, optional): The metric to plot. Defaults to 'rmse_vs_gt_val'.
        screened_points (List[int], optional): The milestones to evaluate. Defaults to [25, 50, 100].
        palette (str, optional): The color palette to use. Defaults to 'colorblind'.
        figsize (Tuple[int, int], optional): The figure size. Defaults to (12, 6).
        experiment_labels (List[str], optional): Labels for the experiments. Defaults to None.
        box_width (float, optional): The width of the boxes in the plot. Defaults to 1.2.
        group_spacing (float, optional): The spacing between groups of boxes. Defaults to 1.5.
        ylabel (str, optional): The label for the y-axis. Defaults to None.
        patched_xticks (bool, optional): Whether to patch the x-ticks. Defaults to True.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes objects.
    """
    from matplotlib.patches import Rectangle
    
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    if experiment_labels is not None:
        experiments = dict(zip(experiment_labels, experiments.values()))

    exp_names = list(experiments.keys())
    n_experiments = len(exp_names)
    n_milestones = len(screened_points)
    
    # Colors for different experiments
    colors = sns.color_palette(palette, n_colors=len(experiments))
    
    # Collect data for boxplots
    # Structure: list of lists, where each inner list is data for one box
    all_data = []
    positions = []
    box_colors = []
    
    # Width parameters for visual clarity
    box_width = box_width / n_experiments  # Width of each box
    group_spacing = group_spacing  # Space between milestone groups

    for milestone_idx, n_samples in enumerate(screened_points):
        for exp_idx, exp_name in enumerate(exp_names):
            points_df, metrics_df = experiments[exp_name]
            
            # Collect metric values at this milestone for each repetition
            milestone_values = []
            
            for rep in metrics_df['repetition'].unique():
                rep_points = points_df[points_df['repetition'] == rep].sort_values('cycle')
                rep_metrics = metrics_df[metrics_df['repetition'] == rep].sort_values('cycle')
                
                # Find the cycle where cumulative samples >= n_samples
                cumulative_samples = 0
                target_cycle = None
                
                for cycle in sorted(rep_points['cycle'].unique()):
                    cycle_count = len(rep_points[rep_points['cycle'] <= cycle])
                    if cycle_count >= n_samples:
                        target_cycle = cycle
                        break
                
                if target_cycle is not None:
                    # Get metric value at this cycle
                    cycle_metric = rep_metrics[rep_metrics['cycle'] == target_cycle]
                    if len(cycle_metric) > 0:
                        milestone_values.append(cycle_metric[metric].iloc[0])
            
            # Only add if we have data
            if len(milestone_values) > 0:
                all_data.append(milestone_values)
                
                # Calculate position for this box
                # Center of milestone group + offset for this experiment
                group_center = milestone_idx * group_spacing
                exp_offset = (exp_idx - (n_experiments - 1) / 2) * box_width
                positions.append(group_center + exp_offset)
                box_colors.append(colors[exp_idx])
    
    # Create boxplots
    bp = ax.boxplot(all_data, 
                    positions=positions, 
                    widths=box_width * 0.9,
                    patch_artist=True,
                    showfliers=True,
                    boxprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=1.8, color='darkred', linestyle='-'),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='black', 
                                    markersize=4, alpha=0.5))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add subtle background shading for each milestone group
    if patched_xticks:
        for milestone_idx in range(n_milestones):
            group_center = milestone_idx * group_spacing
            if milestone_idx % 2 == 0:  # Alternate shading
                rect = Rectangle((group_center - group_spacing/2 + 0.1, ax.get_ylim()[0]),
                                group_spacing - 0.2, ax.get_ylim()[1] - ax.get_ylim()[0],
                                facecolor='gray', alpha=0.05, zorder=-1)
                ax.add_patch(rect)
    
    # Set x-axis ticks and labels
    milestone_positions = [i * group_spacing for i in range(n_milestones)]
    ax.set_xticks(milestone_positions)
    ax.set_xticklabels([f'{n}' for n in screened_points], fontsize=10)
    
    # Labels and title
    ax.set_xlabel('Cumulative Samples Screened', fontsize=10)
    ax.set_ylabel(ylabel or metric.replace('_', ' ').upper(), fontsize=10)
    # ax.set_title(title or f'{metric.replace("_", " ").title()} at Screening Milestones', fontsize=10)

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], alpha=0.7, label=exp_names[i]) 
                      for i in range(n_experiments)]
    ax.legend(handles=legend_elements, loc='best', fontsize=8, framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig, ax


def plot_metric_at_milestones_with_stats(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    metric: str = 'rmse_vs_gt_val',
    screened_points: List[int] = [25, 50, 100],
    palette: str = 'colorblind',
    figsize: Tuple[int, int] = (12, 6),
    experiment_labels: List[str] = None,
    box_width: float = 1.2,
    group_spacing: float = 1.5,
    ylabel: str = None,
    patched_xticks: bool = True,
    show_pvalues: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot boxplots with optional statistical significance annotations.
    Requires scipy for statistical tests.
    """
    from scipy import stats
    
    # First create the base plot
    fig, ax = plot_metric_at_milestones(
        experiments=experiments, metric=metric, screened_points=screened_points, 
        palette=palette, figsize=figsize, experiment_labels=experiment_labels,
        box_width=box_width, group_spacing=group_spacing, ylabel=ylabel,
        patched_xticks=patched_xticks)

    if show_pvalues and len(experiments) == 2:
        # Only show p-values for pairwise comparison
        exp_names = list(experiments.keys())
        
        for milestone_idx, n_samples in enumerate(screened_points):
            # Collect data for both experiments at this milestone
            data_exp1 = []
            data_exp2 = []
            
            for exp_idx, exp_name in enumerate(exp_names):
                points_df, metrics_df = experiments[exp_name]
                milestone_values = []
                
                for rep in metrics_df['repetition'].unique():
                    rep_points = points_df[points_df['repetition'] == rep].sort_values('cycle')
                    rep_metrics = metrics_df[metrics_df['repetition'] == rep].sort_values('cycle')
                    
                    cumulative_samples = 0
                    target_cycle = None
                    
                    for cycle in sorted(rep_points['cycle'].unique()):
                        cycle_count = len(rep_points[rep_points['cycle'] <= cycle])
                        if cycle_count >= n_samples:
                            target_cycle = cycle
                            break
                    
                    if target_cycle is not None:
                        cycle_metric = rep_metrics[rep_metrics['cycle'] == target_cycle]
                        if len(cycle_metric) > 0:
                            milestone_values.append(cycle_metric[metric].iloc[0])
                
                if exp_idx == 0:
                    data_exp1 = milestone_values
                else:
                    data_exp2 = milestone_values
            
            # Perform t-test if both have data
            if len(data_exp1) > 0 and len(data_exp2) > 0:
                t_stat, p_value = stats.ttest_ind(data_exp1, data_exp2)
                
                # Add significance annotation
                group_center = milestone_idx * group_spacing
                y_max = ax.get_ylim()[1]
                y_pos = y_max * 0.95
                
                # Significance stars
                if p_value < 0.001:
                    sig_text = '***'
                elif p_value < 0.01:
                    sig_text = '**'
                elif p_value < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'ns'
                
                ax.text(group_center, y_pos, sig_text, 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    return fig, ax

# ---------------------------------------------------------------------------
# --- PLOT UTILITIES

def compute_time_to_threshold(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], 
    thresholds: List[float] = [0.9, 0.95, 0.99],
    true_optimum: float = None,
    total_points: int = None,
    value_col: str = 'y_best_screened',
    experiment_labels: List[str] = None,
) -> pd.DataFrame:
    """
    Compute how many cycles needed to reach X% of optimum.
    
    Returns:
        DataFrame with experiments as rows and thresholds as columns
    """
    results = []

    if experiment_labels is not None:
        experiments = dict(zip(experiment_labels, experiments.values()))

    for exp_name, (points_df, metrics_df) in experiments.items():
        times = {f"{int(t*100)}%": [] for t in thresholds}
        
        for rep in metrics_df['repetition'].unique():
            rep_data = metrics_df[metrics_df['repetition'] == rep].sort_values('cycle')
            values = rep_data[value_col].values
            
            if true_optimum is not None:
                percentage = values / true_optimum
            else:
                # Use final value as reference
                percentage = values / values[-1]
            
            for thresh in thresholds:
                idx = np.where(percentage >= thresh)[0]
                if len(idx) > 0:
                    if total_points is not None:
                        points_per_cycle = total_points // len(rep_data['cycle'].values)
                        times[f"{int(thresh*100)}%"].append(rep_data['cycle'].iloc[idx[0]] * points_per_cycle)
                    else:
                        times[f"{int(thresh*100)}%"].append(rep_data['cycle'].iloc[idx[0]])
                else:
                    times[f"{int(thresh*100)}%"].append(np.nan)  # Never reached
        
        row = {'Experiment': exp_name}
        for k, v in times.items():
            valid_times = [t for t in v if not np.isnan(t)]
            if valid_times:
                row[f'{k}_mean'] = np.mean(valid_times)
                row[f'{k}_std'] = np.std(valid_times)
                row[f'{k}_success_rate'] = len(valid_times) / len(v)
            else:
                row[f'{k}_mean'] = np.nan
                row[f'{k}_std'] = np.nan
                row[f'{k}_success_rate'] = 0.0
        
        results.append(row)

    return pd.DataFrame(results)


def get_alphas(Z: np.ndarray, scale: bool=False, treshold: float=.50001) -> np.ndarray:
    alphas = (Z.ravel() - Z.ravel().min()) / (Z.ravel().max() - Z.ravel().min())
    if scale:
        for i,av in enumerate(alphas):
            if av <= treshold:
                alphas[i] = 0.
            else:
                pass
    return alphas


def get_axes(plots: int, 
             max_col: int =2, 
             fig_frame: tuple =(3.3,3.), 
             res: int =200):
    """Define Fig and Axes objects.
    """
    # cols and rows definitions
    cols = plots if plots <= max_col else max_col
    rows = int(plots / max_col) + int(plots % max_col != 0)

    fig, axes = plt.subplots(rows,
                             cols,
                             figsize=(cols * fig_frame[0], rows * fig_frame[1]),
                             dpi=res)
    if plots > 1:
        axes = axes.flatten()
        for i in range(plots, max_col*rows):
            remove_frame(axes[i])
    elif plots == 1:
        pass
    
    return fig, axes


def remove_frame(axes) -> None:
    for side in ['bottom', 'right', 'top', 'left']:
        axes.spines[side].set_visible(False)
    axes.set_yticks([])
    axes.set_xticks([])
    axes.xaxis.set_ticks_position('none')
    axes.yaxis.set_ticks_position('none')
    pass


def set_identical_axes(axes) -> None:
    axes.set_xlim(min(axes.get_xlim()[0], axes.get_ylim()[0]),
                  max(axes.get_xlim()[1], axes.get_ylim()[1]))
    axes.set_ylim(axes.get_xlim())

    # Set identical ticks
    ticks = axes.get_xticks()
    axes.set_yticks(ticks)

# ---------------------------------------------------------------------------
# --- PLOT FUNC MULTI-PROPERTY


def plot_objective_space(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    target_names: List[str],
    pool_df: Optional[pd.DataFrame] = None,
    color_by: str = "acquisition_source",
    filter_acquisitions: Optional[List[str]] = None,
    maximize: bool = True,
    palette: str = "colorblind",
    figsize: Tuple[float, float] = (6.0, 5.0),
    column_number: int|None = None,
    experiment_labels: Optional[List[str]] = None,
    acronym_map: Optional[Dict[str, str]] = ACQFUNC_ACRONYMS,
) -> Tuple[plt.Figure, list]:
    """Scatter sampled points in objective space.

    For P=2: a single (y1, y2) scatter per experiment.
    For P>2: a pairwise grid (not yet implemented — raises NotImplementedError).

    Pool points are shown as a gray background when pool_df is provided.
    The true Pareto front of the pool and the discovered Pareto front of the
    sampled points are both highlighted.

    Args:
        experiments: Dict mapping name to (train_points_data df, benchmark_data df).
        target_names: Objective column names, e.g. ['y1', 'y2'].
        pool_df: Optional pool dataset with the same target columns. When provided
                 the full pool is shown as a gray background and its Pareto front
                 as a dashed reference. For runs with a non-null gt_file this CSV
                 does not exist and pool_df should be left as None.
        color_by: How to color sampled points. One of:
                  'acquisition_source' (categorical, consistent across experiments),
                  'cycle' (continuous cycle index),
                  'cumulative_index' (continuous discovery order).
        filter_acquisitions: If set, restrict the displayed (and Pareto-computed)
                             points to these acquisition source names.
        maximize: True (default) when higher objective values are better.
        palette: Seaborn palette for acquisition-source coloring.
        figsize: Size of a single subplot panel.
        experiment_labels: Optional rename list for experiment keys.

    Returns:
        (fig, axes) where axes is a list with one Axes per experiment.
    """
    from activereg.metrics import compute_pareto_front

    P = len(target_names)
    if P != 2:
        raise NotImplementedError(
            "plot_objective_space supports P=2 only. "
            "Pairwise grid for P>2 is not yet implemented."
        )

    if experiment_labels is not None:
        experiments = dict(zip(experiment_labels, experiments.values()))

    n_exp = len(experiments)
    fig, axes = get_axes(n_exp, n_exp if column_number is None else column_number, fig_frame=figsize, res=300)
    if n_exp == 1:
        axes = [axes]

    # Build consistent source-to-color mapping across all experiments
    all_sources: List[str] = []
    for pts, _ in experiments.values():
        for s in pts["acquisition_source"].unique():
            if s not in all_sources:
                all_sources.append(s)
    source_palette = sns.color_palette(palette, n_colors=len(all_sources))
    source_color_map = dict(zip(all_sources, source_palette))

    for ax, (exp_name, (points_df, _)) in zip(axes, experiments.items()):
        # ── pool background ──────────────────────────────────────────────
        if pool_df is not None:
            ax.scatter(
                pool_df[target_names[0]], pool_df[target_names[1]],
                c="lightgray", s=4, alpha=0.35, zorder=0, rasterized=True,
            )
            Y_pool = pool_df[target_names].to_numpy(dtype=float)
            pf_pool = Y_pool[compute_pareto_front(Y_pool, maximize=maximize)]
            pf_pool = pf_pool[np.argsort(pf_pool[:, 0])]
            ax.plot(
                pf_pool[:, 0], pf_pool[:, 1],
                color="gray", linewidth=1.5, linestyle="--", zorder=1,
                label="True PF",
            )

        # ── sampled points ───────────────────────────────────────────────
        df = points_df.copy()
        if filter_acquisitions is not None:
            df = df[df["acquisition_source"].isin(filter_acquisitions)]

        Y = df[target_names].to_numpy(dtype=float)

        if color_by == "acquisition_source":
            for source in df["acquisition_source"].unique():
                mask = df["acquisition_source"] == source
                label = acronym_map.get(source, source) if acronym_map else source
                ax.scatter(
                    df.loc[mask, target_names[0]].values,
                    df.loc[mask, target_names[1]].values,
                    c=[source_color_map[source]], s=25, alpha=0.8, zorder=2,
                    label=label,
                )
        else:
            color_vals = (
                np.arange(len(df)) if color_by == "cumulative_index"
                else df["cycle"].values
            )
            sc = ax.scatter(
                df[target_names[0]].values, df[target_names[1]].values,
                c=color_vals, cmap="viridis", s=25, alpha=0.8, zorder=2,
            )
            label = "Sample index" if color_by == "cumulative_index" else "Cycle"
            fig.colorbar(sc, ax=ax, label=label, shrink=0.8)

        # ── sampled Pareto front ─────────────────────────────────────────
        if len(Y) > 0:
            pf_mask = compute_pareto_front(Y, maximize=maximize)
            pf_pts = Y[pf_mask]
            pf_pts = pf_pts[np.argsort(pf_pts[:, 0])]
            ax.plot(
                pf_pts[:, 0], pf_pts[:, 1],
                color="crimson", linewidth=2.0, marker="o", markersize=5,
                zorder=3, label="Sampled PF",
            )

        ax.set_xlabel(target_names[0])
        ax.set_ylabel(target_names[1])
        ax.set_title(exp_name)
        ax.legend(loc="best", fontsize=7, framealpha=0.8)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig, axes


def plot_hypervolume_over_time(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    target_names: List[str],
    reference_point: np.ndarray,
    x_axis: str = "samples",
    filter_acquisitions: Optional[List[str]] = None,
    exclude_init: bool = False,
    maximize: bool = True,
    palette: str = "colorblind",
    figsize: Tuple[float, float] = (9.0, 5.0),
    experiment_labels: Optional[List[str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot hypervolume indicator growth over time.

    Computes the running hypervolume of the discovered Pareto front after each
    sampled point (chronological order). Averages across repetitions.

    Supports P=2 only (raises NotImplementedError otherwise).

    Args:
        experiments: Dict mapping name to (train_points_data df, benchmark_data df).
        target_names: Objective column names.
        reference_point: Array of shape (P,) dominated by all Pareto points.
                         Tip: use (min_y1 - ε, min_y2 - ε) from the pool.
        x_axis: 'samples' (default, cumulative points — fair across batch sizes)
                or 'cycles'.
        filter_acquisitions: Restrict the Pareto front to these source names only.
        maximize: True (default) when higher objective values are better.
        palette: Seaborn palette for experiment lines.
        figsize: Figure size.
        experiment_labels: Optional rename list for experiment keys.

    Returns:
        (fig, ax)
    """
    from activereg.metrics import compute_pareto_front, compute_hypervolume

    P = len(target_names)
    if P != 2:
        raise NotImplementedError(
            "plot_hypervolume_over_time supports P=2 only."
        )

    reference_point = np.asarray(reference_point, dtype=float)
    if experiment_labels is not None:
        experiments = dict(zip(experiment_labels, experiments.values()))

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    colors = sns.color_palette(palette, n_colors=len(experiments))

    for color, (exp_name, (points_df, _)) in zip(colors, experiments.items()):
        all_hv_curves: List[np.ndarray] = []
        max_len = 0

        for rep in sorted(points_df["repetition"].unique()):
            df = points_df[points_df["repetition"] == rep].sort_values("cycle")
            if exclude_init:
                df = df[df["cycle"] > 0]
            if filter_acquisitions is not None:
                df = df[df["acquisition_source"].isin(filter_acquisitions)]
            df = df.reset_index(drop=True)

            Y_seq = df[target_names].to_numpy(dtype=float)
            current_Y = np.empty((0, P), dtype=float)
            current_hv = 0.0
            hv_curve: List[float] = []

            for y in Y_seq:
                current_Y = np.vstack([current_Y, y.reshape(1, -1)])
                pf = compute_pareto_front(current_Y, maximize=maximize)
                current_hv = compute_hypervolume(current_Y[pf], reference_point)
                hv_curve.append(current_hv)

            all_hv_curves.append(np.array(hv_curve))
            max_len = max(max_len, len(hv_curve))

        # Interpolate repetitions onto a common sample grid
        common_x = np.arange(1, max_len + 1)
        interp_curves = np.array([
            np.interp(common_x, np.arange(1, len(c) + 1), c)
            for c in all_hv_curves
        ])
        mean_hv = interp_curves.mean(axis=0)
        std_hv = interp_curves.std(axis=0)

        if x_axis == "cycles":
            # Aggregate per cycle: take the last HV value within each cycle
            # Use the first repetition's cycle labels as reference
            ref_df = points_df[
                points_df["repetition"] == sorted(points_df["repetition"].unique())[0]
            ].sort_values("cycle")
            if exclude_init:
                ref_df = ref_df[ref_df["cycle"] > 0]
            if filter_acquisitions is not None:
                ref_df = ref_df[ref_df["acquisition_source"].isin(filter_acquisitions)]
            cycle_ends = (
                ref_df.reset_index(drop=True)
                .groupby("cycle")
                .apply(lambda g: g.index[-1])
                .values
            )
            x_plot = ref_df["cycle"].unique()
            x_plot.sort()
            mean_hv = mean_hv[cycle_ends]
            std_hv = std_hv[cycle_ends]
        else:
            x_plot = common_x

        ax.plot(x_plot, mean_hv, label=exp_name, linewidth=2, color=color)
        ax.fill_between(
            x_plot, mean_hv - std_hv, mean_hv + std_hv,
            alpha=0.15, color=color, zorder=-1,
        )

    ax.set_xlabel("Cumulative samples" if x_axis == "samples" else "Cycle")
    ax.set_ylabel("Hypervolume indicator")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_per_property_best_over_time(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    target_names: List[str],
    total_points: Optional[int] = None,
    ceiling_values: Optional[Dict[str, float]] = None,
    palette: str = "colorblind",
    subplotsize: Tuple[float, float] = (6.0, 4.0),
    column_number: int|None = None,
    experiment_labels: Optional[List[str]] = None,
) -> Tuple[plt.Figure, list]:
    """Plot y_best for each property over cycles, one panel per property.

    Reads the y_best_{name} columns from benchmark_data.csv. Analogous to
    plot_best_value_over_time but with one subplot per objective.

    Args:
        experiments: Dict mapping name to (train_points_data df, benchmark_data df).
        target_names: Objective column names, e.g. ['y1', 'y2'].
        total_points: If provided, x-axis shows cumulative points instead of cycles.
        palette: Seaborn palette for experiment lines.
        subplotsize: Size of each per-property panel.
        experiment_labels: Optional rename list for experiment keys.

    Returns:
        (fig, axes) where axes is a list with one Axes per property.
    """
    if experiment_labels is not None:
        experiments = dict(zip(experiment_labels, experiments.values()))

    P = len(target_names)
    fig, axes = get_axes(P, P if column_number is None else column_number, fig_frame=subplotsize, res=300)
    if P == 1:
        axes = [axes]

    colors = sns.color_palette(palette, n_colors=len(experiments))

    for ax, prop in zip(axes, target_names):
        col = f"y_best_{prop}"
        for color, (exp_name, (_, metrics_df)) in zip(colors, experiments.items()):
            all_curves: List[np.ndarray] = []
            for rep in sorted(metrics_df["repetition"].unique()):
                rep_data = metrics_df[metrics_df["repetition"] == rep].sort_values("cycle")
                all_curves.append(rep_data[col].values)

            arr = np.array(all_curves)
            mean_c = arr.mean(axis=0)
            std_c = arr.std(axis=0)

            ref_rep = sorted(metrics_df["repetition"].unique())[0]
            cycles = (
                metrics_df[metrics_df["repetition"] == ref_rep]
                .sort_values("cycle")["cycle"].values
            )
            x = cycles * (total_points // len(cycles)) if total_points else cycles

            ax.plot(x, mean_c, label=exp_name, linewidth=2, color=color)
            ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                            alpha=0.15, color=color, zorder=-1)

        if ceiling_values is not None and prop in ceiling_values:
            ax.axhline(
                ceiling_values[prop], color=".3", linestyle="--",
                linewidth=1.2, label="Pool max",
            )

        ax.set_title(f"y_best: {prop}")
        ax.set_xlabel("Cumulative samples" if total_points else "Cycle")
        ax.set_ylabel(f"Best {prop}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


def plot_weight_distribution(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    joint_entry_name: str,
    palette: str = "colorblind",
    figsize: Tuple[float, float] = (5.0, 5.0),
    column_number: int|None = None,
    experiment_labels: Optional[List[str]] = None,
) -> Tuple[plt.Figure, list]:
    """Scatter the Dirichlet-sampled weights from a joint acquisition entry.

    Reads the resolved_weights_{joint_entry_name} column from benchmark_data.csv
    and plots each (w1, w2, ...) sample. For P=2 this is a 2D simplex scatter
    that visually confirms weight diversity. For P>2 a per-weight histogram is
    shown instead.

    Args:
        experiments: Dict mapping name to (train_points_data df, benchmark_data df).
        joint_entry_name: Name of the joint acquisition entry (e.g. 'parego_joint').
        palette: Seaborn palette for experiment points.
        figsize: Size of each subplot panel.
        experiment_labels: Optional rename list for experiment keys.

    Returns:
        (fig, axes) where axes is a list with one Axes per experiment.
    """
    if experiment_labels is not None:
        experiments = dict(zip(experiment_labels, experiments.values()))

    col = f"resolved_weights_{joint_entry_name}"
    n_exp = len(experiments)
    fig, axes = get_axes(n_exp, n_exp if column_number is None else column_number, fig_frame=figsize, res=300)
    if n_exp == 1:
        axes = [axes]

    colors = sns.color_palette(palette, n_colors=n_exp)

    for ax, color, (exp_name, (_, metrics_df)) in zip(axes, colors, experiments.items()):
        raw = metrics_df[col].dropna()
        weights = np.array([
            ast.literal_eval(w) if isinstance(w, str) else list(w)
            for w in raw
        ])  # (N, P)

        P = weights.shape[1] if weights.ndim == 2 else 1

        if P == 2:
            ax.scatter(weights[:, 0], weights[:, 1], color=color,
                       alpha=0.7, s=25, edgecolors="none")
            # Simplex boundary
            ax.plot([0, 1], [1, 0], "k--", linewidth=1, alpha=0.4)
            ax.set_xlabel("w₁")
            ax.set_ylabel("w₂")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
        else:
            for j in range(P):
                ax.hist(weights[:, j], alpha=0.5, label=f"w{j+1}", bins=20)
            ax.legend(fontsize=8)
            ax.set_xlabel("Weight value")
            ax.set_ylabel("Count")

        ax.set_title(exp_name)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig, axes


def plot_pareto_hit_rate(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    target_names: List[str],
    reference_point: np.ndarray,
    exclude_init: bool = False,
    maximize: bool = True,
    palette: str = "colorblind",
    figsize: Tuple[float, float] = (8.0, 5.0),
    experiment_labels: Optional[List[str]] = None,
    acronym_map: Optional[Dict[str, str]] = ACQFUNC_ACRONYMS,
) -> Tuple[plt.Figure, plt.Axes]:
    """Grouped bar chart of Pareto hit rate per acquisition source.

    Hit rate = (points from source on final Pareto front) / (total points from source).
    Answers "per point spent, which strategy most efficiently finds Pareto-optimal points?"

    When multiple repetitions exist, hit rates are averaged across repetitions.

    Args:
        experiments: Dict mapping name to (train_points_data df, benchmark_data df).
        target_names: Objective column names.
        reference_point: Passed to compute_pareto_attribution (used for HV, not hit rate).
        maximize: True (default) when higher objective values are better.
        palette: Seaborn palette.
        figsize: Figure size.
        experiment_labels: Optional rename list for experiment keys.

    Returns:
        (fig, ax)
    """
    from activereg.metrics import compute_pareto_attribution

    reference_point = np.asarray(reference_point, dtype=float)
    if experiment_labels is not None:
        experiments = dict(zip(experiment_labels, experiments.values()))

    # Collect all unique sources for consistent bar positioning (excluding init if requested)
    all_sources: List[str] = []
    for pts, _ in experiments.values():
        pts_filt = pts[pts["cycle"] > 0] if exclude_init else pts
        for s in pts_filt["acquisition_source"].unique():
            if s not in all_sources:
                all_sources.append(s)

    colors = sns.color_palette(palette, n_colors=len(all_sources))
    source_colors = dict(zip(all_sources, colors))

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    n_exp = len(experiments)
    n_src = len(all_sources)
    total_width = 0.8
    bar_w = total_width / n_src
    offsets = np.linspace(
        -total_width / 2 + bar_w / 2, total_width / 2 - bar_w / 2, n_src
    )
    x = np.arange(n_exp)

    for src_idx, source in enumerate(all_sources):
        hit_rates_per_exp: List[float] = []
        for points_df, _ in experiments.values():
            reps = sorted(points_df["repetition"].unique())
            rep_rates: List[float] = []
            for rep in reps:
                rep_df = points_df[points_df["repetition"] == rep]
                if exclude_init:
                    rep_df = rep_df[rep_df["cycle"] > 0]
                attr = compute_pareto_attribution(
                    rep_df, target_names, reference_point, maximize
                )
                row = attr[attr["acquisition_source"] == source]
                rep_rates.append(
                    float(row["pareto_hit_rate"].iloc[0]) if len(row) > 0 else 0.0
                )
            hit_rates_per_exp.append(float(np.mean(rep_rates)))

        src_label = acronym_map.get(source, source) if acronym_map else source
        ax.bar(
            x + offsets[src_idx], hit_rates_per_exp,
            width=bar_w, label=src_label,
            color=source_colors[source], alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(list(experiments.keys()), ha="center")
    ax.set_ylabel("Pareto hit rate")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig, ax


def plot_hv_gain_attribution(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    target_names: List[str],
    reference_point: np.ndarray,
    exclude_init: bool = False,
    maximize: bool = True,
    palette: str = "colorblind",
    subplotsize: Tuple[float, float] = (8.0, 4.0),
    experiment_labels: Optional[List[str]] = None,
    acronym_map: Optional[Dict[str, str]] = ACQFUNC_ACRONYMS,
) -> Tuple[plt.Figure, list]:
    """Stacked area of cumulative HV gain attributed to each acquisition source.

    At each cumulative sample the total height equals the hypervolume discovered
    so far; the coloured bands show how much of that HV was contributed by each
    source. Multi-property analog of analyze_acquisition_source_distribution.

    Answers "which strategy drove actual Pareto front expansion?"
    Supports P=2 only (raises NotImplementedError otherwise).

    When multiple repetitions exist, only the first is used (multi-rep averaging
    of stacked areas is not straightforward and deferred).

    Args:
        experiments: Dict mapping name to (train_points_data df, benchmark_data df).
        target_names: Objective column names.
        reference_point: Array of shape (P,) for hypervolume computation.
        maximize: True (default) when higher objective values are better.
        palette: Seaborn palette.
        subplotsize: Size of each per-experiment panel.
        experiment_labels: Optional rename list for experiment keys.

    Returns:
        (fig, axes) where axes is a list with one Axes per experiment.
    """
    from activereg.metrics import compute_pareto_front, compute_hypervolume

    P = len(target_names)
    if P != 2:
        raise NotImplementedError(
            "plot_hv_gain_attribution supports P=2 only."
        )

    reference_point = np.asarray(reference_point, dtype=float)
    if experiment_labels is not None:
        experiments = dict(zip(experiment_labels, experiments.values()))

    # Consistent source ordering and colours (excluding init if requested)
    all_sources: List[str] = []
    for pts, _ in experiments.values():
        pts_filt = pts[pts["cycle"] > 0] if exclude_init else pts
        for s in pts_filt["acquisition_source"].unique():
            if s not in all_sources:
                all_sources.append(s)
    colors = sns.color_palette(palette, n_colors=len(all_sources))
    source_colors = dict(zip(all_sources, colors))

    n_exp = len(experiments)
    fig, axes = get_axes(n_exp, n_exp, fig_frame=subplotsize, res=300)
    if n_exp == 1:
        axes = [axes]

    for ax, (exp_name, (points_df, _)) in zip(axes, experiments.items()):
        # Use first repetition
        rep = sorted(points_df["repetition"].unique())[0]
        df = points_df[points_df["repetition"] == rep].sort_values("cycle")
        if exclude_init:
            df = df[df["cycle"] > 0]
        df = df.reset_index(drop=True)
        Y_seq = df[target_names].to_numpy(dtype=float)
        src_seq = df["acquisition_source"].values
        n_pts = len(df)

        # Compute marginal HV gain per point and credit to its source
        current_Y = np.empty((0, P), dtype=float)
        current_hv = 0.0
        running: Dict[str, float] = {s: 0.0 for s in all_sources}
        # cumulative_hv_by_source[s][i] = total HV from source s up to sample i
        cumulative: Dict[str, List[float]] = {s: [] for s in all_sources}

        for y, source in zip(Y_seq, src_seq):
            current_Y = np.vstack([current_Y, y.reshape(1, -1)])
            pf = compute_pareto_front(current_Y, maximize=maximize)
            new_hv = compute_hypervolume(current_Y[pf], reference_point)
            running[source] = running.get(source, 0.0) + (new_hv - current_hv)
            current_hv = new_hv
            for s in all_sources:
                cumulative[s].append(running.get(s, 0.0))

        x = np.arange(1, n_pts + 1)
        bottom = np.zeros(n_pts)
        for source in all_sources:
            vals = np.array(cumulative[source])
            src_label = acronym_map.get(source, source) if acronym_map else source
            ax.fill_between(
                x, bottom, bottom + vals,
                label=src_label, color=source_colors[source], alpha=0.85, step="post",
            )
            bottom = bottom + vals

        ax.set_title(exp_name)
        ax.set_xlabel("Cumulative samples")
        ax.set_ylabel("Cumulative HV gain")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig, axes


# --- ////////////// ---#