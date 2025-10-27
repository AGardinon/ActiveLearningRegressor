#!

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

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
        mean_curve = all_curves.mean(axis=0)
        std_curve = all_curves.std(axis=0)
        
        cycles = metrics_df[metrics_df['repetition'] == metrics_df['repetition'].unique()[0]]['cycle'].values
        # if total points is given, show behavior vs points per cycle
        if total_points is not None:
            points_per_cycle = total_points // len(cycles)
            cycles = cycles * points_per_cycle

        ax.plot(cycles, mean_curve, label=exp_name, linewidth=2, 
                # marker='o', 
                color=colors[len(ax.lines)])
        ax.fill_between(cycles, 
                        mean_curve - std_curve, 
                        mean_curve + std_curve, 
                        alpha=0.15, color=colors[len(ax.lines)-1], zorder=-1)

    if true_optimum is not None:
        ax.axhline(y=true_optimum, color='.5', linestyle=':', label='Maximum value', zorder=-1)
    
    if total_points is not None:
        ax.set_xlabel('Number of Points Sampled', fontsize=12)
    else:
        ax.set_xlabel('BO Cycle', fontsize=12)
    ax.set_ylabel(f'Best Value ({value_col})', fontsize=12)
    ax.set_title('Convergence: Best Value Over Time', fontsize=12)
    ax.legend(fontsize=10, loc=4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_simple_regret(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    true_optimum: float,
    total_points: int = None,
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
            values = rep_data[value_col].values
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
        ax.plot(cycles, mean_regret, label=exp_name, linewidth=2, 
                # marker='o', 
                color=colors[len(ax.lines)])
        ax.fill_between(cycles, 
                        np.maximum(mean_regret - std_regret, 1e-10),
                        mean_regret + std_regret, 
                        alpha=0.15, color=colors[len(ax.lines)-1])

    if total_points is not None:
        ax.set_xlabel('Number of Points Sampled', fontsize=12)
    else:
        ax.set_xlabel('BO Cycle', fontsize=12)
    ax.set_ylabel('Simple Regret', fontsize=12)
    ax.set_title('Convergence Speed (Lower is Better)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    return fig, ax


def plot_sample_efficiency(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], 
    value_col: str = 'y_best_screened',
    palette: str = 'colorblind',
    experiment_labels: List[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot sample efficiency: Best Value vs Total Samples.

    Args:
        experiments (Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]): The experimental data.
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
        all_curves_samples = []
        all_curves_values = []
        
        for rep in metrics_df['repetition'].unique():
            rep_metrics = metrics_df[metrics_df['repetition'] == rep].sort_values('cycle')
            rep_points = points_df[points_df['repetition'] == rep].sort_values('cycle')
            
            # Count cumulative samples per cycle
            cumulative_samples = []
            for cycle in rep_metrics['cycle'].values:
                n_samples = len(rep_points[rep_points['cycle'] <= cycle])
                cumulative_samples.append(n_samples)
            
            values = rep_metrics[value_col].values
            
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
    
    ax.set_xlabel('Total Samples Collected', fontsize=12)
    ax.set_ylabel(f'Best Value ({value_col})', fontsize=12)
    ax.set_title('Sample Efficiency: Best Value vs Total Samples', fontsize=12)
    ax.legend(fontsize=10, loc=4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
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

        ax.bar(x + offsets[i], means, width=bar_width, 
               label=f"{int(thresh*100)}%", color=colors[i], 
               yerr=stds, capsize=5, alpha=0.8)

        if plot_line:
            ax.scatter(x + offsets[i], means, color=colors[i], marker='o', edgecolor='black')
            ax.plot(x + offsets[i], means, color=colors[i], linestyle='--', linewidth=1, zorder=-1)

    if total_points_ref is not None:
        ax.set_ylim(0, total_points_ref)
        ax.set_ylabel('Points sampled', fontsize=12)
        ax.set_title('Points to performance thresholds', fontsize=12)
    else:
        ax.set_ylabel('Cycles', fontsize=12)
        ax.set_title('Cycles to performance thresholds', fontsize=12)

    if experiment_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(experiment_labels, ha='center')
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(threshold_df['Experiment'], ha='center')

    ax.legend(title='Thresholds', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig, ax

# TODO: debug this function
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
                                          fill_value='extrapolate')
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

        ax.plot(normalized_x, mean_div, label=exp_name, linewidth=2,
                # marker='o', 
                color=colors[len(ax.lines)])
        ax.fill_between(normalized_x,
                        mean_div - std_div,
                        mean_div + std_div,
                        alpha=0.15, color=colors[len(ax.lines)-1])
    
    ax.set_xlabel('Experiment Completion (%)', fontsize=12)
    ax.set_ylabel('Mean Pairwise Distance in Batch', fontsize=12)
    ax.set_title('Batch Diversity Over Experiment Progress', fontsize=12)
    ax.set_xlim([-5, 105])
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax

# TODO: debug this function
def analyze_acquisition_source_distribution(
    experiments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], 
    experiment_labels: List[str] = None,
    n_ticks: int = 3,
    subplotsize: Tuple[float, float] = (3., 3.),
    n_subplots_cols: int = 3
) -> Tuple[plt.Figure, plt.Axes]:
    """Analyze the distribution of acquisition sources over time.

    Args:
        experiments (Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]): Dictionary of experiments with their data.
        experiment_labels (List[str], optional): Labels for the experiments. Defaults to None.
        n_ticks (int, optional): Number of ticks for the x-axis. Defaults to 3.
        figsize (Tuple[float, float], optional): Figure size. Defaults to (3.3, 3).

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes objects.
    """
    # defined a dict of acronyms for the acquisition sources
    source_acronyms = {
        'exploration_mutual_info' : 'MI',
        'uncertainty_landscape' : 'UL',
        'upper_confidence_bound' : 'UCB',
        'expected_improvement' : 'EI',
        'target_expected_improvement' : 'TEI',
        'percentage_target_expected_improvement' : '%TEI',
        'random' : 'RND'
    }
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
        colors = plt.cm.Set3(np.linspace(0, 1, len(sources)))
        
        for source, color in zip(sources, colors):
            ax.fill_between(cycles, bottom, bottom + mean_percentages[source], 
                           label=source, alpha=1., color=color)
            bottom += mean_percentages[source]
        
        # Customize ticks and labels
        tick_indices = np.linspace(0, len(cycles) - 1, n_ticks, dtype=int)
        tick_values = [cycles[i] for i in tick_indices]
        ax.set_xticks(tick_values)
        ax.set_xticklabels([str(int(c)) for c in tick_values])
        ax.set_xlabel('BO Cycle', fontsize=10)
        ax.set_ylabel('% of Batch', fontsize=10)
        ax.set_title(exp_name, fontsize=12)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 100])
    
    # Hide unused subplots
    for idx in range(len(experiments), len(axes)):
        axes[idx].axis('off')
    
    fig.tight_layout()
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
        figsize (Tuple[int, int], optional): Figure size. Defaults to (7, 4).

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
            
            ax.plot(cycles, mean_curve, label=exp_name, linewidth=2, 
                    # marker='o', 
                    color=colors[len(ax.lines)])
            ax.fill_between(cycles, 
                            mean_curve - std_curve, 
                            mean_curve + std_curve, 
                            alpha=0.15, color=colors[len(ax.lines)-1])
        
        if total_points is not None:
            ax.set_xlabel('Number of Points Sampled', fontsize=10)
        else:
            ax.set_xlabel('BO Cycle', fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes

# ---------------------------------------------------------------------------
# --- PLOT UTILITIES


def prepare_multiple_experiments(experiment_paths: List[str]) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load multiple experiments for comparison.
    
    Args:
        experiment_paths: {exp_name: (points_csv_path, metrics_csv_path)}
    
    Returns:
        {exp_name: (points_df, metrics_df)}
    """
    POINTS_PATHS = "train_points_data.csv"
    METRICS_PATHS = "benchmark_data.csv"

    experiments = {}
    for exp_name in experiment_paths:
        points_df = pd.read_csv(Path(exp_name) / POINTS_PATHS)
        metrics_df = pd.read_csv(Path(exp_name) / METRICS_PATHS)
        experiments[exp_name] = (points_df, metrics_df)
    return experiments


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

# --- ////////////// ---#