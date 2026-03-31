#!

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Tuple

# Plot 2D interpolated results
def plot2D_interpolated_results(
    interpolated_dict: Dict[str, Dict[str, np.ndarray]],
    x_name: str,
    y_name: str,
    palette: str = 'Set1',
    fig: plt.Figure = None,
    axes: plt.Axes = None,
    figsize: Tuple[int, int] = (6.5, 4),
):
    if fig is None or axes is None:
        fig, axes = plt.subplots(figsize=figsize, dpi=300)

    colors = sns.color_palette(palette, n_colors=len(interpolated_dict.keys()))

    for i, exp_name in enumerate(interpolated_dict.keys()):
        x_values = interpolated_dict[exp_name][x_name]
        y_values = interpolated_dict[exp_name][y_name]

        axes.plot(
            x_values,
            y_values,
            label=exp_name,
            color=colors[i],
        )

    return fig, axes


# Get a specific color from a matplotlib colormap
def get_palette_color(palette_name, n_divisions, position, return_hex=True):
    """
    Extract a specific color from a matplotlib colormap.
    """
    # Get the palette with n_divisions colors
    palette = sns.color_palette(palette_name, n_colors=n_divisions)
    color = palette[position]
    
    if return_hex:
        return mcolors.to_hex(color)
    return color


def lighten_palette(palette_name, n_colors, lightness_factor=1.2):
    """
    Lighten/darken entire palette uniformly.
    
    lightness_factor > 1: lighter
    lightness_factor < 1: darker
    """
    import colorsys
    base_palette = sns.color_palette(palette_name, n_colors=n_colors)
    
    adjusted_palette = []
    for color in base_palette:
        h, l, s = colorsys.rgb_to_hls(*color)
        new_l = min(1.0, l * lightness_factor)  # Cap at 1.0
        new_color = colorsys.hls_to_rgb(h, new_l, s)
        adjusted_palette.append(new_color)
    
    return adjusted_palette


def score_and_plot_joint_metric(
    experiments_data: pd.DataFrame,
    X: str,
    Y: str,
    method: str,
    normalize: bool = True,
    weights: Tuple[float, float] = (1.0, 1.0),
    top_k: int = 3,
    palette: str = 'Set1',
    figsize: Tuple[int, int] = (6.5, 4),
    legend: bool = True,
    legend_top_k: int = 6,
    show_only_top_k: bool = False,
):
    from activereg.metrics import get_top_k_experiments

    if isinstance(palette, str):
        plot_palette = sns.color_palette(palette, n_colors=len(experiments_data['experiment'].unique()))
    else:
        plot_palette = palette

    legend_name_to_index = {exp: i for i, exp in enumerate(experiments_data['experiment'].unique())}

    weight_x, weight_y = weights
    top_experiments, scores = get_top_k_experiments(
        experiments_data=experiments_data,
        X=X,
        Y=Y,
        method=method,
        normalize=normalize,
        weight_x=weight_x,
        weight_y=weight_y,
        top_k=top_k,
    )

    print(f"Top {top_k} experiments for metric '{method}':")
    print(f"Best experiment: {top_experiments[0]}")
    print(f"Score: {scores[top_experiments[0]]:.4f}")
    print(f"Best {top_k} experiments: {top_experiments}")

    # Plotting
    fig, axes = plt.subplots(figsize=figsize, dpi=300)
    for i, exp in enumerate(experiments_data['experiment'].unique()):
        x_data = experiments_data[experiments_data['experiment'] == exp][X].values
        y_data = experiments_data[experiments_data['experiment'] == exp][Y].values

        if exp in top_experiments:
            rank_tmp = top_experiments.index(exp) + 1
            axes.plot(x_data, y_data, linewidth=4, color=plot_palette[i], zorder=10 - rank_tmp, alpha=1.0)
            # Mark the start and end points
            axes.scatter(x_data[-1], y_data[-1], color=plot_palette[i], s=60, zorder=15 - rank_tmp, marker='d', edgecolor='black')
            axes.scatter(x_data[0], y_data[0], color='white', s=60, zorder=10 - rank_tmp, edgecolor='black')
        elif not show_only_top_k:
            axes.plot(x_data, y_data, color=plot_palette[i], linewidth=1.5, alpha=0.5)

    # Custom legend with the top 6 experiments
    ordered_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ordered_experiments = [item[0] for item in ordered_scores]

    if legend:
        # Show top experiments in the legend, or all if there are fewer than legend_top_k
        upto = min(legend_top_k, len(ordered_experiments))
        for i, exp in enumerate(ordered_experiments[:upto]):
            if exp in top_experiments:
                axes.plot([], [], label=f'{exp} (#{i+1})', color=plot_palette[legend_name_to_index[exp]], linewidth=3)
            else:
                axes.plot([], [], label=f'{exp} (#{i+1})', color=plot_palette[legend_name_to_index[exp]], linewidth=1.5, alpha=0.5)
        
        axes.legend(title=f"Ranking top-{upto}", loc='center left', bbox_to_anchor=(1.0, 0.5))

    return fig, axes, top_experiments, scores