#!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.interpolate import griddata
from activereg.utils import create_experiment_name

# ---------------------------------------------------------------------------
# --- PLOT FUNC

def plot_predicted_landscape(X_pool: np.ndarray, pred_array: np.ndarray):
    """Plot the predicted landscape.

    Args:
        X_pool (np.ndarray): The input features for the pool.
        pred_array (np.ndarray): The predicted values, must be 2D (N_cycles, N_samples).
        save_path (Path, optional): The path to save the plot. Defaults to None.
    """
    # Check if the pred_array is 2D

    cycles, _ = pred_array.shape
    fig, ax = get_axes(cycles, 4)

    for i,pred in enumerate(pred_array):
        sc = ax[i].scatter(
            *X_pool.T, c=pred, cmap='coolwarm', s=5
        )
        ax[i].set_title(f'Cycle {i+1}')
        _ = fig.colorbar(sc, ax=ax[i])
        ax[i].set_aspect('equal')
        fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# --- PLOT UTILITIES


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