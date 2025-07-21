#!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.interpolate import griddata
from activereg.utils import create_experiment_name

# ---------------------------------------------------------------------------
# --- PLOT FUNC

def plot2D_surfaceplot(
        df: pd.DataFrame,
        pdf: np.ndarray,
        var1: str,
        var2: str,
        levels: int|list,
        cmap: str,
        contours_dict: dict,
        axis
    ) -> None:
    
    X = df[var1]
    Y = df[var2]

    surface_edges = .0

    # Determine the range for X and Y
    x_min, x_max = X.min()-surface_edges, X.max()+surface_edges
    y_min, y_max = Y.min()-surface_edges, Y.max()+surface_edges

    Z = pdf

    # Generate a grid and interpolate the diffusion coefficients
    bins=100j
    grid_x, grid_y = np.mgrid[x_min:x_max:bins, y_min:y_max:bins]
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='linear')
    grid_z = np.round(grid_z, decimals=4)

    contours = axis.contour(
        grid_x, grid_y, grid_z,
        levels=levels, colors='.0')
    axis.clabel(contours, inline=True, **contours_dict)

    contours = axis.contourf(
        grid_x, grid_y, grid_z, 
        levels=levels, cmap=cmap)

    axis.set_xlabel(var1)
    axis.set_ylabel(var2)


def plot_2Dcycle(train_set, next_set, pool_set, pred_set, landscape_set, name_set, show=False):
    fig, ax = get_axes(3,3)

    X,y,cmap = pool_set
    Xc,yp = pred_set
    Xt,yt = train_set
    Xn,yn = next_set
    landsc,hls,cmapl = landscape_set

    ax[0].scatter(*Xc.T,c=yp,s=5,cmap=cmap,vmin=min(y),vmax=max(y))
    ax[0].scatter(*Xt.T,c=yt,s=30,cmap=cmap,vmin=min(y),vmax=max(y),marker='o',edgecolor='black',zorder=3)
    ax[0].scatter(*Xn.T,c=yn,s=30,marker='*',edgecolor='black',zorder=3)

    ax[1].scatter(*Xc.T,c=landsc,s=5,cmap=cmapl,vmin=min(landsc),vmax=max(landsc))
    ax[1].scatter(*hls.T,c='.5',s=5,alpha=.3)
    ax[1].scatter(*Xt.T,s=30,c='0.',marker='o',edgecolor='black',zorder=3)
    ax[1].scatter(*Xn.T,s=30,c='0.',marker='*',edgecolor='black',zorder=3)

    ax[2].scatter(*X.T,c=y,s=5,cmap=cmap,vmin=min(y),vmax=max(y))
    ax[2].scatter(*Xt.T,c=yt,s=30,cmap=cmap,vmin=min(y),vmax=max(y),marker='o',edgecolor='black',zorder=3)

    fig.tight_layout()
    out_dir = name_set[0]
    fig_name = create_experiment_name(name_set=name_set[1:])
    fig.savefig(out_dir / Path(fig_name+'.png'))
    if show:
        plt.show()

    return fig, ax


def add_table_to_plot(ax, data_dict, title="Model Information", 
                      table_position='upper right', fontsize=9, 
                      cell_facecolor="#ffffff", cell_alpha=0.8,
                      cellLoc='left'):
    """
    Add a formatted table to a matplotlib axes object.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object to add the table to
    data_dict : dict
        Dictionary of dictionaries containing the data to display
    title : str, optional
        Title for the table (not displayed visually)
    table_position : str, optional
        Position of the table ('upper right', 'upper left', etc.)
    fontsize : int, optional
        Font size for table text
    cell_facecolor : str, optional
        Background color of the cells
    cell_alpha : float, optional
        Transparency of the background color
    cellLoc : str, optional
        Cell alignment ('left', 'center', 'right')
    
    Returns:
    --------
    table : matplotlib.table.Table
        The created table object
    """
    
    # Flatten the nested dictionary into a list of rows
    table_data = []
    for section, values in data_dict.items():
        table_data.append([f"{section}:", ""])
        for key, value in values.items():
            if isinstance(value, float):
                if abs(value) < 1e-3 or abs(value) > 1e3:
                    formatted_value = f"{value:.2e}"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            table_data.append([f"  {key}", formatted_value])
        if section != list(data_dict.keys())[-1]:
            table_data.append(["", ""])

    # Create the table
    table = ax.table(cellText=table_data,
                     colLabels=None,
                     loc=table_position,
                     cellLoc=cellLoc)

    # Table styling
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.3)

    # Style section headers
    for i, row in enumerate(table_data):
        is_header = row[0].endswith(':') and not row[0].startswith('  ')
        for j in range(2):
            cell = table[i, j]
            if is_header:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor(cell_facecolor)
            cell.set_alpha(cell_alpha)
            cell.set_linewidth(0)

    return table

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