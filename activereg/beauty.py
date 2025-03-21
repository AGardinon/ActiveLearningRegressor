#!

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# --- PLOT FUNC


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
    fig_name_list = [str(i) for i in name_set[1:]]
    fig_name = fig_name_list[0]+"_".join(fig_name_list[1:])
    fig_name = fig_name+'.png'
    fig.savefig(name_set[0] / fig_name)
    if show:
        plt.show()

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

# --- ////////////// ---#