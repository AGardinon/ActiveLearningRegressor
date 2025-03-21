#!

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from activereg.sampling import sample_landscape
from activereg.acquisition import landscape_acquisition, highest_landscape_selection
from activereg.beauty import get_axes

# Test experiment setup
OUTDIR = './test1/'
X_pool = np.load('./data/point_space_2d_scaled.npy')
y_pool = np.load('./data/proptest2d_2.npy')

# Regressor setup
kernel = Matern(length_scale=1.0, nu=2.5)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# - func

def plot_cycles(train_set, next_set, pool_set, pred_set, landscape_set, name_set, show=False):
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
    fig_name_list = [str(i) for i in name_set]
    fig_name = fig_name_list[0]+"_".join(fig_name_list[1:])
    fig.savefig(fig_name+'.png')
    if show:
        plt.show()

# - main

def main(n_batch: int=4, 
         init_sampling_mode: str='fps',
         n_cycles: int=8,
         acquisition_mode: str='explore_uncertainty',
         sampling_mode: str='voronoi'):

    # 1. init experiment
    # Sample initial points
    init_idx = sample_landscape(X_landscape=X_pool, 
                                n_points=n_batch,
                                mode=init_sampling_mode)
    
    X_train, y_train = X_pool[init_idx], y_pool[init_idx]
    X_candidates = np.delete(X_pool, init_idx, axis=0)
    y_candidates = np.delete(y_pool, init_idx, axis=0)

    # Model training on initial configuration
    gpr.fit(X_train, y_train)

    # 2. cycles
    for c in range(n_cycles):

        print(f'Cycle: {c+1}')

        # 3. compute the navigation landscape
        y_pred, landscape = landscape_acquisition(X_candidates=X_candidates,
                                          gp_model=gpr,
                                          acquisition_mode=acquisition_mode)
        # get the indexes of the top percentile based on the current candidates
        acq_landscape_ndx = highest_landscape_selection(landscape=landscape, percentile=80)
        X_acq_landscape = X_candidates[acq_landscape_ndx]
        
        # 4. sample new points in from the landscape
        sampled_hls_idx = sample_landscape(X_landscape=X_acq_landscape, 
                                           n_points=n_batch, 
                                           sampling_mode=sampling_mode)
        sampled_new_idx = acq_landscape_ndx[sampled_hls_idx]

        X_next, y_next = X_candidates[sampled_new_idx], y_candidates[sampled_new_idx]

        # - plt
        plot_cycles(train_set=(X_train,y_train),
                    pred_set=(X_candidates,y_pred),
                    pool_set=(X_pool,y_pool,'coolwarm'),
                    next_set=(X_next,y_next),
                    landscape_set=(landscape,X_acq_landscape,'plasma'),
                    name_set=(OUTDIR,'fig',init_sampling_mode,acquisition_mode,sampling_mode,c))

        # 5. update the trainig set
        X_train = np.vstack((X_train, X_next))
        y_train = np.append(y_train, y_next)

        # remove selected point from candidates
        X_candidates = np.delete(X_candidates, sampled_new_idx, axis=0)
        y_candidates = np.delete(y_candidates, sampled_new_idx, axis=0)

        # 6. retrain ALmodel with new points
        gpr.fit(X_train, y_train)


if __name__ == '__main__':
    main()
