#!

import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from activereg.beauty import get_axes

MAIN_RESULTS_FILES = {
    "config": "config.yaml",
    "metrics_file" : "benchmark_data.csv",
    "train_points_file" : "train_points_data.csv"
}

ADAPTIVE_REFINEMET_FORMAT = "collective_refinement_points_rep{}.csv"

'''
Files description:

- config: Configuration file used for the benchmark experiment. 
It contains all the parameters and settings used during the experiment.

- metrics_file: This file contains the benchmark metrics collected during the experiment.
CSV structure: repetition,cycle,y_best_screened,y_best_predicted_pool,y_best_predicted_val,rmse_vs_gt_pool,mae_vs_gt_pool,rmse_vs_gt_val,mae_vs_gt_val,nll_val,picp95_val,mpiw95_val

- train_points_file: This file contains the training points data collected during the experiment.
CSV structure: x1,x2,x3,..,xD,cycle,repetition,acquisition_source,y_value

- collective_refinement_points_rep{}.csv: This file contains the adaptive refinement points collected during the experiment for a specific repetition.
CSV structure: x1,x2,x3,..,y,set,refinement_step

'''

DIMENSIONALITY_REDUCTION_METHODS = {
    'pca': PCA(n_components=2),
    'tsne': TSNE(n_components=2),
    'umap': UMAP(n_components=2),
}

# --------------------------------------------------------------
# Functions for plotting visual benchmark results

def plot_2d_landscape(x1, x2, y, colorbar=True):
    fig, ax = get_axes(1,1)
    y_order = y.argsort()
    x1 = x1[y_order]
    x2 = x2[y_order]
    y = y[y_order]
    sc = ax.scatter(x1, x2, c=y, cmap='PuOr_r', s=40)
    if colorbar:
        fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.1, label='Function Value')
    ax.set_xlabel('e1')
    ax.set_ylabel('e2')
    return fig, ax


def acqui_function_to_integer(acqui_function_list):
    """Convert acquisition function names to integers for color mapping."""
    mapping = {
        'random': 0,
        'expected_improvement': 1,
        'exploration_mutual_info': 2,
        'uncertainty_landscape': 2,
    }
    return [mapping[func] for func in acqui_function_list]


# --------------------------------------------------------------
# Main script

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the visual landscape of the benchmark experiment results."
    )
    parser.add_argument("-exp_name", type=str, required=True, help="Experiment name to plot.")
    parser.add_argument("-exp_dim", type=str, required=True, help="Experiment dimension (e.g., 3D, 6D).")
    parser.add_argument("-dim_red", type=str, choices=['pca', 'tsne', 'umap'], help="Dimensionality reduction method to apply.")
    parser.add_argument("-rep", type=int, required=True, help="Repetition number to plot.")
    parser.add_argument("-show_refine", action='store_true', help="Whether to include adaptive refinement points.")
    args = parser.parse_args()

    import yaml
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from activereg.format import BENCHMARKS_REPO, DATASETS_REPO
    
    chosen_repetition = args.rep

    # --------------------------------------------------------------
    # Define paths
    EXP_PATH = BENCHMARKS_REPO /  args.exp_dim / args.exp_name

    # --------------------------------------------------------------
    # Check that all main results files are present
    for file_id, file_name in MAIN_RESULTS_FILES.items():
        file_path = EXP_PATH / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Required file '{file_name}' not found in experiment path: {EXP_PATH}")
    
    # --------------------------------------------------------------
    # Load config file
    with open(EXP_PATH / MAIN_RESULTS_FILES["config"], 'r') as f:
        config = yaml.safe_load(f)

    # --------------------------------------------------------------
    # Load GT file
    gt_file_name = config.get("ground_truth_file", None)
    if gt_file_name is None:
        # Fallback to default generate GT file name
        gt_file_name = "ground_truth_dataset.csv"
        gt_file_path = EXP_PATH / gt_file_name
        if not gt_file_path.exists():
            raise FileNotFoundError(f"Ground truth file not specified in config and default file '{gt_file_name}' not found in experiment path: {EXP_PATH}")
    else:
        gt_file_path = DATASETS_REPO / gt_file_name    
        if not gt_file_path.exists():
            raise FileNotFoundError(f"Ground truth file '{gt_file_name}' not found in datasets path: {DATASETS_REPO}")
    
    gt_data = pd.read_csv(gt_file_path)

    # Get X_pool and y_pool from GT data
    search_variables_names = config.get("search_space_variables", [])
    target_variable_name = config.get("target_variables", None)
    gt_data = gt_data[gt_data['set'] == 'train']
    X_pool = gt_data[search_variables_names].to_numpy()
    y_pool = gt_data[target_variable_name].to_numpy().ravel()

    # Normalize X_pool (like in the experiment)
    scaler = StandardScaler()
    X_pool_scaled = scaler.fit_transform(X_pool)

    # --------------------------------------------------------------
    # Load the refinement points if adaptive refinement is enabled
    if args.show_refine:
        adaptive_refinement_file_name = ADAPTIVE_REFINEMET_FORMAT.format(chosen_repetition)
        adaptive_refinement_file_path = EXP_PATH / adaptive_refinement_file_name
        if not adaptive_refinement_file_path.exists():
            raise FileNotFoundError(f"Adaptive refinement file '{adaptive_refinement_file_name}' not found in experiment path: {EXP_PATH}")
        adaptive_refinement_data = pd.read_csv(adaptive_refinement_file_path)

        adaptive_refinement_data = adaptive_refinement_data[adaptive_refinement_data['set'] == 'train']
        X_refinement = adaptive_refinement_data[search_variables_names].to_numpy()
        y_refinement = adaptive_refinement_data[target_variable_name].to_numpy().ravel()
        refinement_steps = adaptive_refinement_data['refinement_step'].to_numpy().ravel()

        X_refinement_scaled = scaler.transform(X_refinement)  # Normalize refinement points

    # --------------------------------------------------------------
    # Apply dimensionality reduction if specified
    if args.dim_red is not None:
        dimred = DIMENSIONALITY_REDUCTION_METHODS[args.dim_red]
        X_pool_scaled = dimred.fit_transform(X_pool_scaled)
        if args.show_refine:
            X_refinement_scaled = dimred.transform(X_refinement_scaled)

    # --------------------------------------------------------------
    # Load metrics file
    metrics_file_path = EXP_PATH / MAIN_RESULTS_FILES["metrics_file"]
    metrics_data = pd.read_csv(metrics_file_path)

    # get number of repetitions
    n_repetitions = metrics_data['repetition'].nunique()
    # get number of cycles
    n_cycles = metrics_data['cycle'].nunique()

    # --------------------------------------------------------------
    # Load training points file
    train_points_file_path = EXP_PATH / MAIN_RESULTS_FILES["train_points_file"]
    train_points_data = pd.read_csv(train_points_file_path)
    train_points_data = train_points_data[train_points_data['repetition'] == chosen_repetition]

    # Extract training points for the chosen repetition
    train_points = train_points_data[search_variables_names].to_numpy()
    cycle_numbers = train_points_data['cycle'].tolist()
    acqui_funcs = train_points_data['acquisition_source'].tolist()
    acqui_funcs_int = acqui_function_to_integer(acqui_funcs)  # List of integers

    if args.dim_red is not None:
        train_points = dimred.transform(train_points)  # Reduce dimensionality, train is already scaled

    # --------------------------------------------------------------
    # Plot visual benchmark results for the chosen repetition
    if chosen_repetition not in train_points_data['repetition'].unique():
        raise ValueError(f"Repetition {chosen_repetition} not found in training points data.")
    print(f"Plotting results for repetition: {chosen_repetition}")

    # Select which two dimensions to plot
    if args.dim_red is not None:
        e1 = 0
        e2 = 1
    else:
        e1 = 0
        e2 = 2

    # Map acquisition function integers to distinct colors
    cmap = plt.cm.gnuplot2
    n_acqui_types = 3  # Total number of acquisition function types (0, 1, 2)
    # Normalize integers to [0, 1] range for colormap
    norm = plt.Normalize(vmin=0, vmax=n_acqui_types - 1)
    acqui_func_colors = [cmap(norm(i)) for i in acqui_funcs_int]

    # --------------------------------------------------------------
    # PLOT 1: Landscape with training points colored by acquisition function
    # --------------------------------------------------------------
    fig, ax = plot_2d_landscape(    # Landscape plot, coloured by function value (y_pool)
        x1=X_pool_scaled[:, e1],
        x2=X_pool_scaled[:, e2],
        y=y_pool
    )
    ax.set_xticks([])
    ax.set_yticks([])
    sc = ax.scatter(
        train_points[:, e1],
        train_points[:, e2],
        c=acqui_func_colors,
        marker='o',
        s=25,
        edgecolor='black',
    )
    ax.set_title('Rep. {}'.format(chosen_repetition), fontsize=8)
    fig.tight_layout()
    # fig.savefig('fig1.png')
    plt.show()

    fig, ax = plot_2d_landscape(    # Landscape plot, coloured by function value (y_pool)
        x1=X_pool_scaled[:, e1],
        x2=X_pool_scaled[:, e2],
        y=y_pool
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    # fig.savefig('fig2.png')
    plt.show()

    # --------------------------------------------------------------
    # PLOT 2: Landscape with adaptive refinement points (if any)
    # --------------------------------------------------------------
    if args.show_refine:
        # Merge original and refined pools for plotting
        X_pool_combined = np.vstack((X_pool_scaled, X_refinement_scaled))
        y_pool_combined = np.hstack((y_pool, y_refinement))

        fig, ax = plot_2d_landscape(
            x1=X_pool_combined[:, e1],
            x2=X_pool_combined[:, e2],
            y=y_pool_combined
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        # fig.savefig('fig3.png')
        plt.show()

    # --------------------------------------------------------------
    # PLOT 3: Density plot of refined vs unrefined datasets
    # --------------------------------------------------------------
        fig, ax = get_axes(3,3)
        sns.kdeplot(
            x=X_pool_scaled[:, e1],
            y=X_pool_scaled[:, e2],
            cmap="Reds",
            bw_adjust=0.5, 
            fill=True, 
            ax=ax[0])
        ax[0].set_title('Original Pool')
        #
        sns.kdeplot(
            x=X_pool_scaled[:, e1],
            y=X_pool_scaled[:, e2],
            levels=2,
            cmap="Reds",
            bw_adjust=0.5,
            alpha=0.5, 
            fill=True, 
            ax=ax[1])
        sns.kdeplot(
            x=X_refinement_scaled[:, e1],
            y=X_refinement_scaled[:, e2],
            levels=10,
            cmap="Blues",
            bw_adjust=0.5,
            fill=True, 
            ax=ax[1])
        ax[1].set_title('Refinement Points')
        #
        sns.kdeplot(
            x=X_pool_combined[:, e1],
            y=X_pool_combined[:, e2],
            levels=10,
            cmap="Purples",
            bw_adjust=0.5,
            fill=True,
            ax=ax[2])
        ax[2].set_title('Combined Pool')
        #
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        fig.tight_layout()
        # fig.savefig('fig4.png')
        plt.show()

    # --------------------------------------------------------------
    # PLOT 4: Landscape with best value progression
    # --------------------------------------------------------------
        from scipy.interpolate import interp1d, splprep, splev
        from matplotlib.colors import Normalize
        from matplotlib.collections import LineCollection
        fig, ax = plot_2d_landscape(    
            x1=X_pool_combined[:, e1],
            x2=X_pool_combined[:, e2],
            y=y_pool_combined,
            colorbar=False
        )
        ax.set_xticks([])
        ax.set_yticks([])
        best_values = []
        # compute the best value for each cycle and store the corresponding point
        for cycle in range(n_cycles):
            cycle_data = train_points_data[train_points_data['cycle'] == cycle]
            if cycle_data.empty:
                continue
            best_idx = cycle_data["y_value"].idxmax()
            best_point = cycle_data.loc[best_idx, search_variables_names].to_numpy()
            best_values.append(best_point)
        best_values = np.array(best_values, dtype=np.float64)
        # print("Best values shape:", best_values.shape, best_values)

        progression = np.arange(len(best_values))
        x = np.asarray(best_values[:, e1], dtype=np.float64).flatten()
        y = np.asarray(best_values[:, e2], dtype=np.float64).flatten()

        idx = np.argsort(progression)
        x_sorted = x[idx]
        y_sorted = y[idx]
        prog_sorted = progression[idx]

        # Parametric spline interpolation for smooth curve
        tck, u = splprep([x_sorted, y_sorted], s=0, k=min(3, len(x)-1))
        u_new = np.linspace(0, 1, 1000)
        x_interp, y_interp = splev(u_new, tck)
        
        # Interpolate progression values along the curve
        prog_interp = np.interp(u_new, u, prog_sorted)
        
        # Create line segments
        points = np.array([x_interp, y_interp]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create colored line collection
        norm = Normalize(vmin=prog_sorted.min(), vmax=prog_sorted.max())
        lc = LineCollection(segments, cmap='inferno', norm=norm, linewidth=3)
        lc.set_array(prog_interp)
        line = ax.add_collection(lc)

        ax.set_xlabel('')
        ax.set_ylabel('')
        # ax.legend()
        fig.tight_layout()
        # fig.savefig('fig5.png')
        plt.show()

    # --------------------------------------------------------------
