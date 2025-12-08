#!

import argparse
import matplotlib.pyplot as plt
from activereg.beauty import get_axes

MAIN_RESULTS_FILES = {
    "config": "config.yaml",
    "metrics_file" : "benchmark_data.csv",
    "train_points_file" : "train_points_data.csv"
}
'''
Files description:

- config: Configuration file used for the benchmark experiment. 
It contains all the parameters and settings used during the experiment.

- metrics_file: This file contains the benchmark metrics collected during the experiment.
CSV structure: repetition,cycle,y_best_screened,y_best_predicted_pool,y_best_predicted_val,rmse_vs_gt_pool,mae_vs_gt_pool,rmse_vs_gt_val,mae_vs_gt_val,nll_val,picp95_val,mpiw95_val

- train_points_file: This file contains the training points data collected during the experiment.
CSV structure: x1,x2,x3,..,xD,cycle,repetition,acquisition_source,y_value
'''

ADAPTIVE_REFINEMET_FORMAT = "collective_refinement_points_rep{}.csv"

# --------------------------------------------------------------
# Functions for plotting visual benchmark results

def plot_2d_landscape(x1, x2, y):
    fig, ax = get_axes(1,1)
    y_order = y.argsort()
    x1 = x1[y_order]
    x2 = x2[y_order]
    y = y[y_order]
    sc = ax.scatter(x1, x2, c=y, cmap='PuOr_r', s=40)
    ax.set_xlabel('e1')
    ax.set_ylabel('e2')
    return fig, ax


def acqui_function_to_integer(acqui_function_list):
    """Convert acquisition function names to integers for color mapping."""
    mapping = {
        'random': 0,
        'expected_improvement': 1,
        'exploration_mutual_info': 2,
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
    parser.add_argument("-rep", type=int, required=True, help="Repetition number to plot.")
    args = parser.parse_args()

    import yaml
    import pandas as pd
    import numpy as np
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

    # --------------------------------------------------------------
    # Load adaptive refinement files, if defined
    adaptive_refinement_config = config.get("adaptive_refinement", None)
    refinement_files = {}
    if adaptive_refinement_config is not None:
        refinement_file_name = ADAPTIVE_REFINEMET_FORMAT.format(chosen_repetition)
        refinement_file_path = EXP_PATH / refinement_file_name
        if not refinement_file_path.exists():
            raise FileNotFoundError(f"Adaptive refinement file '{refinement_file_name}' not found in experiment path: {EXP_PATH}")
        refined_df = pd.read_csv(refinement_file_path)

        X_pool_refined = refined_df[search_variables_names].to_numpy()
        X_pool_refined_scaled = scaler.transform(X_pool_refined)  # Normalize
        y_pool_refined = refined_df[target_variable_name].to_numpy().ravel()

    # --------------------------------------------------------------
    # Plot visual benchmark results for the chosen repetition
    if chosen_repetition not in train_points_data['repetition'].unique():
        raise ValueError(f"Repetition {chosen_repetition} not found in training points data.")
    print(f"Plotting results for repetition: {chosen_repetition}")

    train_points = train_points_data[train_points_data['repetition'] == chosen_repetition][search_variables_names].to_numpy()
    cycle_numbers = train_points_data[train_points_data['repetition'] == chosen_repetition]['cycle'].tolist()
    acqui_funcs = train_points_data[train_points_data['repetition'] == chosen_repetition]['acquisition_source'].tolist()
    acqui_funcs_int = acqui_function_to_integer(acqui_funcs)  # List of integers

    e1 = 0
    e2 = 2

    # Map acquisition function integers to distinct colors
    cmap = plt.cm.gnuplot2
    n_acqui_types = 3  # Total number of acquisition function types (0, 1, 2)
    # Normalize integers to [0, 1] range for colormap
    norm = plt.Normalize(vmin=0, vmax=n_acqui_types - 1)
    acqui_func_colors = [cmap(norm(i)) for i in acqui_funcs_int]

    fig, ax = plot_2d_landscape(
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
        # cmap='gnuplot2',
        marker='o',
        s=25,
        edgecolor='black',
    )
    ax.set_title('Rep. {}'.format(chosen_repetition), fontsize=8)
    fig.tight_layout()
    fig.savefig('s_1stage.png')
    plt.show()

    fig, ax = plot_2d_landscape(
        x1=X_pool_scaled[:, e1],
        x2=X_pool_scaled[:, e2],
        y=y_pool
    )
    ax.set_xticks([])
    ax.set_yticks([])
    sc = ax.scatter(
        train_points[:, e1],
        train_points[:, e2],
        c=cycle_numbers,
        cmap='inferno',
        marker='o',
        s=25,
        edgecolor='black',
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.set_title('Rep. {}'.format(chosen_repetition), fontsize=8)
    # locate the colorbar on top of the frame to not distort the layout
    fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.1, label='Cycle Number')
    fig.tight_layout()
    # fig.savefig('h_mi_progression.png')
    # plt.show()

    # # plot 2D PCA projection of the pool
    # from sklearn.decomposition import PCA
    # dimred = PCA(n_components=2)
    # # dimred = TSNE(n_components=2, init='random', random_state=0)
    # X_pool_dimred = dimred.fit_transform(X_pool_scaled)
    # train_points_dimred = dimred.transform(train_points)
    # fig, ax = plot_2d_landscape(
    #     x1=X_pool_dimred[:, 0],
    #     x2=X_pool_dimred[:, 1],
    #     y=y_pool
    # )
    # ax.set_xticks([])
    # ax.set_yticks([])
    # sc = ax.scatter(
    #     train_points_dimred[:, 0],
    #     train_points_dimred[:, 1],
    #     c=cycle_numbers,
    #     cmap='inferno',
    #     marker='o',
    #     s=25,
    #     edgecolor='black',
    # )
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    # # ax.set_title('Rep. {}'.format(chosen_repetition), fontsize=8)
    # # locate the colorbar on top of the frame to not distort the layout
    # fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.1, label='Cycle Number')
    # fig.tight_layout()
    # plt.show()

    # if adaptive_refinement_config is not None:
    #     # Merge original and refined pools for plotting
    #     X_pool_combined = np.vstack((X_pool_scaled, X_pool_refined_scaled))
    #     y_pool_combined = np.hstack((y_pool, y_pool_refined))

    #     fig, ax = plot_2d_landscape(
    #         x1=X_pool_combined[:, e1],
    #         x2=X_pool_combined[:, e2],
    #         y=y_pool_combined
    #     )
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     sc = ax.scatter(
    #         train_points[:, e1],
    #         train_points[:, e2],
    #         c=cycle_numbers,
    #         cmap='inferno',
    #         marker='o',
    #         s=25,
    #         edgecolor='black',
    #     )
    #     ax.set_xlabel('')
    #     ax.set_ylabel('')
    #     # ax.set_title('Rep. {}'.format(chosen_repetition), fontsize=8)
    #     # locate the colorbar on top of the frame to not distort the layout
    #     fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.1, label='Cycle Number')
    #     fig.tight_layout()
    #     plt.show()