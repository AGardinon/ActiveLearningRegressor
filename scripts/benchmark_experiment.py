#!

'''
Benchamark Experiment

Actions:
1.

'''

import yaml
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error
from activereg import beauty
from activereg.sampling import sample_landscape
from activereg.utils import create_strict_folder, jsd_histogram, jsd_kde
from activereg.experiment import (get_gt_dataframes, 
                                  sampling_block, 
                                  setup_data_pool,
                                  setup_ml_model,
                                  remove_evidence_from_gt,
                                  setup_experiment_variables)
from typing import List, Tuple

# FUNCTIONS

def create_benchmark_path(
       exp_name: str,
       gt_dataframe: pd.DataFrame,
       search_space_variables: List[str],
       overwrite: bool=False,
    ) -> Tuple[Path, Path, pd.DataFrame]:

    from activereg.format import BENCHMARKS_REPO

    benchmark_path = BENCHMARKS_REPO / exp_name

    # Create the benchmark folder
    create_strict_folder(path_str=str(benchmark_path), overwrite=overwrite)

    # Create the main experiment paths
    pool_csv_path = benchmark_path / f'{exp_name}_POOL.csv'

    # Pool contains only the search space variables form the ground truth dataframe
    assert all(var in gt_dataframe.columns for var in search_space_variables), \
        f"Search space variables {search_space_variables} not found in ground truth dataframe columns."
    pool_df = gt_dataframe[search_space_variables]
    pool_df.to_csv(pool_csv_path, index=False)

    return benchmark_path, pool_csv_path, pool_df

# MAIN

if __name__ == '__main__':

    # Parse the config.yaml
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML configuration file")
    parser.add_argument("-r", "--repetitions", type=int, default=1, help="Number of repetitions for the experiment")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Extracting parameters from the config file &
    # setting parameters for the experiment
    (EXP_NAME, 
     ADDITIONAL_NOTES, 
     N_CYCLES, 
     INIT_BATCH, 
     INIT_SAMPLING, 
     CYCLE_SAMPLING, 
     ACQUI_PARAMS,
     SEARCH_VAR,
     TARGET_VAR) = setup_experiment_variables(config)

    N_BATCH = sum(acp['n_points'] for acp in ACQUI_PARAMS)
    N_REPS = args.repetitions

    landscape_penalization = config.get('landscape_penalization', None)
    if landscape_penalization is not None:
        pen_radius = landscape_penalization.get('radius', None)
        pen_strength = landscape_penalization.get('strength', None)

    # Get ground truth and evidence dataframes
    gt_df, evidence_df = get_gt_dataframes(config)

    # Create benchmark paths and set up dataframes
    # pool_df -> complete search space
    BENCHMARK_PATH, POOL_CSV_PATH, pool_df = create_benchmark_path(
        exp_name=EXP_NAME,
        gt_dataframe=gt_df,
        search_space_variables=SEARCH_VAR,
        overwrite=True,
    )

    # Save a copy of the config file in the benchmark folder
    config_save_path = BENCHMARK_PATH / 'config.yaml'
    with open(config_save_path, 'w') as file:
        yaml.dump(config, file)

    # Set up the data scaler on the complete search space
    data_scaler_type = config.get('data_scaler', None)
    if data_scaler_type is None:
        print("Data scaler type not specified in config file. Using StandardScaler as default.")
        data_scaler_type = "StandardScaler"

    scaler_path = BENCHMARK_PATH / f'{data_scaler_type}_scaler.joblib'
    X_pool, scaler = setup_data_pool(df=pool_df, search_var=SEARCH_VAR, scaler=data_scaler_type)
    joblib.dump(scaler, scaler_path)

    # Create candidates dataframe and target variable candidates used for validation
    # candidates_df -> unscreened space
    candidates_df = remove_evidence_from_gt(gt_df, evidence_df, search_vars=SEARCH_VAR)

    # Set up the model
    ML_MODEL = setup_ml_model(config)

    # --------------------------------------------------------------------------------
    # RUN THE BENCHMARK EXPERIMENT

    benchmark_data = []
    train_points_data = []
    pred_landscape_array = np.full((N_REPS, N_CYCLES, X_pool.shape[0]), np.nan)
    unc_landscape_array = np.full((N_REPS, N_CYCLES, X_pool.shape[0]), np.nan)

    # Get the acquisition mode per N_batch point that will be acquired
    predefined_acquisition_modes = []
    print("Predefined acquisition modes:")
    for acp in ACQUI_PARAMS:
        predefined_acquisition_modes.extend([acp['acquisition_mode']] * acp['n_points'])
        print(f" - {acp['n_points']} points with {acp['acquisition_mode']} acquisition")
    '''
    TODO: set up a acquisition scheduler that can handle how many points to acquire at each cycle
    following some rules in the config, eg, for the first 3 cycle only exploration and then mixed
    -> understanding how to do this
    '''

    for rep in range(N_REPS):

        # Set up candidates and train sets
        X_candidates = scaler.transform(candidates_df[SEARCH_VAR].to_numpy())
        y_candidates = candidates_df[TARGET_VAR].to_numpy()

        if evidence_df is not None:
            X_train = scaler.transform(evidence_df[SEARCH_VAR].to_numpy())
            y_train = evidence_df[TARGET_VAR].to_numpy()

        elif evidence_df is None:
            screened_indexes = sample_landscape(
                X_landscape=X_candidates,
                n_points=INIT_BATCH,
                sampling_mode=INIT_SAMPLING
            )
            X_train = X_candidates[screened_indexes]
            y_train = y_candidates[screened_indexes]

            X_candidates = np.delete(X_candidates, screened_indexes, axis=0)
            y_candidates = np.delete(y_candidates, screened_indexes, axis=0)
        
        # Add initial training points to tracking
        for i in range(len(X_train)):
            train_points_data.append({
            **{col : X_train[i,j] for j,col in enumerate(SEARCH_VAR)},
            "cycle": 0,
            "repetition": rep+1,
            "acquisition_source": INIT_SAMPLING if evidence_df is None else "evidence",
            "y_value": y_train[i][0]
            })

        for cycle in range(N_CYCLES):

            print(f"--- Repetition {rep+1}/{N_REPS} | Cycle {cycle+1}/{N_CYCLES} ---")

            # Compute the best screened value and train the model
            y_best = np.max(y_train)

            ML_MODEL.train(X_train, y_train)
            _, y_pred, y_unc = ML_MODEL.predict(X_pool)

            # Sample from the candidates
            sampled_indexes, landscape = sampling_block(
                X_candidates=X_candidates,
                X_train=X_train,
                y_best=y_best,
                ml_model=ML_MODEL,
                acquisition_params=ACQUI_PARAMS,
                sampling_mode=CYCLE_SAMPLING,
                # Hard penalization: (0.25, 1.0) (radius, strength)
                # Soft penalization: (0.25, 0.5) (radius, strength)
                # Weak penalization: (0.25, 0.1) (radius, strength)
                penalization_params=(pen_radius, pen_strength) if landscape_penalization is not None else None
            )

            benchmark_data.append({
                "repetition": rep+1,
                "cycle": cycle+1,
                "y_best_screened": y_best,
                "y_best_predicted": np.max(y_pred),
                "nll": ML_MODEL.model.log_marginal_likelihood().astype(float),
                "model_params": ML_MODEL.__repr__()
            })

            pred_landscape_array[rep, cycle] = y_pred
            unc_landscape_array[rep, cycle] = y_unc

            # Update the train and candidates sets
            X_train = np.vstack((X_train, X_candidates[sampled_indexes]))
            y_train = np.concatenate((y_train, y_candidates[sampled_indexes]))

            # Update training points tracking
            for i in range(len(X_candidates[sampled_indexes])):
                train_points_data.append({
                    **{col : X_candidates[sampled_indexes][i,j] for j,col in enumerate(SEARCH_VAR)},
                    "cycle": cycle+1,
                    "repetition": rep+1,
                    "acquisition_source": predefined_acquisition_modes[i],
                    "y_value": y_candidates[sampled_indexes][i][0]
                })

            X_candidates = np.delete(X_candidates, sampled_indexes, axis=0)
            y_candidates = np.delete(y_candidates, sampled_indexes, axis=0)


    # Save benchmark data to CSV
    benchmark_df = pd.DataFrame(benchmark_data)
    benchmark_df.to_csv(BENCHMARK_PATH / 'benchmark_data.csv', index=False)

    train_points_df = pd.DataFrame(train_points_data)
    train_points_df.to_csv(BENCHMARK_PATH / 'train_points_data.csv', index=False)

    np.save(BENCHMARK_PATH / 'predicted_landscape.npy', pred_landscape_array)
    np.save(BENCHMARK_PATH / 'uncertainty_landscape.npy', unc_landscape_array)


    # PLOTS

    acquisition_markers_dict = {
        'fps': {'marker': '*', 'color': 'black', 's': 30},
        'random': {'marker': '*', 'color': 'black', 's': 30},
        'voronoi': {'marker': '*', 'color': 'black', 's': 30},
        'exploration_mutual_info': {'marker': '^', 'color': 'white', 's': 20},
        'expected_improvement': {'marker': 'o', 'color': 'yellow', 's': 20},
        'target_expected_improvement': {'marker': 'D', 'color': 'orange', 's': 20}
    }

    rmse_per_rep = []
    jsd_per_rep = []

    for rep in range(N_REPS):

        # -------------------------------------------------------------------------------------------------------------
        # 1. Highlight target_expected_improvement points
        # if "target_expected_improvement" is in the acquisition modes,
        # plot hilight of the points in the plane that have value = target
        tei_dict = next((acp for acp in ACQUI_PARAMS if acp['acquisition_mode'] == 'target_expected_improvement'), None)
        if tei_dict is not None:

            target_value = tei_dict.get('y_target', None)
            target_epsilon = tei_dict.get('epsilon', 50)
            assert target_value is not None, "Target value for target_expected_improvement not specified in ACQUI_PARAMS."
            
            fig1, ax1 = beauty.plot_predicted_landscape(X_pool, pred_landscape_array[rep], columns=3)
            
            for i, tei_land in enumerate(pred_landscape_array[rep]):
                # Create a regular grid for interpolation
                grid_size = 100  # Adjust for smoother contours
                x = np.linspace(X_pool[:, 0].min()-0.1, X_pool[:, 0].max()+0.1, grid_size)
                y = np.linspace(X_pool[:, 1].min()-0.1, X_pool[:, 1].max()+0.1, grid_size)

                # Create meshgrid
                X, Y = np.meshgrid(x, y)
                
                # Interpolate the prediction values onto the grid
                Z = griddata((X_pool[:, 0], X_pool[:, 1]), tei_land, (X, Y), method='linear')
                
                # Plot the contour at the target value
                ax1[i].contour(
                    X, Y, Z,
                    levels=[target_value-target_epsilon, target_value, target_value+target_epsilon],
                    colors='black',
                    linestyles=['--', '-', '--'],
                    linewidths=1.2
                )

                # Highlight the target area
                ax1[i].contourf(
                    X, Y, Z,
                    levels=[target_value-target_epsilon, target_value+target_epsilon],
                    colors='grey',
                    alpha=0.2
                )

        # -------------------------------------------------------------------------------------------------------------
        # 2. Plot the predicted landscape with the acquired points
        rep_train_data = train_points_df[train_points_df['repetition'] == rep + 1]
        fig, ax = beauty.plot_predicted_landscape(X_pool, pred_landscape_array[rep], columns=3)

        # plot train points having a different marker based on the predefined acquisition modes
        for i in range(len(ax)):
            if i < N_CYCLES:

                cycle_train_data = rep_train_data[rep_train_data['cycle'] <= i]
                acquisition_modes = cycle_train_data['acquisition_source'].values

                for acqui_mode in acquisition_modes:

                    acqui_data = cycle_train_data[cycle_train_data['acquisition_source'] == acqui_mode]

                    ax[i].scatter(
                        acqui_data[SEARCH_VAR[0]],
                        acqui_data[SEARCH_VAR[1]],
                        **acquisition_markers_dict.get(acqui_mode, {'marker': 'x', 'color': 'grey', 's': 20}),
                        edgecolor='black'
                    )

                    if tei_dict is not None:
                        ax1[i].scatter(
                            acqui_data[SEARCH_VAR[0]],
                            acqui_data[SEARCH_VAR[1]],
                            **acquisition_markers_dict.get(acqui_mode, {'marker': 'x', 'color': 'grey', 's': 20}),
                            edgecolor='black'
                        )

        fig.savefig(BENCHMARK_PATH / f'predicted_landscape_rep{rep+1}.png')
        plt.close(fig)

        if tei_dict is not None:
            fig1.savefig(BENCHMARK_PATH / f'predicted_landscape_tei_rep{rep+1}.png')
            plt.close(fig1)

        # -------------------------------------------------------------------------------------------------------------
        # 3. RMSE per cycle against the ground truth landscape
        gt_landscape = gt_df[TARGET_VAR].to_numpy().ravel()
        rmse_per_cycle = [
            np.sqrt(mean_squared_error(gt_landscape, pred_landscape_array[rep, cycle]))
            for cycle in range(N_CYCLES)
        ]
        rmse_per_rep.append(rmse_per_cycle)

        # -------------------------------------------------------------------------------------------------------------
        # 4. JSD per cycle against the ground truth landscape
        jsd_per_cycle = [
            jsd_kde(gt_landscape, pred_landscape_array[rep, cycle])
            for cycle in range(N_CYCLES)
        ]
        jsd_per_rep.append(jsd_per_cycle)


    # -> 3. + 4. Plot the mean and std of the RMSE per cycle over the repetitions
    # and the JSD per cycle over the repetitions
    rmse_per_rep = np.array(rmse_per_rep)
    mean_rmse = np.mean(rmse_per_rep, axis=0)
    std_rmse = np.std(rmse_per_rep, axis=0)

    jsd_per_rep = np.array(jsd_per_rep)
    mean_jsd = np.mean(jsd_per_rep, axis=0)
    std_jsd = np.std(jsd_per_rep, axis=0)

    fig, ax = beauty.get_axes(2, 2)
    ax[0].errorbar(
        x=np.arange(1, N_CYCLES+1),
        y=mean_rmse,
        yerr=std_rmse,
        fmt='-o',
        color='blue',
        ecolor='black',
        elinewidth=1.5,
        capsize=2.5
    )
    ax[0].set_xlabel('Cycle')
    ax[0].set_ylabel('RMSE')
    ax[0].set_title('Predicted landscape vs GT')
    ax[0].grid(linestyle='--', alpha=0.5, zorder=-1)

    ax[1].errorbar(
        x=np.arange(1, N_CYCLES+1),
        y=mean_jsd,
        yerr=std_jsd,
        fmt='-o',
        color='green',
        ecolor='black',
        elinewidth=1.5,
        capsize=2.5
    )
    ax[1].set_xlabel('Cycle')
    ax[1].set_ylabel('JSD')
    ax[1].set_title('Predicted landscape vs GT')
    ax[1].grid(linestyle='--', alpha=0.5, zorder=-1)
    
    fig.tight_layout()
    fig.savefig(BENCHMARK_PATH / 'rmse_jsd_per_cycle.png')
    plt.close(fig)

    # save the average rmse and jsd data as csv
    rmse_jsd_df = pd.DataFrame({
        'cycle': np.arange(1, N_CYCLES+1),
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'mean_jsd': mean_jsd,
        'std_jsd': std_jsd
    })
    rmse_jsd_df.to_csv(BENCHMARK_PATH / 'rmse_jsd_per_cycle.csv', index=False)