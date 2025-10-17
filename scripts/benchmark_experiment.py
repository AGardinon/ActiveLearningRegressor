# ------------------------------------------------------------------------------
#
# Benchmark Experiment Script for Active Learning Regressor
#
# ------------------------------------------------------------------------------

'''
TODO:
- Adding new metrics:
    - Negative Log-Likelihood (predictive NLL) on the validation set
    - Calibration metrics: PICP@a and MPIW@a (e.g. a=95%)
    - If multi-objective: Hypervolume (HV) and Hypervolume Improvement (HVI)
    - Optimization / goal-directed metrics: improvement per query, cumulative regret, time-to-target
'''

import yaml
import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from activereg.sampling import sample_landscape
from activereg.utils import create_strict_folder
from activereg.experiment import (get_gt_dataframes, 
                                  sampling_block, 
                                  setup_data_pool,
                                  setup_ml_model,
                                  remove_evidence_from_gt,
                                  setup_experiment_variables,
                                  AcquisitionParametersGenerator)
from typing import List, Tuple
from sklearn.base import BaseEstimator

# ------------------------------------------------------------------------------------------
# FUNCTIONS


def create_benchmark_path(exp_name: str, overwrite: bool=False) -> Path:
    """Creates the benchmark experiment folder.
    Args:
        exp_name (str): Name of the experiment.
        overwrite (bool, optional): Whether to overwrite existing folder. Defaults to False.
    Returns:
        Path: The path to the created benchmark folder.
    """
    from activereg.format import BENCHMARKS_REPO
    benchmark_path = BENCHMARKS_REPO / exp_name
    create_strict_folder(path_str=str(benchmark_path), overwrite=overwrite)
    return benchmark_path


def process_dataset(df, search_var: List[str], target_var: str, split_column: str=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Processes the dataset to create training and validation sets.
    Args:
        df (pd.DataFrame): The complete dataset.
        search_var (List[str]): List of search variable names.
        target_var (str): Name of the target variable.
        split_column (str, optional): Column name to split the dataset. Defaults to None.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and validation dataframes.
    """
    if split_column is not None and split_column in df.columns:
        assert set(df[split_column].unique()).issubset({'train', 'val'}), "Invalid split column values, must be 'train' and 'val'."
        train_df = df[df[split_column] == 'train'].reset_index(drop=True)
        val_df = df[df[split_column] == 'val'].reset_index(drop=True)
    else:
        # If no split column is provided, use the entire dataset as training set
        train_df = df.copy()
        val_df = pd.DataFrame(columns=search_var + [target_var])  # Empty validation set

    return train_df, val_df


def data_scaler_setup(config: dict) -> BaseEstimator:
    """Sets up the data scaler based on the configuration.
    Args:
        config (dict): Configuration dictionary.
    Returns:
        BaseEstimator: The data scaler instance.
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    data_scaler_type = config.get('data_scaler', 'StandardScaler')
    data_scaler_params = config.get('data_scaler_params', {})

    if data_scaler_type == 'StandardScaler':
        scaler = StandardScaler(**data_scaler_params)
    elif data_scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler(**data_scaler_params)
    else:
        raise ValueError(f"Unsupported data scaler type: {data_scaler_type}")
    
    return scaler

# ------------------------------------------------------------------------------------------
# MAIN

#! The cycles will be N_CYCLES + 1 because of the last new prediction after the last acquisition
#! is used to train the last model but not for any acquisition

if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------
    # Parse the config.yaml
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML configuration file")
    parser.add_argument("-r", "--repetitions", type=int, default=1, help="Number of repetitions for the experiment")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # --------------------------------------------------------------------------------
    # SETTING UP THE EXPERIMENT
    N_REPS = args.repetitions

    (EXP_NAME, 
     ADDITIONAL_NOTES, 
     N_CYCLES, 
     INIT_BATCH, 
     INIT_SAMPLING, 
     CYCLE_SAMPLING, 
     ACQUI_PARAMS,
     SEARCH_VAR,
     TARGET_VAR) = setup_experiment_variables(config)

    ACQUI_PROTOCOL = config.get('acquisition_protocol', None)

    print(f"Experiment Name: {EXP_NAME}")
    print(f"Search Variables: {SEARCH_VAR}")
    print(f"Target Variable: {TARGET_VAR}")
    print(f"Initial Sampling: {INIT_SAMPLING} with {INIT_BATCH} points")
    print(f"Cycle Sampling: {CYCLE_SAMPLING} for {N_CYCLES} cycles")

    # Get ground truth and evidence dataframes
    gt_df, evidence_df = get_gt_dataframes(config)

    # Create benchmark paths and set up dataframes (pool_df -> complete search space)
    BENCHMARK_PATH = create_benchmark_path(exp_name=EXP_NAME, overwrite=True)

    # Process the gt dataframe dividing the test from the training set (-> pool dataset)
    pool_df, val_df = process_dataset(gt_df, search_var=SEARCH_VAR, target_var=TARGET_VAR, split_column='set')
    pool_target_landscape = pool_df[TARGET_VAR].to_numpy().ravel()
    val_target_landscape = val_df[TARGET_VAR].to_numpy().ravel() if not val_df.empty else None

    # Save a copy of the config file in the benchmark folder
    config_save_path = BENCHMARK_PATH / 'config.yaml'
    with open(config_save_path, 'w') as file:
        yaml.dump(config, file)

    # Landscape Penalization if defined in the config file
    landscape_penalization = config.get('landscape_penalization', None)
    if landscape_penalization is not None:
        pen_radius = landscape_penalization.get('radius', None)
        pen_strength = landscape_penalization.get('strength', None)
        print(f"Landscape penalization activated with radius {pen_radius} and strength {pen_strength}.")

    # Set up and apply the data scaler on the complete search space
    data_scaler = data_scaler_setup(config)
    X_pool, data_scaler = setup_data_pool(df=pool_df, search_var=SEARCH_VAR, scaler=data_scaler)
    scaler_path = BENCHMARK_PATH / 'pool_scaler.joblib'
    joblib.dump(data_scaler, scaler_path)

    # Create candidates dataframe (candidates_df -> unscreened space)
    candidates_df = remove_evidence_from_gt(pool_df, evidence_df, search_vars=SEARCH_VAR)

    # Set up the model
    ML_MODEL = setup_ml_model(config)

    # Scale the validation set if available to compute metrics
    if not val_df.empty:
        X_val = data_scaler.transform(val_df[SEARCH_VAR].to_numpy())
    else:
        X_val = None

    # --------------------------------------------------------------------------------
    # RUN THE BENCHMARK EXPERIMENT

    # Contains data that summarizes the benchmark experiment
    benchmark_data = []
    # Contains data of all the training points acquired during the experiment with additional metadata
    train_points_data = []
    # Predefined acquisition modes for tracking the source of acquisition of each point
    predefined_acquisition_modes = []

    # Arrays to store the predicted and uncertainty landscapes for all repetitions and cycles
    pool_pred_landscape_array = np.full((N_REPS, N_CYCLES+1, X_pool.shape[0]), np.nan)
    pool_unc_landscape_array = np.full((N_REPS, N_CYCLES+1, X_pool.shape[0]), np.nan)

    val_pred_landscape_array = np.full((N_REPS, N_CYCLES+1, X_val.shape[0]), np.nan) if X_val is not None else None
    val_unc_landscape_array = np.full((N_REPS, N_CYCLES+1, X_val.shape[0]), np.nan) if X_val is not None else None

    # Init the acquisition parameters generator
    acqui_param_gen = AcquisitionParametersGenerator(
        acquisition_params=ACQUI_PARAMS,
        acquisition_protocol=ACQUI_PROTOCOL
    )

    # --------------------------------------------------------------------------------
    # EXPERIMENT REPETITIONS
    for rep in range(N_REPS):

        # --------------------------------------------------------------------------------
        # CYCLE 0 - INITIAL SAMPLING
        X_candidates = data_scaler.transform(candidates_df[SEARCH_VAR].to_numpy())
        y_candidates = candidates_df[TARGET_VAR].to_numpy()

        if evidence_df is not None:
            X_train = data_scaler.transform(evidence_df[SEARCH_VAR].to_numpy())
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
        
        for i in range(len(X_train)):
            train_points_data.append({
                **{col : X_train[i,j] for j,col in enumerate(SEARCH_VAR)},
                "cycle": 0,
                "repetition": rep+1,
                "acquisition_source": INIT_SAMPLING if evidence_df is None else "evidence",
                "y_value": y_train[i][0]
            })
        # END of initial sampling
        # --------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------
        # CYCLES 1 -> N_CYCLES
        for cycle in tqdm(range(N_CYCLES), desc=f"Repetition {rep+1}/{N_REPS}"):

            # Get the acquisition parameters for the current cycle
            cycle_acqui_params = acqui_param_gen.get_params_for_cycle(cycle)
            for acp in cycle_acqui_params:
                predefined_acquisition_modes.extend([acp['acquisition_mode']] * acp['n_points'])

            # Compute the best screened value and train the model
            y_best = np.max(y_train)
            ML_MODEL.train(X_train, y_train)
            _, y_pred_pool, y_unc_pool = ML_MODEL.predict(X_pool)
            _, y_pred_val, y_unc_val = ML_MODEL.predict(X_val) if X_val is not None else (None, None, None)

            # Sample from the candidates
            sampled_indexes, landscape = sampling_block(
                X_candidates=X_candidates,
                X_train=X_train,
                y_best=y_best,
                ml_model=ML_MODEL,
                acquisition_params=cycle_acqui_params,
                sampling_mode=CYCLE_SAMPLING,
                penalization_params=(pen_radius, pen_strength) if landscape_penalization is not None else None
            )

            # Store the predicted and uncertainty landscapes for the current cycle
            pool_pred_landscape_array[rep, cycle] = y_pred_pool
            pool_unc_landscape_array[rep, cycle] = y_unc_pool

            if X_val is not None:
                val_pred_landscape_array[rep, cycle] = y_pred_val
                val_unc_landscape_array[rep, cycle] = y_unc_val

            benchmark_data.append({
                "repetition": rep+1,
                "cycle": cycle+1,
                "y_best_screened": y_best,
                "y_best_predicted_pool": np.max(y_pred_pool),
                "y_best_predicted_val": np.max(y_pred_val) if y_pred_val is not None else np.nan,
                "rmse_vs_gt_pool": np.sqrt(mean_squared_error(pool_target_landscape, pool_pred_landscape_array[rep, cycle])),
                "rmse_vs_gt_val": np.sqrt(mean_squared_error(val_target_landscape, y_pred_val)) if y_pred_val is not None else np.nan,
                "mae_vs_gt_pool": mean_absolute_error(pool_target_landscape, pool_pred_landscape_array[rep, cycle]),
                "mae_vs_gt_val": mean_absolute_error(val_target_landscape, y_pred_val) if y_pred_val is not None else np.nan,
                "rmse_vs_prev": np.sqrt(mean_squared_error(pool_pred_landscape_array[rep, cycle-1], pool_pred_landscape_array[rep, cycle])) if cycle > 0 else np.nan,
                "mae_vs_prev": mean_absolute_error(pool_pred_landscape_array[rep, cycle-1], pool_pred_landscape_array[rep, cycle]) if cycle > 0 else np.nan,
                # "nll": ML_MODEL.model.log_marginal_likelihood().astype(float) if config.get('ml_model', {}) == 'GPR' else np.nan
                # TODO: add other metrics
            })

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
            # END of cycles
            # -------------------------------------------------------------------------------- 

        # --------------------------------------------------------------------------------
        # Compute the final landscape after the last cycle with the full training set
        # This is important for evaluating the final model performance without acquisition
        # of new points (the tot. num of cycles is N_CYCLES+1 including the initial sampling)
        y_best = np.max(y_train)
        ML_MODEL.train(X_train, y_train)
        _, y_pred_pool, y_unc_pool = ML_MODEL.predict(X_pool)
        _, y_pred_val, y_unc_val = ML_MODEL.predict(X_val) if X_val is not None else (None, None, None)

        pool_pred_landscape_array[rep, N_CYCLES] = y_pred_pool
        pool_unc_landscape_array[rep, N_CYCLES] = y_unc_pool

        if X_val is not None:
            val_pred_landscape_array[rep, N_CYCLES] = y_pred_val
            val_unc_landscape_array[rep, N_CYCLES] = y_unc_val

        benchmark_data.append({
            "repetition": rep+1,
            "cycle": N_CYCLES+1,
            "y_best_screened": y_best,
            "y_best_predicted_pool": np.max(y_pred_pool),
            "y_best_predicted_val": np.max(y_pred_val) if y_pred_val is not None else np.nan,
            "rmse_vs_gt_pool": np.sqrt(mean_squared_error(pool_target_landscape, pool_pred_landscape_array[rep, cycle])),
            "rmse_vs_gt_val": np.sqrt(mean_squared_error(val_target_landscape, y_pred_val)) if y_pred_val is not None else np.nan,
            "mae_vs_gt_pool": mean_absolute_error(pool_target_landscape, pool_pred_landscape_array[rep, cycle]),
            "mae_vs_gt_val": mean_absolute_error(val_target_landscape, y_pred_val) if y_pred_val is not None else np.nan,
            "rmse_vs_prev": np.sqrt(mean_squared_error(pool_pred_landscape_array[rep, cycle-1], pool_pred_landscape_array[rep, cycle])) if cycle > 0 else np.nan,
            "mae_vs_prev": mean_absolute_error(pool_pred_landscape_array[rep, cycle-1], pool_pred_landscape_array[rep, cycle]) if cycle > 0 else np.nan,
            # "nll": ML_MODEL.model.log_marginal_likelihood().astype(float) if config.get('ml_model', {}) == 'GPR' else np.nan
            # TODO: add other metrics
        })
        # END of repetition
        # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Save benchmark data to CSV
    benchmark_df = pd.DataFrame(benchmark_data)
    benchmark_df.to_csv(BENCHMARK_PATH / 'benchmark_data.csv', index=False)

    train_points_df = pd.DataFrame(train_points_data)
    train_points_df.to_csv(BENCHMARK_PATH / 'train_points_data.csv', index=False)

    np.save(BENCHMARK_PATH / 'pool_predicted_landscape.npy', pool_pred_landscape_array)
    np.save(BENCHMARK_PATH / 'pool_uncertainty_landscape.npy', pool_unc_landscape_array)

    if X_val is not None:
        np.save(BENCHMARK_PATH / 'val_predicted_landscape.npy', val_pred_landscape_array)
        np.save(BENCHMARK_PATH / 'val_uncertainty_landscape.npy', val_unc_landscape_array)

# END of the experiment
# --------------------------------------------------------------------------------
