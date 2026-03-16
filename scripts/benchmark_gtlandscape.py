# ------------------------------------------------------------------------------
#
# Benchmark Experiment Script for Active Learning Regressor
#
# ------------------------------------------------------------------------------

'''
TODO:
- Adding new metrics:
    x Negative Log-Likelihood (predictive NLL) on the validation set
    x Calibration metrics: PICP@a and MPIW@a (e.g. a=95%)
    - If multi-objective: Hypervolume (HV) and Hypervolume Improvement (HVI)
    - Optimization / goal-directed metrics: improvement per query, cumulative regret, time-to-target

x Refactor the metric computation into a dedicated function for clarity and reusability.

- Refine the space sampling as the function in high dimensional spaces can be tricky to handle
  due to low density of points in the area to optimize.
    - Adaptive LHS sampling to improve POOL coverage

- Remove saving of the landscapes at each cycle to reduce memory usage and disk space.
'''

import yaml
import joblib
import argparse
import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from activereg.data import DatasetGenerator
from activereg.metrics import evaluate_cycle_metrics
from activereg.sampling import sample_landscape
from activereg.utils import create_strict_folder
from activereg.adaptiveRefinement import (select_centers_from_batch,
                                          pointwise_hypercube_refinement,
                                          filter_refined_additions)
from activereg.experiment import (get_gt_dataframes, 
                                  sampling_block, 
                                  setup_data_pool,
                                  setup_ml_model,
                                  remove_evidence_from_gt,
                                  setup_experiment_variables,
                                  AcquisitionParametersGenerator)
from typing import List, Tuple
from sklearn.base import BaseEstimator

# --------------------------------------------------------------------------------
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


def extract_landscape_bounds(df: pd.DataFrame, search_var: List[str]) -> np.ndarray:
    """Extracts the bounds of the search space from the dataset.
    Args:
        df (pd.DataFrame): The complete dataset.
        search_var (List[str]): List of search variable names.
    Returns:
        np.ndarray: Array of shape (n_variables, 2) containing the lower and upper bounds for each variable.
    """
    bounds = []
    for var in search_var:
        var_min = df[var].min()
        var_max = df[var].max()
        bounds.append((var_min, var_max))
    return np.array(bounds)


def data_scaler_setup(data_scaler_type: str, data_scaler_params: dict) -> BaseEstimator:
    """Sets up the data scaler based on the configuration.
    Args:
        data_scaler_type (str): Type of the data scaler.
        data_scaler_params (dict): Parameters for the data scaler.
    Returns:
        BaseEstimator: The data scaler instance.
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    if data_scaler_type == 'StandardScaler':
        scaler = StandardScaler(**data_scaler_params)
    elif data_scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler(**data_scaler_params)
    else:
        raise ValueError(f"Unsupported data scaler type: {data_scaler_type}")
    
    return scaler


def aggregate_configurations(benchmark_config: dict, model_config: dict, acquisition_config: dict, target_function_config: dict) -> dict:
    """Aggregates all configurations into a single dictionary for easier access/output.
    Args:
        benchmark_config (dict): Benchmark configuration dictionary.
        model_config (dict): Model configuration dictionary.
        acquisition_config (dict): Acquisition mode settings dictionary.
        target_function_config (dict): Target function configuration dictionary.
    Returns:
        dict: Aggregated configuration dictionary.
    """
    config = {
        "benchmark": benchmark_config,
        "model": model_config,
        "acquisition": acquisition_config,
        "target_function": target_function_config
    }
    return config

# --------------------------------------------------------------------------------
# MAIN EXPERIMENT SCRIPT
if __name__ == '__main__':
    # --------------------------------------------------------------------------------
    # Parse the config.yaml
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("-bc", "--benchmark_config", required=True, help="Path to the YAML configuration file")
    parser.add_argument("-mc", "--model_config", required=True, help="Path to the YAML configuration file for the ML model and parameters")
    parser.add_argument("-acqmodes", "--acquisition_mode_settings", required=True, help="Acquisition mode settings for the experiment")
    parser.add_argument("-r", "--repetitions", type=int, default=1, help="Number of repetitions for the experiment")
    parser.add_argument("--rerun", action='store_true', help="Rerun the experiment even if the benchmark folder exists")
    args = parser.parse_args()

    # --------------------------------------------------------------------------------
    # LOAD CONFIGURATIONS FILES

    # Load the benchmark configuration
    with open(args.benchmark_config, "r") as file:
        benchmark_config = yaml.safe_load(file)

    # Load the model configuration
    with open(args.model_config, "r") as file:
        model_config = yaml.safe_load(file)

    # Load the acquisition mode settings
    with open(args.acquisition_mode_settings, "r") as file:
        acquisition_config = yaml.safe_load(file)
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # SET RANDOM SEEDS
    seed = benchmark_config.get('SEED', None)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # SETTING UP THE EXPERIMENT
    N_REPS = args.repetitions

    (EXP_NAME,           # Experiment name
     ADDITIONAL_NOTES,   # Additional notes for the experiment
     N_CYCLES,           # Number of active learning cycles
     INIT_BATCH,         # Initial batch size
     INIT_SAMPLING,      # Initial sampling method
     SEARCH_VAR,         # Search space variables
     TARGET_VAR) = setup_experiment_variables(benchmark_config)

    # Create benchmark paths and set up dataframes (pool_df -> complete search space)
    BENCHMARK_PATH = create_benchmark_path(exp_name=EXP_NAME, overwrite=True if args.rerun else False)

    # Save a copy of the config file in the benchmark folder
    config_save_path = BENCHMARK_PATH / 'experiment_config.yaml'
    config_all = aggregate_configurations(benchmark_config, model_config, acquisition_config, target_function_config=None)
    with open(config_save_path, 'w') as file:
        yaml.dump(config_all, file)

    # Scaler path
    scaler_path = BENCHMARK_PATH / 'pool_scaler.joblib'

    print(f"Experiment Name: {EXP_NAME}\n->\t{BENCHMARK_PATH}")
    print(f"Search Variables: {SEARCH_VAR}")
    print(f"Target Variable: {TARGET_VAR}")
    print(f"Initial Sampling: {INIT_SAMPLING} with {INIT_BATCH} points")
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # MODEL SECTION
    ml_model_type = model_config.get('ml_model', None)
    ml_model_params = model_config.get('model_parameters', {})
    assert ml_model_type is not None, "ML model type must be specified in the model_config file."
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # ACQUISITION MODES SETTINGS
    acquisition_parameters = acquisition_config.get('acquisition_parameters', None)
    acquisition_protocol = benchmark_config.get('acquisition_protocol', None)
    assert acquisition_parameters is not None, "Acquisition parameters must be specified in the acquisition_mode_settings file."
    assert acquisition_protocol is not None, "Acquisition protocol must be specified in the benchmark_config file."

    # Init the acquisition parameters generator
    acqui_param_gen = AcquisitionParametersGenerator(
        acquisition_params=acquisition_parameters,
        acquisition_protocol=acquisition_protocol
    )
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # GET GT DATAFRAME
    # read pre-defined ground truth and evidence dataframes
    gt_file = benchmark_config.get('ground_truth_file', None)
    evidence_file = benchmark_config.get('experiment_evidence', None)

    gt_df, evidence_df = None, None

    if gt_file is not None:
        gt_df, evidence_df = get_gt_dataframes(gt_file, evidence_file)
        print(f"Ground truth dataset loaded from -> {gt_file} with {len(gt_df)} points.")
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # PREPARE THE DATASETS
    pool_df, val_df = process_dataset(
        df=gt_df,
        search_var=SEARCH_VAR,
        target_var=TARGET_VAR,
        split_column='set'  # the GT file can optionally contain a 'set' column to split the dataset into train and validation, if not present the entire GT will be used as training
    )

    gt_pool_bounds = extract_landscape_bounds(df=pool_df, search_var=SEARCH_VAR)

    # Save the generated POOL dataset for reference
    pool_save_path = BENCHMARK_PATH / 'pool_dataset.csv'
    pool_df.to_csv(pool_save_path, index=False)
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # PREPARE DATAFRAMES AND SCALER
    # train dataset -> pool_df : from which the candidates will be sampled and refined during the experiment
    # gt dataset -> val_df : used as holdout validation set for metrics computation during the experiment

    # Extract the target landscapes for metrics computation
    pool_target_landscape = pool_df[TARGET_VAR].to_numpy().ravel()
    val_target_landscape = val_df[TARGET_VAR].to_numpy().ravel() if not val_df.empty else None

    # Set up and apply the data scaler on the complete search space
    data_scaler_type = benchmark_config.get('data_scaler', None)
    data_scaler_params = benchmark_config.get('scaler_params') or {}
    data_scaler = data_scaler_setup(data_scaler_type, data_scaler_params)
    X_pool, data_scaler = setup_data_pool(df=pool_df, search_var=SEARCH_VAR, scaler=data_scaler)
    joblib.dump(data_scaler, scaler_path)

    # Create candidates dataframe (candidates_df -> unscreened space)
    candidates_df = remove_evidence_from_gt(gt=pool_df, evidence=evidence_df, search_vars=SEARCH_VAR)

    # Scale the validation set if available to compute metrics
    X_val = data_scaler.transform(val_df[SEARCH_VAR].to_numpy()) if not val_df.empty else None
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # LANDSCAPE ADDITIONAL PARAMETERS AND SETUP
    # Set up landscape penalization parameters
    landscape_penalization = benchmark_config.get('landscape_penalization', None)
    if landscape_penalization is not None:
        pen_radius = landscape_penalization.get('radius', None)
        pen_strength = landscape_penalization.get('strength', None)
        print(f"Landscape penalization activated with radius {pen_radius} and strength {pen_strength}.")
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # BATCH SELECTION STRATEGY SETUP
    batch_selection_config = benchmark_config.get('batch_selection', None)
    if batch_selection_config is not None:
        batch_selection_method = batch_selection_config.get('method', 'highest_landscape')
        batch_selection_params = batch_selection_config.get('method_params', {})
        print(f"Batch selection strategy: {batch_selection_method} with params: {batch_selection_params}")
    else:
        raise ValueError("Batch selection configuration must be provided in the config file, with at least the method defined.")
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # ADAPTIVE REFINEMENT SETUP
    # Set up the space refinement sampler, it will only refine the X_pool
    refine_generator = None
    refinement_config = benchmark_config.get('adaptive_refinement', None)

    if refinement_config is not None:
        refine_method = refinement_config.get('method', 'lhs')
        refine_noise_std = refinement_config.get('refinement_noise_std', 0.0)

        refine_generator = DatasetGenerator(
            n_dimensions=gt_pool_bounds.shape[0],
            bounds=gt_pool_bounds,
            seed=seed
        )

        refine_step = refinement_config.get('refinement_step', 20)
        refine_centroids = refinement_config.get('refinement_centroids', 5)
        refine_batch_size = refinement_config.get('refinement_batch_size', 500)
        half_side_length_strategy = refinement_config.get('half_side_length_strategy', 'density')
        half_side_length_strategy_params = refinement_config.get('half_side_length_strategy_params', {})
        refinement_filtering_min_distance = refinement_config.get('refinement_filtering_min_distance', 0.0001)

        print(f"Adaptive refinement activated every {refine_step} points acquired, "
              f"with {refine_centroids} centroids and adding {refine_batch_size} points per centroids.")
        
        # DataFrame to collect all refinement points added during the experiment (for output purposes)
        collective_refinement_df = pd.DataFrame()
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # RUN THE BENCHMARK EXPERIMENT

    # Contains data that summarizes the benchmark experiment
    benchmark_data = []
    # Contains data of all the training points acquired during the experiment with additional metadata
    train_points_data = []

    # --------------------------------------------------------------------------------
    # EXPERIMENT REPETITIONS
    for rep in range(N_REPS):

        # --------------------------------------------------------------------------------
        # SET UP THE MODEL
        ML_MODEL = setup_ml_model(ml_model_type=ml_model_type, ml_model_params=ml_model_params)
        # --------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------
        print(f"\n--- Starting repetition {rep+1}/{N_REPS} ---\n")
        # Refinement step tracking per repetition
        if refine_generator is not None:
            refine_counter = 0
            next_refine_target = refine_step

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
            # Predefined acquisition modes for tracking the source of acquisition of each point
            predefined_acquisition_modes = []
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
                y_train=y_train,
                ml_model=ML_MODEL,
                acquisition_params=cycle_acqui_params,
                batch_selection_method=batch_selection_method,
                batch_selection_params=batch_selection_params,
                penalization_params=(pen_radius, pen_strength) if landscape_penalization is not None else None
            )

            # Store benchmark data for the current cycle
            cycle_data_dict = {
                "repetition": rep+1,
                "cycle": cycle+1,
                "y_best_screened": y_best,
            }
            cycle_metrics_dict = evaluate_cycle_metrics(
                y_true_pool=pool_target_landscape,
                y_pred_pool=y_pred_pool,
                y_true_val=val_target_landscape,
                y_pred_val=y_pred_val,
                y_uncertainty_val=y_unc_val
            )
            cycle_data_dict.update(cycle_metrics_dict)
            benchmark_data.append(cycle_data_dict)

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
            # Adaptive refinement of candidates pool if defined in the config file
            # TODO: consider refactoring into a class or function for better readability and modularity
            if refine_generator is not None:

                # TODO: check for possible inconsistencies in the countig of acquired points
                n_points_acquired = (cycle + 1) * sum([acp['n_points'] for acp in cycle_acqui_params]) + (INIT_BATCH if evidence_df is None else 0)
                if n_points_acquired >= next_refine_target:
                    refine_counter += 1

                    # From the current `sampled_indexes` select the hypercube centres
                    X_centroids_refinement = select_centers_from_batch(
                        candidate_points=X_train[-next_refine_target:],  # select the last acquired points as candidates for centroids selection
                        n_centers=refine_centroids,
                        min_centers=2
                    )

                    # Generate new candidates around each centroid
                    refined_df = pd.DataFrame()
                    for centroid in X_centroids_refinement:
                        refinement_df = pointwise_hypercube_refinement(
                            refine_generator=refine_generator,
                            design_bounds=gt_pool_bounds,
                            design_dimensions=gt_pool_bounds.shape[0],
                            refine_centroid=np.reshape(centroid, (1, -1)),  # reshape to (1, D)
                            pool_scaled=X_pool,
                            half_side_length_strategy=half_side_length_strategy,
                            hsl_strategy_params=half_side_length_strategy_params,
                            n_points=refine_batch_size,
                            refine_noise_std=refine_noise_std,
                            scaler=data_scaler,
                            refine_function=None,  # the refinement is performed only on the pool points
                            refine_method=refine_method
                        )
                        # CAREFUL: the refinement_df is in the original space (unscaled)
                        refined_df = pd.concat([refined_df, refinement_df], ignore_index=True)

                    # Separate refined_pool and refined_validation sets
                    # a) Pool set
                    X_pool_refined = refined_df[SEARCH_VAR].to_numpy()
                    X_refined_scaled = data_scaler.transform(X_pool_refined)
                    y_pool_refined = refined_df[TARGET_VAR].to_numpy()

                    # Filter out already existing points in the candidates pool
                    X_pool_refined_filtered, filtered_indexes = filter_refined_additions(
                        X_addition=X_refined_scaled,
                        X_existing=X_pool,
                        min_distance=refinement_filtering_min_distance,
                        filter_internal=True
                    )
                    y_refined_filtered = y_pool_refined[filtered_indexes]

                    # Append the refined and filtered points to the candidates pool
                    X_candidates = np.vstack((X_candidates, X_pool_refined_filtered))
                    y_candidates = np.concatenate((y_candidates, y_refined_filtered))

                    # Append the refined and filtered points to the pool set
                    # Careful: the target from the pool (and validation) set are (Npoints, ) shaped arrays
                    X_pool = np.vstack((X_pool, X_pool_refined_filtered))
                    pool_target_landscape = np.concatenate((pool_target_landscape, y_refined_filtered.ravel()))

                    # Update the next refinement target
                    print(f"-> Adaptive refinement step {refine_counter} at {n_points_acquired} points acquired, "
                          f"added {len(y_refined_filtered)}/{refine_centroids*refine_batch_size} points.")
                    
                    # Retain only the filtered points in the refined_df for output purposes
                    refined_df = refined_df.iloc[filtered_indexes]

                    # Collect all refinement points added during the experiment
                    # Add a column to identify the refinement step
                    refined_df['refinement_step'] = [refine_counter] * len(refined_df)
                    collective_refinement_df = pd.concat([collective_refinement_df, refined_df], ignore_index=True)
                    collective_refinement_save_path = BENCHMARK_PATH / f'collective_refinement_points_rep{rep+1}.csv'
                    collective_refinement_df.to_csv(collective_refinement_save_path, index=False)

                    next_refine_target += refine_step

            # END of Adaptive refinement
            # --------------------------------------------------------------------------------
        
        print(f"\n--- End of {rep+1}/{N_REPS} ---\n")
        # End of repetition
        # Resetting the pool and candidates sets for the next repetition if the refinement took place
        if refine_generator is not None:
            # Reset the pool set; the candidates set is derived from the candidates_df so it is not affected by the refinement
            X_pool = data_scaler.transform(pool_df[SEARCH_VAR].to_numpy())
            pool_target_landscape = pool_df[TARGET_VAR].to_numpy().ravel()

        # Save the model at the end of the repetition
        model_save_path = BENCHMARK_PATH / f'ml_model_rep{rep+1}.joblib'
        joblib.dump(ML_MODEL, model_save_path)

        # END of repetition
        # --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Save benchmark data to CSV files
    benchmark_df = pd.DataFrame(benchmark_data)
    benchmark_df.to_csv(BENCHMARK_PATH / 'benchmark_data.csv', index=False)

    train_points_df = pd.DataFrame(train_points_data)
    train_points_df.to_csv(BENCHMARK_PATH / 'train_points_data.csv', index=False)

# END of the experiment
# --------------------------------------------------------------------------------
