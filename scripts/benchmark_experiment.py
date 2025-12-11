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
from activereg.data import (DatasetGenerator,
                            compute_reference_distance,
                            pointwise_hypercube_refinement,
                            filter_refined_additions)
from activereg.metrics import evaluate_cycle_metrics
from activereg.sampling import sample_landscape
from activereg.utils import create_strict_folder
from activereg.acquisition import highest_landscape_selection
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
# FUNCTIONAL FORMS (USED IN THE BENCHMARKS STUDY)

from botorch.test_functions import Hartmann, Ackley, StyblinskiTang

FUNCTIONS_DICT = {
    "Hartmann": Hartmann,
    "Ackley": Ackley,
    "StyblinskiTang" : StyblinskiTang
}

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

# --------------------------------------------------------------------------------
# MAIN EXPERIMENT SCRIPT
if __name__ == '__main__':
    # --------------------------------------------------------------------------------
    # Parse the config.yaml
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML configuration file")
    parser.add_argument("-r", "--repetitions", type=int, default=1, help="Number of repetitions for the experiment")
    parser.add_argument("--rerun", action='store_true', help="Rerun the experiment even if the benchmark folder exists")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # --------------------------------------------------------------------------------
    # SET RANDOM SEEDS
    seed = config.get('SEED', None)
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
     ACQUI_PARAMS,       # Acquisition function parameters
     SEARCH_VAR,         # Search space variables
     TARGET_VAR) = setup_experiment_variables(config)

    # Create benchmark paths and set up dataframes (pool_df -> complete search space)
    BENCHMARK_PATH = create_benchmark_path(exp_name=EXP_NAME, overwrite=True if args.rerun else False)

    # Save a copy of the config file in the benchmark folder
    config_save_path = BENCHMARK_PATH / 'config.yaml'
    with open(config_save_path, 'w') as file:
        yaml.dump(config, file)

    # Scaler path
    scaler_path = BENCHMARK_PATH / 'pool_scaler.joblib'

    # Acquisition protocol
    ACQUI_PROTOCOL = config.get('acquisition_protocol', None)

    print(f"Experiment Name: {EXP_NAME}\n->\t{BENCHMARK_PATH}")
    print(f"Search Variables: {SEARCH_VAR}")
    print(f"Target Variable: {TARGET_VAR}")
    print(f"Initial Sampling: {INIT_SAMPLING} with {INIT_BATCH} points")

    # --------------------------------------------------------------------------------
    # GET/CREATE GT DATAFRAMES
    # read pre-defined ground truth and evidence dataframes
    gt_file = config.get('ground_truth_file', None)
    evidence_file = config.get('experiment_evidence', None)

    if gt_file is not None:
        gt_df, evidence_df = get_gt_dataframes(gt_file, evidence_file)

    # Generate synthetic dataset based on functional forms
    elif gt_file is None:
        gt_config = config.get('ground_truth_parameters', None)
        assert gt_config is not None, "Ground truth parameters must be provided in the config file."

        function_name = gt_config.pop('function', None)
        function_dim = gt_config.pop('n_dimensions', 3)
        function_bounds = np.array(gt_config.pop('bounds', [[0, 1]] * function_dim))
        function_negate = gt_config.pop('negate', False)

        if function_name not in FUNCTIONS_DICT:
            raise ValueError(f"Function '{function_name}' is not defined. Choose from {list(FUNCTIONS_DICT.keys())}.")
        function_class = FUNCTIONS_DICT[function_name]

        dataset_generator = DatasetGenerator(
            n_dimensions=function_dim,
            bounds=function_bounds,
            negate=function_negate,
            seed=seed
        )
        
        sampling_method = gt_config.pop('method', 'lhs')
        gt_df = dataset_generator.generate_dataset(
            function=function_class,
            method=sampling_method,
            **gt_config
        )
        evidence_df = None
        print(f"Synthetic ground truth dataset generated using the {function_name} function in {function_dim}D.")

        # Save the generated ground truth dataset for reference
        gt_save_path = BENCHMARK_PATH / 'ground_truth_dataset.csv'
        gt_df.to_csv(gt_save_path, index=False)
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # PREPARE DATAFRAMES AND SCALER
    # Process the gt dataframe dividing the test from the training set (-> pool dataset)
    pool_df, val_df = process_dataset(gt_df, search_var=SEARCH_VAR, target_var=TARGET_VAR, split_column='set')

    # Extract the target (GT) landscapes for metrics computation
    pool_target_landscape = pool_df[TARGET_VAR].to_numpy().ravel()
    val_target_landscape = val_df[TARGET_VAR].to_numpy().ravel() if not val_df.empty else None

    # Set up and apply the data scaler on the complete search space
    data_scaler = data_scaler_setup(config)
    X_pool, data_scaler = setup_data_pool(df=pool_df, search_var=SEARCH_VAR, scaler=data_scaler)
    X_pool_reference_distance = compute_reference_distance(X=X_pool, method="median_nn")
    joblib.dump(data_scaler, scaler_path)

    # Create candidates dataframe (candidates_df -> unscreened space)
    candidates_df = remove_evidence_from_gt(pool_df, evidence_df, search_vars=SEARCH_VAR)

    # Scale the validation set if available to compute metrics
    X_val = data_scaler.transform(val_df[SEARCH_VAR].to_numpy()) if not val_df.empty else None
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # LANDSCAPE ADDITIONAL PARAMETERS AND SETUP
    # Set up landscape penalization parameters
    landscape_penalization = config.get('landscape_penalization', None)
    if landscape_penalization is not None:
        pen_radius = landscape_penalization.get('radius', None)
        pen_strength = landscape_penalization.get('strength', None)
        print(f"Landscape penalization activated with radius {pen_radius} and strength {pen_strength}.")

    # --------------------------------------------------------------------------------
    # BATCH SELECTION STRATEGY SETUP
    batch_selection_config = config.get('batch_selection', None)
    if batch_selection_config is not None:
        batch_selection_method = batch_selection_config.get('method', 'highest_landscape')
        batch_selection_params = batch_selection_config.get('method_params', None)
        print(f"Batch selection strategy: {batch_selection_method} with params: {batch_selection_params}")
    else:
        raise ValueError("Batch selection configuration must be provided in the config file, with at least the method defined.")

    # --------------------------------------------------------------------------------
    # ADAPTIVE REFINEMENT SETUP
    # Set up the space refinement sampler
    refinement_config = config.get('adaptive_refinement', None)
    if refinement_config is not None:
        refine_function_name = refinement_config.get('function', None)
        refine_function_dim = refinement_config.get('n_dimensions', None)
        refine_function_bounds = np.array(refinement_config.get('bounds', [[0, 1]] * refine_function_dim))
        refine_method = refinement_config.get('method', 'lhs')
        refine_noise_std = refinement_config.get('refinement_noise_std', 0.0)
        refine_function_negate = refinement_config.get('negate', False)

        if gt_file is None:
            assert refine_function_name == function_name, "Refinement function must be the same as the defined ground truth function."
            assert refine_function_dim == function_dim, "Refinement function dimension must be the same as the defined ground truth function."
            
            assert refine_function_negate == function_negate, "Refinement function negate parameter must be the same as the defined ground truth function."

        elif gt_file is not None:
            print(f"Adaptive refinement is set with negate={refine_function_negate}. "
                  f"Check that these parameters are consistent with the ground truth function settings.")
            if refine_function_name not in FUNCTIONS_DICT:
                raise ValueError(f"Refinement function '{refine_function_name}' is not defined. Choose from {list(FUNCTIONS_DICT.keys())}.")

        refine_generator = DatasetGenerator(
            n_dimensions=refine_function_dim,
            bounds=refine_function_bounds,
            negate=refine_function_negate,
            seed=seed
        )

        refine_function = FUNCTIONS_DICT[refine_function_name]
        refine_step = refinement_config.get('refinement_step', 20)
        refine_centroids = refinement_config.get('refinement_centroids', 5)
        refine_batch_size = refinement_config.get('refinement_batch_size', 500)
        centroids_selection_method = refinement_config.get('centroids_selection_method', 'fps')

        print(f"Adaptive refinement activated every {refine_step} points acquired, "
              f"with {refine_centroids} centroids and adding {refine_batch_size} points per centroids.")
        
        # DataFrame to collect all refinement points added during the experiment (for output purposes)
        collective_refinement_df = pd.DataFrame()

    else:
        refine_generator = None

    # --------------------------------------------------------------------------------
    # RUN THE BENCHMARK EXPERIMENT

    # Contains data that summarizes the benchmark experiment
    benchmark_data = []
    # Contains data of all the training points acquired during the experiment with additional metadata
    train_points_data = []

    # Init the acquisition parameters generator
    acqui_param_gen = AcquisitionParametersGenerator(
        acquisition_params=ACQUI_PARAMS,
        acquisition_protocol=ACQUI_PROTOCOL
    )

    # --------------------------------------------------------------------------------
    # EXPERIMENT REPETITIONS
    for rep in range(N_REPS):

        # --------------------------------------------------------------------------------
        # SET UP THE MODEL
        ML_MODEL = setup_ml_model(config)
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
            if refine_generator is not None:

                # TODO: check for possible inconsistencies in the countig of acquired points
                n_points_acquired = (cycle + 1) * sum([acp['n_points'] for acp in cycle_acqui_params]) + (INIT_BATCH if evidence_df is None else 0)
                if n_points_acquired >= next_refine_target:
                    refine_counter += 1

                    # Compute the reference distance on the training set and determine the half–side-length of the hypercube
                    nn_ref_distance = compute_reference_distance(X=X_train, method="median_nn")
                    half_side_hyperc_length = nn_ref_distance

                    # Accumulate the seed (size of refine_step) of the refinement points from the top predicted candidates so far
                    highest_prediction_ndx = highest_landscape_selection(
                        landscape=y_pred_pool,
                        percentile=90,
                        min_points=100,
                        max_points=1000
                    )
                    X_seed_refinement = X_pool[highest_prediction_ndx]  # Based on highest predicted values in the pool

                    # Compress the seed points to just the centroids via Farthest Point Sampling (FPS)
                    centroids_ndx = sample_landscape(
                        X_landscape=X_seed_refinement,
                        n_points=refine_centroids,
                        sampling_mode=centroids_selection_method
                    )
                    X_centroids_refinement = X_seed_refinement[centroids_ndx]

                    # Generate new candidates around each centroid
                    refined_df = pd.DataFrame()
                    for centroid in X_centroids_refinement:
                        refinement_df = pointwise_hypercube_refinement(
                            refine_generator=refine_generator,
                            design_bounds=refine_function_bounds,
                            design_dimensions=refine_function_dim,
                            refine_centroid=np.reshape(centroid, (1, -1)),  # reshape to (1, D)
                            half_side_length=half_side_hyperc_length,
                            n_points=refine_batch_size,
                            refine_noise_std=refine_noise_std,
                            scaler=data_scaler,
                            refine_function=refine_function,
                            refine_method=refine_method
                        )
                        # CAREFUL: the refinement_df is in the original space (unscaled)
                        refined_df = pd.concat([refined_df, refinement_df], ignore_index=True)

                    # Separate refined_pool and refined_validation sets
                    # a) Pool set
                    X_pool_refined = refined_df[refined_df['set'] == 'train'][SEARCH_VAR].to_numpy()
                    X_refined_scaled = data_scaler.transform(X_pool_refined)
                    y_pool_refined = refined_df[refined_df['set'] == 'train'][TARGET_VAR].to_numpy()

                    # Filter out already existing points in the candidates pool
                    # TODO: make the min distance scaling parameter configurable or adaptive
                    X_pool_refined_filtered, filtered_indexes = filter_refined_additions(
                        X_addition=X_refined_scaled,
                        X_existing=X_pool,
                        min_distance=X_pool_reference_distance * 0.5
                    )
                    y_refined_filtered = y_pool_refined[filtered_indexes]

                    # Append the refined and filtered points to the candidates pool
                    X_candidates = np.vstack((X_candidates, X_pool_refined_filtered))
                    y_candidates = np.concatenate((y_candidates, y_refined_filtered))

                    # Append the refined and filtered points to the pool set
                    # Careful: the target from the pool (and validation) set are (Npoints, ) shaped arrays
                    X_pool = np.vstack((X_pool, X_pool_refined_filtered))
                    pool_target_landscape = np.concatenate((pool_target_landscape, y_refined_filtered.ravel()))

                    # b) Validation set
                    if not val_df.empty:
                        X_val_refined = refined_df[refined_df['set'] == 'val'][SEARCH_VAR].to_numpy()
                        X_val_refined_scaled = data_scaler.transform(X_val_refined)
                        y_val_refined = refined_df[refined_df['set'] == 'val'][TARGET_VAR].to_numpy()

                        X_val_refined_filtered, val_filtered_indexes = filter_refined_additions(
                            X_addition=X_val_refined_scaled,
                            X_existing=X_val,
                            min_distance=X_pool_reference_distance * 0.5
                        )
                        y_val_refined_filtered = y_val_refined[val_filtered_indexes]

                        # Append the refined and filtered points to the validation set
                        X_val = np.vstack((X_val, X_val_refined_filtered))
                        val_target_landscape = np.concatenate((val_target_landscape, y_val_refined_filtered.ravel()))

                    # Update the next refinement target
                    print(f"-> Adaptive refinement step {refine_counter} at {n_points_acquired} points acquired, "
                          f"added {len(y_refined_filtered)}/{refine_centroids*refine_batch_size} points.")
                    
                    # Collect all refinement points added during the experiment
                    # Add a column to identify the refinement step
                    refined_df['refinement_step'] = [refine_counter] * len(refined_df)
                    # TODO: Keep only the filtered points added to the pool/validation sets
                    # refined_df = refined_df.iloc[filtered_indexes]
                    collective_refinement_df = pd.concat([collective_refinement_df, refined_df], ignore_index=True)
                    collective_refinement_save_path = BENCHMARK_PATH / f'collective_refinement_points_rep{rep+1}.csv'
                    collective_refinement_df.to_csv(collective_refinement_save_path, index=False)

                    next_refine_target += refine_step

            # END of Adaptive refinement
            # --------------------------------------------------------------------------------
        
        print(f"\n--- End of {rep+1}/{N_REPS} ---\n")
        # Resetting the pool and candidates sets for the next repetition if the refinement took place
        if refine_generator is not None:
            # Reset the pool set; the candidates set is derived from the candidates_df so it is not affected by the refinement
            X_pool = data_scaler.transform(pool_df[SEARCH_VAR].to_numpy())
            pool_target_landscape = pool_df[TARGET_VAR].to_numpy().ravel()

            # Reset the validation set
            X_val = data_scaler.transform(val_df[SEARCH_VAR].to_numpy()) if not val_df.empty else None
            val_target_landscape = val_df[TARGET_VAR].to_numpy().ravel() if not val_df.empty else None

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
