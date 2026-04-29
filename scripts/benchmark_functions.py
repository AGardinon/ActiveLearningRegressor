# ------------------------------------------------------------------------------
# Benchmark Experiment Script for Active Learning Regressor
# ------------------------------------------------------------------------------

'''Script description:
This script is designed to run benchmark experiments for active learning regression 
using the ActiveRegressor framework. It operates with a mixed-batch pool-based AL
protocol designed to evaluate a benchmark function (either synthetic or custom) where
the ground truth (GT) is known and used as a holdout validation set.
An initial training dataset is generated based on the parameters provided in the 
config file and it can be expanded with a simple geometric-based adaptive refinement strategy.

The script requires a few configuration files in YAML format:
- benchmark_config.yaml: contains the general settings for the benchmark experiment, 
    such as the experiment name, number of cycles, initial sampling method, acquisition protocol, and other relevant parameters.

- model_config.yaml: contains the settings for the machine learning model used in the experiment, 
    including the model type and its hyperparameters.

- acquisition_mode_settings.yaml: contains the settings for the acquisition modes used in the experiment, 
    such as the number of points to acquire for each mode and the specific parameters for each acquisition mode.

- target_function_config.yaml: contains the settings for the benchmark function, 
    including the function type, its parameters, and the settings for the training dataset generation.
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
                                  setup_multi_property_ml_model,
                                  remove_evidence_from_gt,
                                  setup_experiment_variables,
                                  validate_acquisition_params,
                                  AcquisitionParametersGenerator)
from activereg.hyperparams import (get_fixed_params, 
                                   merge_model_params, 
                                   grid_search_cv)
from activereg.benchmarkFunctions import FUNCTION_CLASSES, FUNCTIONS_DICT
from typing import List, Tuple
from sklearn.base import BaseEstimator
from functools import partial

# --------------------------------------------------------------------------------
# FUNCTIONS

def setup_gridsearch(ml_model_name: str) -> dict:
    """Sets up the grid search parameters for the specified machine learning model.

    Args:
        ml_model_name (str): Name of the machine learning model.

    Raises:
        ValueError: If the specified ML model is not supported.

    Returns:
        dict: The parameter grid for the specified ML model.
    """
    from activereg.hyperparams import GPR_MATERN_PARAM_GRID, MLP_PARAM_GRID, KNN_PARAM_GRID
    if ml_model_name == 'GPR':
        param_grid = GPR_MATERN_PARAM_GRID
    elif ml_model_name == 'AnchoredEnsembleMLP':
        param_grid = MLP_PARAM_GRID
    elif ml_model_name == 'kNNRegressorAL':
        param_grid = KNN_PARAM_GRID
    else:
        raise ValueError(f"Unsupported ML model for grid search: {ml_model_name}")
    return param_grid


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


def save_configs_files(
    path: Path, 
    benchmark_config: dict = {}, 
    model_config: dict = {}, 
    acquisition_config: dict = {}, 
    target_function_config: dict = {}
) -> None:
    """Saves the aggregated configuration dictionary to a YAML file in the benchmark folder.
    Args:
        path (Path): The path to the benchmark folder where the config file will be saved.
        benchmark_config (dict, optional): Benchmark configuration dictionary. Defaults to {}.
        model_config (dict, optional): Model configuration dictionary. Defaults to {}.
        acquisition_config (dict, optional): Acquisition mode settings dictionary. Defaults to {}.
        target_function_config (dict, optional): Target function configuration dictionary. Defaults to {}.
    """
    config_folder = path / 'config'
    config_folder.mkdir(exist_ok=True)
    # Benchmark configuration
    with open(config_folder / 'benchmark_config.yaml', 'w') as file:
        yaml.dump(benchmark_config, file)
    # Model configuration
    with open(config_folder / 'model_config.yaml', 'w') as file:
        yaml.dump(model_config, file)
    # Acquisition mode settings
    with open(config_folder / 'acquisition_mode_settings.yaml', 'w') as file:
        yaml.dump(acquisition_config, file)
    # Target function configuration
    with open(config_folder / 'target_function_config.yaml', 'w') as file:
        yaml.dump(target_function_config, file)

# --------------------------------------------------------------------------------
# MAIN EXPERIMENT SCRIPT
if __name__ == '__main__':
    # --------------------------------------------------------------------------------
    # Parse the config.yaml
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("-bc", "--benchmark_config", required=True, help="Path to the YAML configuration file")
    parser.add_argument("-mc", "--model_config", required=True, help="Path to the YAML configuration file for the ML model and parameters")
    parser.add_argument("-acqmodes", "--acquisition_mode_settings", required=True, help="Acquisition mode settings for the experiment")
    parser.add_argument("-tfc", "--target_function_config", required=True, help="Path to the YAML configuration file for the benchmark function parameters")
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

    # Load the target function configuration if provided, otherwise it will be skipped
    with open(args.target_function_config, "r") as file:
        target_function_config = yaml.safe_load(file)
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # SET RANDOM SEEDS
    seed = benchmark_config.get('SEED', None)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    rng = np.random.default_rng(seed)   # seeded Generator for sampling_block weight sampling
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

    # Save a copy of the config file in the benchmark folder for reference and reproducibility purposes
    save_configs_files(
        path=BENCHMARK_PATH, 
        benchmark_config=benchmark_config, 
        model_config=model_config, 
        acquisition_config=acquisition_config, 
        target_function_config=target_function_config
    )

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

    perform_grid_search = model_config.get('grid_search', {}).get('perform_grid_search', False)
    if perform_grid_search:
        print("Grid search activated for hyperparameter tuning.")
        param_grid = setup_gridsearch(ml_model_name=ml_model_type)
        gridsearch_every_n_points = model_config.get('grid_search', {}).get('every_n_points', 10)
        print(f"Grid search will be performed every {gridsearch_every_n_points} acquired points during the experiment.")

        # Create a folder to store the grid search results and best hyperparameters for each grid search step
        grid_search_results_path = BENCHMARK_PATH / 'grid_search_results'
        grid_search_results_path.mkdir(exist_ok=True)

        fixed_model_params = get_fixed_params(config_params=ml_model_params, param_grid=param_grid)
        with open(grid_search_results_path / 'grid_search_fixed_params.yaml', 'w') as f:
            yaml.dump(fixed_model_params, f)
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
    # Validate acquisition entries against the declared target names (catches config
    # mistakes — wrong target names, ambiguous target keys, missing weight config —
    # before any expensive computation starts).
    validate_acquisition_params(acquisition_parameters, TARGET_VAR)
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # FUNCTION SELECTION
    function_config = target_function_config.get('function_parameters', None)
    assert function_config is not None, "Function parameters must be provided in the config file."

    function_name = function_config.pop('function', None)
    function_dict = FUNCTIONS_DICT.get(function_name, None)

    if function_dict is None:
        assert function_config.pop('custom', False) is True, "If the function is not in the predefined FUNCTIONS_DICT, it must be defined as a custom function in a separate file and the 'custom' parameter must be set to True in the config file."
        function_class = FUNCTION_CLASSES.get(function_name, None)
        if function_class is None:
            raise ValueError(f"Custom function '{function_name}' is not defined in FUNCTION_CLASSES. Please define it or check the function name.")
        else:
            function_params = function_config
            print(f"Selected custom benchmark function: {function_name} with parameters: {function_params}")

    else:
        function_class = function_dict['function_class']
        function_params = function_dict['function_params']
        print(f"Selected benchmark function: {function_name} with parameters: {function_params}")

    function_dim = function_params.get('n_dimensions', None)
    function_bounds = function_params.get('bounds', None)
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
    # GENERATE TRAINING DATASET
    # Generate synthetic dataset
    train_set_config = target_function_config.get('train_set_parameters', None)
    
    if train_set_config is None:
        raise ValueError("Train set parameters must be provided in the config file to generate the training dataset regardless of the GT file.")

    elif train_set_config is not None:
        train_function_negate = train_set_config.pop('negate', False)
        dataset_generator = DatasetGenerator(
            n_dimensions=function_dim,
            bounds=function_bounds,
            negate=train_function_negate,
            seed=seed
        )

        if gt_file is not None:
            print("Both train_set_parameters and ground_truth_file are provided in the config file. "
            "The GT will be used as holdout validation, while the pool will be generated based on the train_set_parameters.")
            train_set_config['val_size'] = None  # ensure that the train set generator does not split the dataset into train and validation, as the GT will be used as validation

            pool_df = dataset_generator.generate_dataset(
                function=function_class,
                **train_set_config
            )
            val_df = gt_df.copy()  # use the GT as validation set

        elif gt_file is None:
            print("The training dataset will be generated based on the parameters provided.")
            print("A validation set will be generated as gt with the same settings.")
            train_set_config['val_size'] = 1.0

            train_df = dataset_generator.generate_dataset(
                function=function_class,
                **train_set_config
            )
            pool_df, val_df = process_dataset(train_df, search_var=SEARCH_VAR, target_var=TARGET_VAR, split_column='set')

            # Save the generated VAL dataset for reference
            val_save_path = BENCHMARK_PATH / 'gt_validation_dataset.csv'
            val_df.to_csv(val_save_path, index=False)
    # --------------------------------------------------------------------------------

    # Save the generated POOL dataset for reference
    pool_save_path = BENCHMARK_PATH / 'pool_dataset.csv'
    pool_df.to_csv(pool_save_path, index=False)
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # PREPARE DATAFRAMES AND SCALER
    # train dataset -> pool_df : from which the candidates will be sampled and refined during the experiment
    # gt dataset -> val_df : used as holdout validation set for metrics computation during the experiment

    # Extract the target landscapes for metrics computation — shape (N, P)
    pool_target_landscape = pool_df[TARGET_VAR].to_numpy()
    val_target_landscape = val_df[TARGET_VAR].to_numpy()

    # Set up and apply the data scaler on the complete search space
    data_scaler_type = benchmark_config.get('data_scaler', None)
    data_scaler_params = benchmark_config.get('scaler_params') or {}
    data_scaler = data_scaler_setup(data_scaler_type, data_scaler_params)
    X_pool, data_scaler = setup_data_pool(df=pool_df, search_var=SEARCH_VAR, scaler=data_scaler)
    joblib.dump(data_scaler, scaler_path)

    # Create candidates dataframe (candidates_df -> unscreened space)
    candidates_df = remove_evidence_from_gt(gt=pool_df, evidence=evidence_df, search_vars=SEARCH_VAR)

    # Scale the validation set if available to compute metrics
    X_val = data_scaler.transform(val_df[SEARCH_VAR].to_numpy())
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
            n_dimensions=function_dim,
            bounds=function_bounds,
            negate=train_function_negate,
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
        # SET UP THE MODEL (params from the config file)
        ML_MODEL = setup_multi_property_ml_model(model_config, TARGET_VAR)
        # --------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------
        print(f"\n--- Starting repetition {rep+1}/{N_REPS} ---\n")
        # Refinement step tracking per repetition
        if refine_generator is not None:
            refine_counter = 0
            next_refine_target = refine_step

        # Grid search tracking per repetition
        if perform_grid_search is True:
            grid_search_counter = 0  # to track when to perform grid search during the cycles
            next_grid_search_target = gridsearch_every_n_points

        # --------------------------------------------------------------------------------
        # CYCLE 0 - INITIAL SAMPLING
        X_candidates = data_scaler.transform(candidates_df[SEARCH_VAR].to_numpy())
        y_candidates = candidates_df[TARGET_VAR].to_numpy()

        if evidence_df is not None:
            X_train = data_scaler.transform(evidence_df[SEARCH_VAR].to_numpy())
            Y_train = evidence_df[TARGET_VAR].to_numpy().reshape(-1, len(TARGET_VAR))

        elif evidence_df is None:
            screened_indexes = sample_landscape(
                X_landscape=X_candidates,
                n_points=INIT_BATCH,
                sampling_mode=INIT_SAMPLING
            )
            X_train = X_candidates[screened_indexes]
            Y_train = y_candidates[screened_indexes].reshape(-1, len(TARGET_VAR))

            X_candidates = np.delete(X_candidates, screened_indexes, axis=0)
            y_candidates = np.delete(y_candidates, screened_indexes, axis=0)

        for i in range(len(X_train)):
            train_points_data.append({
                **{col: X_train[i, j] for j, col in enumerate(SEARCH_VAR)},
                "cycle": 0,
                "repetition": rep+1,
                "acquisition_source": INIT_SAMPLING if evidence_df is None else "evidence",
                **{name: float(Y_train[i, j]) for j, name in enumerate(TARGET_VAR)},
            })

        # !!! Careful
        n_points_acquired = len(X_train)
        # END of initial sampling
        # --------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------
        # CYCLES 1 -> N_CYCLES
        for cycle in tqdm(range(N_CYCLES), desc=f"Repetition {rep+1}/{N_REPS}"):
            # print(f"points acquired so far: {n_points_acquired, len(X_train)}")
            # ----------------------------------------------------------------------------
            # Grid search step (every N acquired points as defined in the config file)
            if perform_grid_search is True and n_points_acquired >= next_grid_search_target:
                grid_search_counter += 1
                # Use the first underlying model's class for grid search (global spec only).
                _first_underlying = list(ML_MODEL._models.values())[0]
                base_model_factory = partial(_first_underlying.__class__, **fixed_model_params)
                # Perform grid search cross-validation on the first target property.
                gridsearch_result = grid_search_cv(
                    model_factory=base_model_factory,
                    param_grid=param_grid,
                    X=X_train,
                    y=Y_train[:, 0],
                    scoring='neg_mean_squared_error',
                    verbose=1
                )

                # TODO: fix bug with saving of the parameters, as it saves non text serializable objects in the yaml file,
                # need to convert them to string or find a way to save only the relevant information for the parameters (e.g. for the kernel, save the recipe or the name instead of the whole kernel object)

                # # Save the grid search results and best hyperparameters for this step
                # grid_search_step_path = grid_search_results_path / f'grid_search_rep{rep+1}_cycle{cycle+1}.yaml'
                # with open(grid_search_step_path, 'w') as f:
                #     yaml.dump(gridsearch_result, f)

                best_params = gridsearch_result['best_params']
                merge_best_params = merge_model_params(config_params=ml_model_params, best_params=best_params)
                # Rebuild the multi-property model with the updated hyperparameters.
                ML_MODEL = setup_multi_property_ml_model(
                    {**model_config, 'model_parameters': merge_best_params},
                    TARGET_VAR
                )

                next_grid_search_target += gridsearch_every_n_points
            # ----------------------------------------------------------------------------

            # Get the acquisition parameters for the current cycle
            # Predefined acquisition modes for tracking the source of acquisition of each point
            predefined_acquisition_modes = []
            cycle_acqui_params = acqui_param_gen.get_params_for_cycle(cycle)
            for acp in cycle_acqui_params:
                predefined_acquisition_modes.extend([acp['acquisition_mode']] * acp['n_points'])

            # Train the model on current training set
            ML_MODEL.train(X_train, Y_train)

            _, y_pred_pool, y_unc_pool = ML_MODEL.predict(X_pool)  # each (N_pool, P)
            _, y_pred_val, y_unc_val = ML_MODEL.predict(X_val)      # each (N_val, P)

            # Sample from the candidates
            sampled_indexes, landscape, cycle_meta = sampling_block(
                X_candidates=X_candidates,
                X_train=X_train,
                Y_train=Y_train,
                ml_model=ML_MODEL,
                acquisition_params=cycle_acqui_params,
                batch_selection_method=batch_selection_method,
                batch_selection_params=batch_selection_params,
                penalization_params=(pen_radius, pen_strength) if landscape_penalization is not None else None,
                rng=rng,
            )

            # Store benchmark data for the current cycle
            cycle_data_dict = {
                "repetition": rep+1,
                "cycle": cycle+1,
                **{f"y_best_{name}": float(np.max(Y_train[:, j])) for j, name in enumerate(TARGET_VAR)},
            }
            cycle_metrics_dict = {}
            for j, name in enumerate(TARGET_VAR):
                prop_metrics = evaluate_cycle_metrics(
                    y_true_pool=pool_target_landscape[:, j],
                    y_pred_pool=y_pred_pool[:, j],
                    y_true_val=val_target_landscape[:, j],
                    y_pred_val=y_pred_val[:, j],
                    y_uncertainty_val=y_unc_val[:, j]
                )
                cycle_metrics_dict.update({f"{k}_{name}": v for k, v in prop_metrics.items()})
            cycle_data_dict.update(cycle_metrics_dict)
            # Log joint-entry metadata: resolved weights and y_best_z for ParEGO entries.
            # Per-property and fast-path entries have meta=None and are skipped.
            for entry, meta in zip(cycle_acqui_params, cycle_meta):
                if meta is not None:
                    entry_id = entry.get('name', entry.get('acquisition_mode', 'joint'))
                    cycle_data_dict[f"y_best_z_{entry_id}"] = meta['_y_best_z']
                    cycle_data_dict[f"resolved_weights_{entry_id}"] = meta['_resolved_weights'].tolist()
            benchmark_data.append(cycle_data_dict)

            # Update the train and candidates sets
            X_train = np.vstack((X_train, X_candidates[sampled_indexes]))
            Y_train = np.concatenate((Y_train, y_candidates[sampled_indexes].reshape(-1, len(TARGET_VAR))))

            # Update training points tracking
            sampled_Y = y_candidates[sampled_indexes].reshape(-1, len(TARGET_VAR))
            for i in range(len(X_candidates[sampled_indexes])):
                train_points_data.append({
                    **{col: X_candidates[sampled_indexes][i, j] for j, col in enumerate(SEARCH_VAR)},
                    "cycle": cycle+1,
                    "repetition": rep+1,
                    "acquisition_source": predefined_acquisition_modes[i],
                    **{name: float(sampled_Y[i, j]) for j, name in enumerate(TARGET_VAR)},
                })

            X_candidates = np.delete(X_candidates, sampled_indexes, axis=0)
            y_candidates = np.delete(y_candidates, sampled_indexes, axis=0)
            # END of cycles
            # -------------------------------------------------------------------------------- 

            n_points_acquired += len(sampled_indexes)

            # --------------------------------------------------------------------------------
            # Adaptive refinement of candidates pool if defined in the config file
            # TODO: consider refactoring into a class or function for better readability and modularity
            if refine_generator is not None:

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
                            design_bounds=function_bounds,
                            design_dimensions=function_dim,
                            refine_centroid=np.reshape(centroid, (1, -1)),  # reshape to (1, D)
                            pool_scaled=X_pool,
                            half_side_length_strategy=half_side_length_strategy,
                            hsl_strategy_params=half_side_length_strategy_params,
                            n_points=refine_batch_size,
                            refine_noise_std=refine_noise_std,
                            scaler=data_scaler,
                            refine_function=function_class,
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
                    X_pool = np.vstack((X_pool, X_pool_refined_filtered))
                    pool_target_landscape = np.vstack((
                        pool_target_landscape,
                        y_refined_filtered.reshape(-1, len(TARGET_VAR))
                    ))

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
            pool_target_landscape = pool_df[TARGET_VAR].to_numpy()

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
