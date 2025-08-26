#!

import yaml
import argparse
# from pprint import pprint
from typing import Any, List, Tuple

import joblib
import pandas as pd
import numpy as np
import activereg.mlmodel as regmodels
from pathlib import Path
from activereg.utils import create_strict_folder
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import root_mean_squared_error
from activereg.sampling import sample_landscape
from activereg.acquisition import AcquisitionFunction, highest_landscape_selection, penalize_landscape_fast
from activereg.utils import save_to_json

# INIT
# EXP_NAME = 'labsim_benchmark_1'
# ADDITIONAL_NOTES = 'acqui_func.landscape_acquisition() with penalize landscape and predictiong only on candidates.'

# POOL_CSV_FILENAME = 'single_variable_doe_POOL.csv'
# CANDIDATES_CSV_FILENAME = 'single_variable_doe_CANDIDATES.csv'
# TRAIN_CSV_FILENAME = 'single_variable_doe_TRAIN.csv'
# POOL_CSV_PATH = Path(EXP_NAME + '/dataset/' + POOL_CSV_FILENAME)
# TRAIN_CSV_PATH = Path(EXP_NAME + '/dataset/' + TRAIN_CSV_FILENAME)
# CANDIDATES_CSV_PATH = Path(EXP_NAME + '/dataset/' + CANDIDATES_CSV_FILENAME)

# SEARCH_VAR = ['x1', 'x2']
# TARGET_VAR = ['Number_of_Particles']

# ## Dataset
# gt_df_name = 'gt_number_of_particles_maternw.csv'
# gt_df = pd.read_csv(f'../datasets/{gt_df_name}')
# target_var = 'Number_of_Particles'

# ## Exp variables
# N_CYCLES = 3
# INIT_SAMPLING = 'fps'
# CYCLE_SAMPLING = 'voronoi'

# ACQUI_PARAMS_EXPLOIT = {
#     'acquisition_mode' : 'expected_improvement',
#     'percentile' : 97,
#     'n_points' : 1
# }

# ACQUI_PARAMS_EXPLOIT_TARGET = {
#     'acquisition_mode' : 'target_expected_improvement',
#     'y_target' : 800,
#     'epsilon' : 25,
#     'percentile' : 97,
#     'n_points' : 1
# }

# ACQUI_PARAMS_EXPLORE = {
#     'acquisition_mode' : 'exploration_mutual_info',
#     'percentile' : 95,
#     'n_points' : 2
# }

# FUNCTIONS

# # Define acquisition parameters
# def get_acquisition_params(param_list: list[dict]) -> list[dict]:
#     """Defines acquisition parameters for the active learning cycle.

#     Args:
#         param_list (list[dict]): List of dictionaries with acquisition parameters.

#     Returns:
#         list[dict]: unified list of acquisition parameters.
#     """
#     return [
#         {
#             'acquisition_mode': param['acquisition_mode'],
#             'percentile': param.get('percentile', 95),
#             'n_points': param['n_points'],
#             **{k: v for k, v in param.items() if k not in ['acquisition_mode', 'percentile', 'n_points']}
#         }
#         for param in param_list
#     ]


# def setup_data_pool(df: pd.DataFrame, search_var: list[str]) -> tuple:
#     """Gets the search space and scales it using StandardScaler.

#     Args:
#         df (pd.DataFrame): pool dataframe.
#         search_var (list[str]): list of search variable names.

#     Returns:
#         df_scaled_array, scaler (np.ndarray, Any): scaled search space as a numpy array and the scaler used.
#     """
#     scaler = StandardScaler()
#     df = df.copy()

#     if search_var is None:
#         search_var = df.columns.tolist()
#     else:
#         assert all(var in df.columns for var in search_var), "Some search variables are not in the dataframe."

#     # Scale the dataframe
#     X = df[search_var].to_numpy()
#     X_scaled_array = scaler.fit_transform(X)

#     return X_scaled_array, scaler
    

# def sampling_block(
#         X_candidates: np.ndarray, 
#         X_train: np.ndarray,
#         y_best: float, 
#         ml_model: regmodels.MLModel, 
#         acquisition_params: list[dict],
#         sampling_mode: str = 'voronoi', 
#     ) -> tuple[list[int], np.ndarray]:

#     X_candidates_indexes = np.arange(0,len(X_candidates))
#     sampled_new_idx = []
#     landscape_list = []

#     X_train_copy = X_train.copy()

#     for acp in acquisition_params:

#         acqui_param = acp.copy()
#         n_points_per_style = acqui_param['n_points']
#         percentile = acqui_param.pop('percentile')

#         acqui_func = AcquisitionFunction(y_best=y_best, **acqui_param)
#         _, landscape = acqui_func.landscape_acquisition(X_candidates=X_candidates, ml_model=ml_model)

#         penalized_landscape = penalize_landscape_fast(
#             landscape=landscape, 
#             X_candidates=X_candidates, 
#             X_train=X_train_copy,
#             radius=0.25, strength=1.0,
#         )

#         acq_landscape_ndx = highest_landscape_selection(landscape=penalized_landscape, percentile=percentile)
#         X_acq_landscape = X_candidates[acq_landscape_ndx]
#         X_acq_landscape_indexes = X_candidates_indexes[acq_landscape_ndx]

#         sampled_hls_idx = sample_landscape(
#             X_landscape=X_acq_landscape, 
#             n_points=n_points_per_style, 
#             sampling_mode=sampling_mode
#         )

#         sampled_new_idx += list(X_acq_landscape_indexes[sampled_hls_idx])
        
#         landscape_list.append(landscape)

#         X_train_copy = np.concatenate([X_train, X_candidates[sampled_new_idx]])

#     return sampled_new_idx, np.vstack(landscape_list)


# def validation_block(gt_df: pd.DataFrame, sampled_df: pd.DataFrame, search_vars: list) -> pd.DataFrame:
#     # Get target values from ground truth for screened points
#     merged_df = pd.merge(sampled_df[search_vars], 
#             gt_df,
#             on=search_vars,
#             how='left')
#     return merged_df.reset_index(drop=True)

def create_insilico_al_experiment_paths(
        exp_name: str,
        gt_dataframe: pd.DataFrame,
        search_space_variables: List[str],
        evidence_dataframe: pd.DataFrame = None,
        dataset_path_name: str = 'dataset',
    ) -> Tuple[Path, Path, Path]:
    """Initializes the experiment paths for the insilico active learning simulation.

    Args:
        exp_name (str): experiment name.
        gt_dataframe (pd.DataFrame): ground truth dataframe.
        search_space_variables (List[str]): search space variables.
        evidence_dataframe (pd.DataFrame, optional): starting evidence dataframe. Defaults to None.

    Returns:
        Tuple[Path, Path, Path]: Paths for the experiment, pool CSV, and candidates CSV.
    """
    from activereg.format import INSILICO_AL_REPO
    insilico_al_path = INSILICO_AL_REPO / exp_name

    # Create the experiment folder
    create_strict_folder(path_str=str(insilico_al_path))

    # Create the dataset folder
    dataset_path = insilico_al_path / dataset_path_name
    create_strict_folder(path_str=str(dataset_path))

    # Create the main experiment paths
    pool_csv_path = dataset_path / f'{exp_name}_POOL.csv'
    candidates_csv_path = dataset_path / f'{exp_name}_CANDIDATES.csv'
    train_csv_path = dataset_path / f'{exp_name}_TRAIN.csv'

    # Pool contains only the search space variables form the ground truth dataframe
    assert all(var in gt_dataframe.columns for var in search_space_variables), \
        f"Search space variables {search_space_variables} not found in ground truth dataframe columns."
    gt_dataframe[search_space_variables].to_csv(pool_csv_path, index=False)

    if evidence_dataframe is not None:
        # If evidence dataframe is provided, save it as the training set
        assert all(var in evidence_dataframe.columns for var in search_space_variables), \
            f"Search space variables {search_space_variables} not found in evidence dataframe columns."
        assert all(var in evidence_dataframe.columns for var in gt_dataframe.columns), \
            f"Ground truth variables {gt_dataframe.columns.tolist()} not found in evidence dataframe columns."

        # Remove evidence points from the ground truth dataframe
        evidence_set = set(evidence_dataframe[search_space_variables].apply(tuple, axis=1))
        candidates_df = gt_dataframe[~gt_dataframe[search_space_variables].apply(tuple, axis=1).isin(evidence_set)]

        # Save the candidates and the evidence dataframe to the dataset folder
        candidates_df.to_csv(candidates_csv_path, index=False)
        evidence_dataframe.to_csv(dataset_path / f'{exp_name}_EVIDENCE.csv', index=False)

    return insilico_al_path, pool_csv_path, candidates_csv_path, train_csv_path


def setup_experiment(config: dict) -> Tuple[str, str, int, int, str, str, List[dict]]:
    """Sets up the experiment parameters from the config dictionary.

    Args:
        config (dict): Configuration dictionary containing experiment parameters.

    Returns:
        tuple: Contains experiment name, additional notes, number of cycles, initial batch size,
               initial sampling method, cycle sampling method, and acquisition parameters.
    """
    exp_name = config.get('experiment_name', 'Insilico_AL_Simulation')
    additional_notes = config.get('experiment_notes', '')
    n_cycles = config.get('n_cycles', 3)
    init_batch = config.get('init_batch_size', 8)
    init_sampling = config.get('init_sampling', 'fps')
    cycle_sampling = config.get('cycle_sampling', 'voronoi')
    acquisition_params = config.get('acquisition_parameters', [])

    return (exp_name, additional_notes, n_cycles, init_batch, init_sampling, cycle_sampling, acquisition_params)

# MAIN

DATASET_PATH_NAME = 'dataset'

## Model
kernel_func = ConstantKernel(1.)*RBF(length_scale=1.) + WhiteKernel(noise_level=0.1)
gpr_params = {
    'kernel' : kernel_func,
    'alpha' : 1e-10,
    'n_restarts_optimizer' : 150,
    'optimizer' : 'fmin_l_bfgs_b',
    'normalize_y' : True,
    'use_gridsearch' : False,
    'param_grid' : None,
    'log_transform' : False,
}
ML_MODEL = regmodels.GPR(**gpr_params)


if __name__ == '__main__':

    from activereg.format import DATASETS_REPO
    from activereg.experiment import sampling_block, validation_block

    # Parse the config.yaml
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Extracting parameters from the config file
    (EXP_NAME, ADDITIONAL_NOTES, N_CYCLES, INIT_BATCH, INIT_SAMPLING, CYCLE_SAMPLING, ACQUI_PARAMS) = setup_experiment(config)

    # Setting parameters for the experiment
    N_BATCH = sum(acp['n_points'] for acp in ACQUI_PARAMS)
    SEARCH_VAR = config.get('search_space_variables', [])
    TARGET_VAR = config.get('target_variables', 'target')

    # Paths for the experiment
    gt_df_name = config.get('ground_truth_file', None)
    if gt_df_name is None:
        raise ValueError("Ground truth dataframe name must be provided in the config file.")
    gt_df = pd.read_csv(DATASETS_REPO / gt_df_name)

    exp_evidence_df_name = config.get('experiment_evidence', None)
    if exp_evidence_df_name is not None:
        evidence_df = pd.read_csv(DATASETS_REPO / exp_evidence_df_name)
    else:
        evidence_df = None

    INSILICO_AL_PATH, POOL_CSV_PATH, CANDIDATES_CSV_PATH, TRAIN_CSV_PATH = create_insilico_al_experiment_paths(
        exp_name=EXP_NAME,
        gt_dataframe=gt_df,
        search_space_variables=SEARCH_VAR,
        evidence_dataframe=evidence_df,
        dataset_path_name=DATASET_PATH_NAME
    )

    # Set up the data scaler
    scaler_path = INSILICO_AL_PATH / DATASET_PATH_NAME / 'scaler.pkl'
    pool_df = pd.read_csv(POOL_CSV_PATH)
    scaler = StandardScaler().fit(pool_df.values)
    joblib.dump(scaler, scaler_path)

    print('# ----------------------------------------------------------------------------\n'\
          f'# \tExperiment: {EXP_NAME} \t\n'\
          '# ----------------------------------------------------------------------------\n'\
          f'Additional notes: {ADDITIONAL_NOTES}\n'
          )

    # AL cycles    
    for cycle in range(N_CYCLES):

        print(f'\n# Cycle {cycle+1} of {N_CYCLES} - Sampling {N_BATCH} points')

        # Candidates are the pool of points availalble for sampling at the current cycle
        # For cycle 0, candidates are the whole pool and for the next cycles, 
        # candidates are the points not yet sampled.
        if cycle == 0:
            # Load the total pool dataframe
            candidates_df = pd.read_csv(POOL_CSV_PATH)

        elif cycle > 0:
            # Load candidates dataframe
            if not CANDIDATES_CSV_PATH.exists():
                raise FileNotFoundError(f'Candidates CSV file not found at {CANDIDATES_CSV_PATH}. Please ensure it exists.')
            # Load candidates dataframe
            candidates_df = pd.read_csv(CANDIDATES_CSV_PATH)

        if not TRAIN_CSV_PATH.exists():
            print(f'Training CSV file not found at {TRAIN_CSV_PATH}. Starting from scratch.')
            train_df = pd.DataFrame(columns=candidates_df.columns)
        else:
            # Load training dataframe
            train_df = pd.read_csv(TRAIN_CSV_PATH)

        # Removing training dataframe from the candidates dataframe
        # This way we can start from a pre-defined set of training points
        # Given that the training dataframe is a subset of the candidates dataframe
        candidates_df = candidates_df[~candidates_df[SEARCH_VAR].apply(tuple, axis=1).isin(train_df[SEARCH_VAR].apply(tuple, axis=1))]

        # Create the cycle output folder and save the outputs
        cycle_output_path = INSILICO_AL_PATH / Path(f'cycle_{cycle}')
        create_strict_folder(path_str=str(cycle_output_path))

        # -------------------------------------- #
        # --- INIT of cycle 0
        if cycle == 0:

            X_candidates = scaler.transform(candidates_df)
            
            # # Save scaler to pkl file
            # scaler_path = Path(EXP_NAME + '/dataset/scaler.pkl')
            # joblib.dump(scaler, scaler_path)

            # Sample initial points from the candidates
            # using the sampling mode defined in the experiment setup
            screened_indexes = sample_landscape(
                X_landscape=X_candidates, 
                n_points=N_BATCH,
                sampling_mode=INIT_SAMPLING
            )

            cycle_0_log_dict = {
                'cycle' : cycle,
                'init_sampling_mode' : INIT_SAMPLING,
                'n_points' : N_BATCH,
                'screened_indexes' : np.array(screened_indexes).astype(int).tolist(),
                'candidates_df_shape' : candidates_df.shape,
                'train_df_shape' : train_df.shape,
            }
            save_to_json(
                dictionary=cycle_0_log_dict,
                fout_name=cycle_output_path / Path('cycle_0_log.json'),
                timestamp=False
            )
        # --- END of cycle 0
        # -------------------------------------- #

        # -------------------------------------- #
        # --- INIT of cycle > 0
        if cycle > 0:

            # # Load the scaler if it exists
            # try:
            #     scaler = joblib.load(scaler_path)
            # except FileNotFoundError:
            #     raise FileNotFoundError(f'Scaler not found at {scaler_path}. Please ensure it exists.')
            
            # Prepare candidates and training data
            candidates_df = candidates_df.reset_index(drop=True)
            train_df = train_df.reset_index(drop=True)
            X_candidates = scaler.transform(candidates_df[SEARCH_VAR].to_numpy())
            y_train = train_df[TARGET_VAR].to_numpy()
            X_train = scaler.transform(train_df[SEARCH_VAR].to_numpy())

            print(f'Cycle {cycle} - Candidates shape: {X_candidates.shape}, Training shape: {X_train.shape}')

            # Compute the best target value from the training set
            y_best = max(y_train)

            # Train model on evidence and predict on candidates
            ML_MODEL.train(X_train, y_train)
            _, y_pred, y_unc = ML_MODEL.predict(X_candidates)

            # Sample new points based on the acquisition function
            screened_indexes, landscape = sampling_block(
                X_candidates=X_candidates, 
                X_train=X_train,
                y_best=y_best,
                ml_model=ML_MODEL,
                acquisition_params=ACQUI_PARAMS,
                sampling_mode=CYCLE_SAMPLING
            )

            # Save the cycle log
            cycle_log_dict = {
                'cycle' : cycle,
                'sampling_mode' : CYCLE_SAMPLING,
                'n_points' : N_BATCH,
                'acquisition_params' : ACQUI_PARAMS,
                'screened_indexes' : np.array(screened_indexes).astype(int).tolist(),
                'candidates_df_shape' : candidates_df.shape,
                'train_df_shape' : train_df.shape,
                'y_best' : float(y_best),
                'nll' : ML_MODEL.model.log_marginal_likelihood().astype(float),
                'model_params' : ML_MODEL.__repr__()
            }
            save_to_json(
                dictionary=cycle_log_dict,
                fout_name=cycle_output_path / Path(f'cycle_{cycle}_log.json'),
                timestamp=False
            )
            
            # Save the landscape and predictions &
            # Add landscapes as columns with acquisition parameter names
            model_predictions_df = pd.DataFrame({
                **{col: candidates_df[col] for col in SEARCH_VAR},
                'y_pred': y_pred,
                'y_uncertainty': y_unc
            })

            print(landscape.shape, len(ACQUI_PARAMS))
            
            for i, acqui_param in enumerate(ACQUI_PARAMS):
                col_name = f"landscape_{acqui_param['acquisition_mode']}"
                model_predictions_df[col_name] = landscape[i]

            model_predictions_df.to_csv(cycle_output_path / Path(f'cycle_{cycle}_predictions.csv'), index=False)

            train_df.to_csv(cycle_output_path / Path(f'X_train_cycle_{cycle}.csv'), index=False)

            # Save model snapshot
            model_snapshot_path = cycle_output_path / Path(f'model_snapshot_cycle_{cycle}.pkl')
            joblib.dump(ML_MODEL, model_snapshot_path)
        # --- END of cycle > 0
        # -------------------------------------- #

        # Update training set
        next_df = candidates_df.iloc[screened_indexes][SEARCH_VAR]
        next_df.to_csv(cycle_output_path / Path(f'cycle_{cycle}_output_sampled.csv'), index=False)

        # Validation block
        validated_df = validation_block(
            gt_df=gt_df, 
            sampled_df=next_df, 
            search_vars=SEARCH_VAR
        )
        validated_df.to_csv(cycle_output_path / Path(f'cycle_{cycle}_validated.csv'), index=False)

        # Merge the new validated data with the existing training data
        train_df = pd.concat([train_df, validated_df], ignore_index=True)
        train_df.to_csv(TRAIN_CSV_PATH, index=False)

        # Update candidates dataframe removing the sampled points
        candidates_df = candidates_df.drop(index=screened_indexes)
        candidates_df.to_csv(CANDIDATES_CSV_PATH, index=False)

    # END of AL cycle
