#!

import numpy as np
import pandas as pd
import activereg.mlmodel as regmodels
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from activereg.format import DATASETS_REPO
from activereg.sampling import sample_landscape
from activereg.hyperparams import get_gp_kernel
from activereg.acquisition import (
    penalize_landscape_fast,
    highest_landscape_selection,
    AcquisitionFunction,
)


def get_gt_dataframes(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gets the ground truth and evidence dataframes from the config.

    Args:
        config (dict): Configuration dictionary containing paths to the dataframes.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Ground truth dataframe and evidence dataframe.
    """
    gt_df_name = config.get('ground_truth_file', None)
    if gt_df_name is None:
        raise ValueError("Ground truth dataframe name must be provided in the config file.")
    gt_df = pd.read_csv(DATASETS_REPO / gt_df_name)

    exp_evidence_df_name = config.get('experiment_evidence', None)
    if exp_evidence_df_name is not None:
        evidence_df = pd.read_csv(DATASETS_REPO / exp_evidence_df_name)
    else:
        evidence_df = None

    return gt_df, evidence_df


def setup_data_pool(df: pd.DataFrame, search_var: list[str], scaler: str) -> tuple[np.ndarray, BaseEstimator]:
    """Gets the search space and scales it using StandardScaler or MinMaxScaler.

    Args:
        df (pd.DataFrame): pool dataframe.
        search_var (list[str]): list of search variable names.
        scaler (str): type of scaler to use ('StandardScaler' or 'MinMaxScaler')

    Returns:
        tuple[np.ndarray, BaseEstimator]: scaled search space as a numpy array and the scaler instance
    """
    scaler_classes = {
        'StandardScaler': StandardScaler,
        'MinMaxScaler': MinMaxScaler
    }
    
    ScalerClass = scaler_classes.get(scaler)
    if ScalerClass is None:
        raise ValueError(f"Scaler {scaler} is not supported. Use {', '.join(scaler_classes.keys())}.")

    df = df.copy()
    search_var = search_var or df.columns.tolist()
    
    if not all(var in df.columns for var in search_var):
        raise ValueError("Some search variables are not in the dataframe.")

    # Scale the dataframe
    X = df[search_var].to_numpy()
    scaler_instance = ScalerClass()
    X_scaled_array = scaler_instance.fit_transform(X)

    return X_scaled_array, scaler_instance


def setup_experiment_variables(config: dict) -> tuple[str, str, int, int, str, str, list[dict]]:
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

    search_space_vars = config.get('search_space_variables', [])
    assert len(search_space_vars) > 0, "Search space variables must be defined in the config file."

    target_vars = config.get('target_variables', None)
    assert target_vars is not None, "Target variables must be defined in the config file."

    return (exp_name, 
            additional_notes, 
            n_cycles, 
            init_batch, 
            init_sampling, 
            cycle_sampling, 
            acquisition_params,
            search_space_vars,
            target_vars)


def setup_ml_model(config: dict) -> regmodels.MLModel:
    """Sets up the machine learning model based on the configuration.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        regmodels.MLModel: The configured machine learning model.
    """
    ml_model_type = config.get('ml_model', None)
    assert ml_model_type is not None, "ML model type must be specified in the config file."

    if ml_model_type == 'GPR':
        return create_gpr_instance(config)
    else:
        raise ValueError(f"Unknown ML model type: {ml_model_type}. Supported types are: ['GPR'].")


def create_gpr_instance(config: dict) -> regmodels.MLModel:
    """Creates a Gaussian Process Regressor instance.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        regmodels.MLModel: The created Gaussian Process Regressor instance.
    """
    kernel_recipe = config.get('kernel_recipe', None)
    assert kernel_recipe is not None, "Kernel recipe must be specified for GPR in the config file."

    kernel_recipe = get_gp_kernel(kernel_recipe)

    model_parameters = config.get('model_parameters', {})
    model_parameters['alpha'] = float(model_parameters['alpha'])

    return regmodels.GPR(kernel=kernel_recipe, **model_parameters)


def sampling_block(
        X_candidates: np.ndarray, 
        X_train: np.ndarray,
        y_best: float, 
        ml_model: regmodels.MLModel, 
        acquisition_params: list[dict],
        penlanscape_params: tuple[float, float] = (0.25, 1.0),
        sampling_mode: str = 'voronoi'
    ) -> tuple[list[int], np.ndarray]:
    """Samples new points from the landscape of the acquisition function.

    Args:
        X_candidates (np.ndarray): candidates points for the cycle.
        X_train (np.ndarray): training points from the cycle.
        y_best (float): best value of the target variable.
        ml_model (regmodels.MLModel): machine learning model used for the experiment.
        acquisition_params (list[dict]): acquisition function parameters for the cycle.
        penlanscape_params (tuple[float, float], optional): penalization parameters for the landscape. Defaults to (0.25, 1.0).
        sampling_mode (str, optional): sampling mode for the landscape. Defaults to 'voronoi'.

    Returns:
        tuple[list[int], np.ndarray]: new sampled indexes and the landscapes of the acquisition function.
    """

    # init variables
    X_candidates_indexes = np.arange(0,len(X_candidates))
    X_train_copy = X_train.copy()
    radius, strength = penlanscape_params

    sampled_new_idx = []
    landscape_list = []

    # loop over acquisition function types
    for acp in acquisition_params:

        acqui_param = acp.copy()
        n_points_per_style = acqui_param['n_points']
        percentile = acqui_param.pop('percentile')

        acqui_func = AcquisitionFunction(y_best=y_best, **acqui_param)
        _, landscape = acqui_func.landscape_acquisition(X_candidates=X_candidates, ml_model=ml_model)

        penalized_landscape = penalize_landscape_fast(
            landscape=landscape, 
            X_candidates=X_candidates, 
            X_train=X_train_copy,
            radius=radius, strength=strength,
        )

        acq_landscape_ndx = highest_landscape_selection(landscape=penalized_landscape, percentile=percentile)
        X_acq_landscape = X_candidates[acq_landscape_ndx]
        X_acq_landscape_indexes = X_candidates_indexes[acq_landscape_ndx]

        sampled_hls_idx = sample_landscape(
            X_landscape=X_acq_landscape, 
            n_points=n_points_per_style, 
            sampling_mode=sampling_mode
        )

        sampled_new_idx += list(X_acq_landscape_indexes[sampled_hls_idx])
        landscape_list.append(landscape)
        X_train_copy = np.concatenate([X_train, X_candidates[sampled_new_idx]])

    return sampled_new_idx, np.vstack(landscape_list)


def validation_block(gt_df: pd.DataFrame, sampled_df: pd.DataFrame, search_vars: list) -> pd.DataFrame:
    """Validates the sampled points against the ground truth.

    Args:
        gt_df (pd.DataFrame): ground truth dataframe.
        sampled_df (pd.DataFrame): sampled dataframe from a al cycle.
        search_vars (list): variables that define the search space.

    Returns:
        pd.DataFrame: validated dataframe with the sampled points and their ground truth values.
    """

    merged_df = pd.merge(
        sampled_df[search_vars], 
        gt_df,
        on=search_vars,
        how='left'
    )
    return merged_df.reset_index(drop=True)

