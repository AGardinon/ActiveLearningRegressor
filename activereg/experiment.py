#!

import numpy as np
import pandas as pd
import activereg.mlmodel as regmodels
from activereg.sampling import sample_landscape
from activereg.acquisition import (
    penalize_landscape_fast,
    highest_landscape_selection,
    AcquisitionFunction,
)


def setup_data_pool(df: pd.DataFrame, search_var: list[str], scaler: str) -> tuple:
    """Gets the search space and scales it using StandardScaler.

    Args:
        df (pd.DataFrame): pool dataframe.
        search_var (list[str]): list of search variable names.

    Returns:
        df_scaled_array, scaler (np.ndarray, Any): scaled search space as a numpy array and the scaler used.
    """
    if scaler == 'StandardScaler':
        from sklearn.preprocessing import StandardScaler
    elif scaler == 'MinMaxScaler':
        from sklearn.preprocessing import MinMaxScaler
    else:
        raise ValueError(f"Scaler {scaler} is not supported. Use 'StandardScaler' or 'MinMaxScaler'.")

    scaler = StandardScaler() if scaler == 'StandardScaler' else MinMaxScaler()

    df = df.copy()

    if search_var is None:
        search_var = df.columns.tolist()
    else:
        assert all(var in df.columns for var in search_var), "Some search variables are not in the dataframe."

    # Scale the dataframe
    X = df[search_var].to_numpy()
    X_scaled_array = scaler.fit_transform(X)

    return X_scaled_array, scaler


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

