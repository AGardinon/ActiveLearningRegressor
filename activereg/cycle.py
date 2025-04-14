#!
import numpy as np
import pandas as pd
from activereg.sampling import sample_landscape
from activereg.acquisition import AcquisitionFunction, highest_landscape_selection
from typing import List


def prepare_training_data(search_space_df: pd.DataFrame, 
                          target_evidence_df: pd.DataFrame, 
                          target_column: str):
    """
    Prepares data for training an ML model by separating points with evidence from candidates.
    
    Parameters:
    ----------
    search_space_df : pandas.DataFrame
        The full pool of candidate points
    target_evidence_df : pandas.DataFrame
        The points for which we have target variable values
    target_column : str
        The name of the target variable column in target_evidence_df
        
    Returns:
    -------
    X_train : numpy.ndarray
        Feature matrix for training data
    y_train : numpy.ndarray
        Target values for training data
    X_candidates : numpy.ndarray
        Feature matrix for candidate points (those without target values)
    """
    # Extract target values and search space from the target_evidence_df
    y_train = target_evidence_df[target_column].values
    X_train = target_evidence_df.drop(columns=[target_column]).values
    
    # Find candidate points not in the training set
    feature_cols = [col for col in target_evidence_df.columns if col != target_column]
    train_points_set = set(map(tuple, X_train))
    
    # Filter search_space_df to get points not in training set
    mask = search_space_df[feature_cols].apply(lambda row: tuple(row) not in train_points_set, axis=1)
    X_candidates = search_space_df[mask][feature_cols].values
    
    return X_train, y_train, X_candidates


# def lab_al_cycle()


def active_learning_cycle_insilico(
        X_candidates,
        y_candidates,
        y_best,
        model, 
        acquisition_parameters,
        percentile,
        n_batch,
        sampling_mode
        ):
    """
    Performs the active learning regression cycle:
    1.  compute the navigation landscape
    2.  select top-portion of the landscape
    3.  sample new points from the selected portion
    """

    # 1.
    acqui_fun = AcquisitionFunction(y_best=y_best, **acquisition_parameters)
    y_pred, landscape = acqui_fun.landscape_acquisition(X_candidates=X_candidates, ml_model=model)
    
    # 2.
    acq_landscape_ndx = highest_landscape_selection(landscape=landscape, 
                                                    percentile=percentile)
    X_acq_landscape = X_candidates[acq_landscape_ndx]

    # 3.
    sampled_hls_idx = sample_landscape(X_landscape=X_acq_landscape,
                                       n_points=n_batch,
                                       sampling_mode=sampling_mode)
    sampled_new_idx = acq_landscape_ndx[sampled_hls_idx]
    
    # 4.
    X_next, y_next = X_candidates[sampled_new_idx], y_candidates[sampled_new_idx]

    return X_next, y_next, y_pred, landscape, X_acq_landscape, sampled_new_idx