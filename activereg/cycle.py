#!

import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from activereg.sampling import sample_landscape
from activereg.acquisition import AcquisitionFunction, highest_landscape_selection
from activereg.format import FILE_VALIDATED, FILE_TO_VAL
from typing import List


def prepare_training_data(search_space_df: pd.DataFrame, 
                          target_evidence_df: pd.DataFrame, 
                          target_column: str,
                          scale: bool=True):
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
    feature_cols : list[str]
        Columns name for the feature matrix
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

    # scaler
    scaler = StandardScaler()
    scaler.fit(search_space_df)

    return scaler.transform(X_train), y_train, scaler.transform(X_candidates), feature_cols, scaler


def lab_al_cycle(
        X_candidates, 
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
    X_next = X_candidates[sampled_new_idx]

    return X_next


def validate_next_points(experiment_path, gt_df, target_column, cycle_num):
    """
    Validates candidate points by adding target values from a ground truth dataframe.
    
    Parameters:
    ----------
    experiment_path : Path or str
        Path to the experiment directory containing cycle folders
    gt_df : pandas.DataFrame
        Ground truth dataframe containing features and target values
    target_column : str
        Name of the target column in gt_df
        
    Returns:
    -------
    None (writes validated files to disk)
    """
    experiment_path = Path(experiment_path)    
    cycle_dir = experiment_path / 'cycles' / f'cycle_{cycle_num}'
    assert cycle_dir.exists(), f'Folder {cycle_dir} does not exist!'

    # Read the candidate points file
    candidates_df = pd.read_csv(cycle_dir / FILE_TO_VAL.format(cycle_num))
    
    # Create a copy for validation
    validated_df = candidates_df.copy()
    
    # Match candidates with ground truth to get target values
    feature_cols = [col for col in gt_df.columns if col != target_column]
    
    # Create a dictionary for faster lookup from ground truth data
    # Using tuples of feature values as keys and target values as values
    gt_dict = {tuple(row[feature_cols]): row[target_column] 
                for _, row in gt_df.iterrows()}
    
    # Add target column to the validated dataframe
    validated_df[target_column] = validated_df[feature_cols].apply(
        lambda row: gt_dict.get(tuple(row)), axis=1
    )
    
    # Save validated dataframe to the cycle directory
    validated_path = cycle_dir / FILE_VALIDATED.format(cycle_num)
    validated_df.to_csv(validated_path, index=False)
    
    print(f"Created validated file: {validated_path}")


def update_validation_df(experiment_path: str, target_evidence_df: str, cycle_num):

    evidence_df = pd.read_csv(experiment_path / 'dataset' / target_evidence_df)
    cycle_validated_df = pd.read_csv(experiment_path / 'cycles' / f'cycle_{cycle_num}' / FILE_VALIDATED.format(cycle_num))

    concat_df = pd.concat([evidence_df, cycle_validated_df])
    concat_df.to_csv(experiment_path / 'dataset' / target_evidence_df, index=False)

    print(f"Updated evidence df: {target_evidence_df}")


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