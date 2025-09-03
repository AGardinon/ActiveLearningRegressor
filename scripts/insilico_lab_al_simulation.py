#############################################################################################
#
# Insilico Active Learning Simulation Script
#
# This script simulates an active learning experiment in a controlled environment.
# It allows for the configuration of various parameters and the execution of the experiment.
#
#############################################################################################

import yaml
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
from activereg.utils import save_to_json
from activereg.sampling import sample_landscape
from activereg.utils import create_strict_folder
from activereg.experiment import (sampling_block, 
                                  setup_ml_model, 
                                  validation_block, 
                                  setup_data_pool,
                                  get_gt_dataframes,
                                  remove_evidence_from_gt,
                                  setup_experiment_variables)

# FUNCTIONS

def create_insilico_al_experiment_paths(
        exp_name: str,
        pool_dataframe: pd.DataFrame,
        search_space_variables: List[str],
        evidence_dataframe: pd.DataFrame = None,
        dataset_path_name: str = 'dataset',
    ) -> Tuple[Path, Path, Path]:
    """Initializes the experiment paths for the insilico active learning simulation.

    Args:
        exp_name (str): experiment name.
        pool_dataframe (pd.DataFrame): ground truth dataframe.
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
    assert all(var in pool_dataframe.columns for var in search_space_variables), \
        f"Search space variables {search_space_variables} not found in ground truth dataframe columns."
    pool_df = pool_dataframe[search_space_variables]
    pool_df.to_csv(pool_csv_path, index=False)

    candidates_df = remove_evidence_from_gt(pool_df, evidence_df, search_space_variables)
    candidates_df.to_csv(candidates_csv_path, index=False)

    if evidence_dataframe is not None:
        evidence_dataframe.to_csv(dataset_path / f'{exp_name}_EVIDENCE.csv', index=False)

    return insilico_al_path, pool_csv_path, candidates_csv_path, train_csv_path

# MAIN

if __name__ == '__main__':
    # Parse the config.yaml
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    DATASET_PATH_NAME = 'dataset'

    # Extracting parameters from the config file &
    # setting the parameters for the experiment
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

    # Get ground truth and evidence dataframes
    gt_df, evidence_df = get_gt_dataframes(config)

    # Paths for the experiment
    INSILICO_AL_PATH, POOL_CSV_PATH, CANDIDATES_CSV_PATH, TRAIN_CSV_PATH = create_insilico_al_experiment_paths(
        exp_name=EXP_NAME,
        pool_dataframe=gt_df,
        search_space_variables=SEARCH_VAR,
        evidence_dataframe=evidence_df,
        dataset_path_name=DATASET_PATH_NAME
    )

    # Set up the data scaler
    scaler_path = INSILICO_AL_PATH / DATASET_PATH_NAME / 'scaler.joblib'
    pool_df = pd.read_csv(POOL_CSV_PATH)

    data_scaler_type = config.get('data_scaler', None)
    if data_scaler_type is None:
        print("Data scaler type not specified in config file. Using StandardScaler as default.")
        data_scaler_type = "StandardScaler"

    X_pool, scaler = setup_data_pool(df=pool_df, search_var=SEARCH_VAR, scaler=data_scaler_type)
    joblib.dump(scaler, scaler_path)

    # Set up the model
    ML_MODEL = setup_ml_model(config)

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
            candidates_df = pd.read_csv(CANDIDATES_CSV_PATH)

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

            X_candidates = scaler.transform(candidates_df[SEARCH_VAR].to_numpy())

            # Sample initial points from the candidates
            # using the sampling mode defined in the experiment setup
            screened_indexes = sample_landscape(
                X_landscape=X_candidates, 
                n_points=INIT_BATCH,
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
            
            # Prepare candidates and training data
            candidates_df = candidates_df.reset_index(drop=True)
            train_df = train_df.reset_index(drop=True)
            X_candidates = scaler.transform(candidates_df[SEARCH_VAR].to_numpy())
            y_train = train_df[TARGET_VAR].to_numpy()
            X_train = scaler.transform(train_df[SEARCH_VAR].to_numpy())

            print(f'Cycle {cycle} - Candidates shape: {X_candidates.shape}, Training shape: {X_train.shape}')

            # Compute the best target value from the training set
            y_best = max(y_train).item()

            # Train model on evidence and predict on pool to generate
            # the outputs per cycle
            ML_MODEL.train(X_train, y_train)
            _, y_pred, y_unc = ML_MODEL.predict(X_pool)

            # Sample new points from candidates based on the acquisition function
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
                'y_best' : int(y_best),
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
                **{col: pool_df[col] for col in SEARCH_VAR},
                'y_pred': y_pred,
                'y_uncertainty': y_unc
            })

            model_landscapes_df = pd.DataFrame({
                **{col: candidates_df[col] for col in SEARCH_VAR}
            })
            for i, acqui_param in enumerate(ACQUI_PARAMS):
                col_name = f"landscape_{acqui_param['acquisition_mode']}"
                model_landscapes_df[col_name] = landscape[i]

            model_predictions_df.to_csv(cycle_output_path / Path(f'cycle_{cycle}_predictions.csv'), index=False)
            model_landscapes_df.to_csv(cycle_output_path / Path(f'cycle_{cycle}_landscapes.csv'), index=False)

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
        if cycle == 0:
            train_df = validated_df
        elif cycle > 0:
            train_df = pd.concat([train_df, validated_df], ignore_index=True)
        train_df.to_csv(TRAIN_CSV_PATH, index=False)

        # Update candidates dataframe removing the sampled points
        candidates_df = candidates_df.drop(index=screened_indexes)
        candidates_df.to_csv(CANDIDATES_CSV_PATH, index=False)

    # END of AL cycle
