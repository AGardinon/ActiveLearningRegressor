#!

'''
Benchamark Experiment

Actions:
1.

'''

import yaml
import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from activereg import beauty
from activereg.sampling import sample_landscape
from activereg.utils import create_strict_folder
from activereg.experiment import (get_gt_dataframes, 
                                  sampling_block, 
                                  setup_data_pool,
                                  setup_ml_model,
                                  remove_evidence_from_gt,
                                  setup_experiment_variables)
from typing import List, Tuple

# FUNCTIONS

def create_benchmark_path(
       exp_name: str,
       gt_dataframe: pd.DataFrame,
       search_space_variables: List[str],
       overwrite: bool=False,
    ) -> Tuple[Path, Path, pd.DataFrame]:

    from activereg.format import BENCHMARKS_REPO

    benchmark_path = BENCHMARKS_REPO / exp_name

    # Create the benchmark folder
    create_strict_folder(path_str=str(benchmark_path), overwrite=overwrite)

    # Create the main experiment paths
    pool_csv_path = benchmark_path / f'{exp_name}_POOL.csv'

    # Pool contains only the search space variables form the ground truth dataframe
    assert all(var in gt_dataframe.columns for var in search_space_variables), \
        f"Search space variables {search_space_variables} not found in ground truth dataframe columns."
    pool_df = gt_dataframe[search_space_variables]
    pool_df.to_csv(pool_csv_path, index=False)

    return benchmark_path, pool_csv_path, pool_df

# MAIN

if __name__ == '__main__':

    # Parse the config.yaml
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML configuration file")
    parser.add_argument("-r", "--repetitions", type=int, default=1, help="Number of repetitions for the experiment")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Extracting parameters from the config file &
    # setting parameters for the experiment
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
    N_REPS = args.repetitions

    # Get ground truth and evidence dataframes
    gt_df, evidence_df = get_gt_dataframes(config)

    # Create benchmark paths and set up dataframes
    # pool_df -> complete search space
    BENCHMARK_PATH, POOL_CSV_PATH, pool_df = create_benchmark_path(
        exp_name=EXP_NAME,
        gt_dataframe=gt_df,
        search_space_variables=SEARCH_VAR,
        overwrite=True,
    )

    # Set up the data scaler on the complete search space
    data_scaler_type = config.get('data_scaler', None)
    if data_scaler_type is None:
        print("Data scaler type not specified in config file. Using StandardScaler as default.")
        data_scaler_type = "StandardScaler"

    scaler_path = BENCHMARK_PATH / f'{data_scaler_type}_scaler.joblib'
    X_pool, scaler = setup_data_pool(df=pool_df, search_var=SEARCH_VAR, scaler=data_scaler_type)
    joblib.dump(scaler, scaler_path)

    # Create candidates dataframe and target variable candidates used for validation
    # candidates_df -> unscreened space
    candidates_df = remove_evidence_from_gt(gt_df, evidence_df, search_vars=SEARCH_VAR)

    # Set up the model
    ML_MODEL = setup_ml_model(config)

    # --------------------------------------------------------------------------------
    # RUN THE BENCHMARK EXPERIMENT

    benchmark_data = []
    pred_landscape_array = np.full((N_REPS, N_CYCLES, X_pool.shape[0]), np.nan)

    for rep in range(N_REPS):

        # Set up candidates and train sets
        X_candidates = scaler.transform(candidates_df[SEARCH_VAR].to_numpy())
        y_candidates = candidates_df[TARGET_VAR].to_numpy()

        if evidence_df is not None:
            X_train = scaler.transform(evidence_df[SEARCH_VAR].to_numpy())
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

        for cycle in range(N_CYCLES):

            print(f"--- Repetition {rep+1}/{N_REPS} | Cycle {cycle+1}/{N_CYCLES} ---")

            # Compute the best screened value and train the model
            y_best = np.max(y_train)

            ML_MODEL.train(X_train, y_train)
            _, y_pred, y_unc = ML_MODEL.predict(X_pool)

            # Sample from the candidates
            sampled_indexes, landscape = sampling_block(
                X_candidates=X_candidates,
                X_train=X_train,
                y_best=y_best,
                ml_model=ML_MODEL,
                acquisition_params=ACQUI_PARAMS,
                sampling_mode=CYCLE_SAMPLING,
                penlanscape_params=(0.25, 1.0)
            )

            benchmark_data.append({
                "repetition": rep+1,
                "cycle": cycle+1,
                "y_best_screened": y_best,
                "y_best_predicted": np.max(y_pred),
                "nll": ML_MODEL.model.log_marginal_likelihood().astype(float),
                "model_params": ML_MODEL.__repr__()
            })

            pred_landscape_array[rep, cycle] = y_pred

            # Update the train and candidates sets
            X_train = np.vstack((X_train, X_candidates[sampled_indexes]))
            y_train = np.concatenate((y_train, y_candidates[sampled_indexes]))

            X_candidates = np.delete(X_candidates, sampled_indexes, axis=0)
            y_candidates = np.delete(y_candidates, sampled_indexes, axis=0)


    # Save benchmark data to CSV
    benchmark_df = pd.DataFrame(benchmark_data)
    benchmark_df.to_csv(BENCHMARK_PATH / 'benchmark_data.csv', index=False)

    np.save(BENCHMARK_PATH / 'predicted_landscape.npy', pred_landscape_array)
    beauty.plot_predicted_landscape(X_pool, pred_landscape_array, BENCHMARK_PATH)
