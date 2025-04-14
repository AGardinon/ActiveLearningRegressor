#!

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from activereg.utils import create_strict_folder
from activereg.cycle import prepare_training_data, lab_al_cycle, validate_next_points, update_validation_df
from activereg.format import REPO_ROOT,FILE_TO_VAL

# INPUT VARIABLES

exp_input_config = {
    'experiment_dir' : 'test_labAL_experiment',
    'experiment_name' : 'test_labAL_2D',
}

data_input_config = {
    'search_space' : 'test_dataset_2D.csv',
    'target_variable' : 'target',
    'target_validated_evidence' : 'test_dataset_2D_evidence.csv'
}

KERNEL_RECIPE = ['*', 
                 {'type': 'C', 'constant_value': 1.0}, 
                 {'type': 'Matern', 'length_scale': 1.0, 'nu' : 2.5}]

ml_model_config = {
    'ml_model' : 'GPR',
    'ml_model_param' : {
        'n_restarts_optimizer' : 15
    },
    'kernel_recipe' : KERNEL_RECIPE
}

lab_al_config = {
    'n_batch' : 4,
    'acquisition_parameters' : {
        'acquisition_mode' : 'expected_improvement',
        'xi' : 1.e-2
    },
    'percentile' : 95,
    'sampling_mode' : 'voronoi'
}

# MAIN

CYCLE_NUMBER = 3
GT_FILE = 'double_well_gaussian_gt.csv'


if __name__ == '__main__':

    print('# --------------------------------------\n'\
         f'# \tActive Cycle {CYCLE_NUMBER}     \n'\
          '# --------------------------------------\n'\
          )

    # 1. set up folders
    experiment_path = REPO_ROOT / exp_input_config['experiment_dir'] / exp_input_config['experiment_name']
    cycle_output_path = experiment_path / 'cycles' / f'cycle_{CYCLE_NUMBER}'

    # create output files folder
    create_strict_folder(path_str=str(cycle_output_path))
    print(f'Output files will be stored in:\n{cycle_output_path}')

    # init dict for logging info of cycle
    cycle_log_dict = {
        'exp_input' : exp_input_config,
        'data_input' : data_input_config,
        'ml_model' : ml_model_config,
        'lab_al' : lab_al_config,
    }

    # 2. load search space
    search_space_df = pd.read_csv(experiment_path / 'dataset' / data_input_config['search_space'])

    if CYCLE_NUMBER > 0:
        update_validation_df(experiment_path=experiment_path, 
                             target_evidence_df=data_input_config['target_validated_evidence'],
                             cycle_num=CYCLE_NUMBER-1)
        
    target_evidence_df = pd.read_csv(experiment_path / 'dataset' / data_input_config['target_validated_evidence'])
    print(f'Screened points: {target_evidence_df.info()}')

    # 3. get experiment splits
    X_train, y_train, X_candidates, feature_cols, scaler = prepare_training_data(
        search_space_df=search_space_df, 
        target_evidence_df=target_evidence_df,
        target_column=data_input_config['target_variable'],
        scale=True
    )

    # 4. init ml model
    if ml_model_config['ml_model'] == 'GPR':
        from activereg.mlmodel import GPR, KernelFactory

        kernel_func = KernelFactory(kernel_recipe=ml_model_config['kernel_recipe']).get_kernel()
        ml_model = GPR(kernel=kernel_func, **ml_model_config['ml_model_param'])

    # Model training on evidence data
    ml_model.train(X_train, y_train)
    joblib.dump(ml_model, cycle_output_path / f'ml_model_cycle_{CYCLE_NUMBER}.joblib')

    # 5. Cycle
    y_best = max(y_train)
    X_next = lab_al_cycle(X_candidates=X_candidates, 
                          y_best=y_best,
                          model=ml_model,
                          **lab_al_config)
    
    X_next = scaler.inverse_transform(X=X_next)
    X_next_df = pd.DataFrame(data={
        col : X_next[:,i] for i, col in enumerate(feature_cols)
    })
    X_next_df[data_input_config['target_variable']] = [np.nan]*len(X_next)

    # 6. outputs
    X_next_df.to_csv(cycle_output_path / FILE_TO_VAL.format(CYCLE_NUMBER), index=False)

    _, target_variable_predict, target_variable_uncertainty = ml_model.predict(x=search_space_df.values)
    np.save(cycle_output_path / f'X_predic_cycle_{CYCLE_NUMBER}.npy', target_variable_predict)
    np.save(cycle_output_path / f'X_predic_uncertainty_cycle_{CYCLE_NUMBER}.npy', target_variable_uncertainty)

    cycle_log_dict['y_best'] = y_best
    cycle_log_dict['cycle'] = CYCLE_NUMBER

    with open(cycle_output_path / Path(f'report_cycle_{CYCLE_NUMBER}.json'), "w") as f:
        json.dump(cycle_log_dict, f, indent=4)

    # 7. validation
    if GT_FILE:
        gt_df = pd.read_csv(experiment_path / 'dataset' / GT_FILE)

        validate_next_points(experiment_path=experiment_path,
                             gt_df=gt_df,
                             target_column=data_input_config['target_variable'],
                             cycle_num=CYCLE_NUMBER)
        
    