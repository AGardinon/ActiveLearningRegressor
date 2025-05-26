#!
import yaml
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from activereg.mlmodel import MLModel
from activereg.sampling import sample_landscape
from activereg.cycle import active_learning_cycle_insilico
from activereg.utils import create_experiment_name
from activereg.format import EXAMPLES_REPO
from typing import Any

# - func

def run_insilico_experiment(
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        ml_model: MLModel,
        n_batch: int,
        init_sampling_mode: str,
        n_cycles: int,
        acquisition_parameters: dict,
        percentile: int,
        sampling_mode: str,
        out_dir: str,
        make_plot: bool=True,
        ) -> Path:
    
    """
    Run an in-silico experiment on a dataset (X_pool,y_pool), where the `y_pool` is
    also the GT and `X_pool` is the generic search space.
    `ml_model` is an object that must have a .train() and .predict() functions as
    implemented in the `activereg.mlmodel` module.
    """

    # 0. Create experiment folder
    exp_folder = create_experiment_name(name_set=(
        'results_',acquisition_parameters['acquisition_mode'],sampling_mode,
        f'nb{n_batch}',f'ncy{n_cycles}',f'pcent{percentile}'))
    out_dir = out_dir / exp_folder
    if out_dir.exists():
        counter = 1
        while (new_out_dir := out_dir.parent / f"{out_dir.name}_{counter}").exists():
            counter += 1
        out_dir = new_out_dir
    out_dir.mkdir(exist_ok=False)

    # 1. init experiment
    # Sample initial points
    if isinstance(init_sampling_mode, str):
        init_idx = sample_landscape(X_landscape=X_pool, 
                                    n_points=n_batch,
                                    sampling_mode=init_sampling_mode)
        init_sampling_mode_str = init_sampling_mode
    elif isinstance(init_sampling_mode, list):
        init_idx = init_sampling_mode
        init_sampling_mode_str = 'predef'
    
    X_train, y_train = X_pool[init_idx], y_pool[init_idx]
    X_candidates = np.delete(X_pool, init_idx, axis=0)
    y_candidates = np.delete(y_pool, init_idx, axis=0)

    # Model training on initial configuration
    ml_model.train(X_train, y_train)

    # report
    X_sampled = []
    y_sampled = []
    cycles_label = []
    cycle_predictions = []
    cycle_predictions_uncertainty = []

    X_sampled.append(X_train)
    y_sampled.append(y_train)
    cycles_label.append([0]*len(init_idx))

    if acquisition_parameters['acquisition_mode'] == 'target_expected_improvement':
        target_value = acquisition_parameters.pop('target_value')

    # 2. cycles
    for c in range(n_cycles):
        print(f'Cycle: {c+1}')

        if acquisition_parameters['acquisition_mode'] == 'target_expected_improvement':
            y_best = target_value
        else:
            y_best = max(y_train)

        X_next, y_next, y_pred, landscape, X_acq_landscape, sampled_new_idx = active_learning_cycle_insilico(
            X_candidates=X_candidates,
            y_candidates=y_candidates,
            y_best=y_best,
            model=ml_model,
            acquisition_parameters=acquisition_parameters,
            percentile=percentile,
            n_batch=n_batch,
            sampling_mode=sampling_mode,
        )

        # - plt
        if make_plot:
            from activereg.beauty import plot_2Dcycle
            # 2D data
            if X_pool.shape[1] == 2:
                plot_2Dcycle(train_set=(X_train,y_train),
                            pred_set=(X_candidates,y_pred),
                            pool_set=(X_pool,y_pool,'coolwarm'),
                            next_set=(X_next,y_next),
                            landscape_set=(landscape,X_acq_landscape,'plasma'),
                            name_set=(out_dir,'fig_',init_sampling_mode_str,
                                    acquisition_parameters['acquisition_mode'],sampling_mode,c)
                            )

        # 3. report
        X_sampled.append(X_next)
        y_sampled.append(y_next)
        cycles_label.append([c+1]*n_batch)
        cycle_predictions.append(ml_model.predict(x=X_pool)[1])
        cycle_predictions_uncertainty.append(ml_model.predict(x=X_pool)[2])

        # 4. update the trainig set
        X_train = np.vstack((X_train, X_next))
        y_train = np.append(y_train, y_next)

        # 5. remove selected point from candidates
        X_candidates = np.delete(X_candidates, sampled_new_idx, axis=0)
        y_candidates = np.delete(y_candidates, sampled_new_idx, axis=0)

        # 6. retrain ALmodel with new points
        ml_model.train(X_train, y_train)

    # 7. outputs
    print(f'Saving outputs in: {out_dir}')

    X_sampled = np.concatenate(X_sampled)
    al_report_df_dict = {f'X_s{v}' : X_sampled[:,v] for v in range(X_sampled.shape[1])}
    al_report_df_dict['y_s'] = np.concatenate(y_sampled)
    al_report_df_dict['cycles_label'] = np.concatenate(cycles_label)
    al_report_df = pd.DataFrame(al_report_df_dict)
    al_report_df.to_csv(out_dir / Path('report_experiment.csv'), index=False)

    np.save(out_dir / Path('cycle_predictions.npy'), np.array(cycle_predictions))
    np.save(out_dir / Path('cycle_predictions_uncertainty.npy'), np.array(cycle_predictions_uncertainty))

    al_report_dict = {
        'ml_model' : ml_model.__repr__(),
        'n_cycles' : n_cycles,
        'n_batch' : n_batch,
        'init_sampling_mode' : init_sampling_mode,
        'acquisition_parameters' : acquisition_parameters,
        'percentile' : percentile,
        'sampling_mode' : sampling_mode,
    }

    with open(out_dir / Path('report_experiment.json'), "w") as f:
        json.dump(al_report_dict, f, indent=4)

    if 'GPR' in ml_model.__repr__():
        joblib.dump(ml_model, out_dir / Path("gpr_model.pkl"))

    return out_dir


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Split the config files
    # 1. data
    data_config = config['data']
    data_folder = Path(data_config['data_folder'])

    X_pool = np.load(EXAMPLES_REPO / data_folder / data_config['X_pool'])
    y_pool = np.load(EXAMPLES_REPO / data_folder / data_config['y_pool'])
    
    # 2. ml
    ml_config = config['ml']

    if ml_config['ml_model'] == 'GPR':    
        from activereg.mlmodel import GPR, KernelFactory

        out_dir = EXAMPLES_REPO / data_config['out_dir'] / ml_config['ml_model'] / ml_config['kernel_recipe']['type']

        kernel = KernelFactory(kernel_recipe=ml_config['kernel_recipe']).get_kernel()
        ml_model = GPR(kernel=kernel, **ml_config['ml_model_param'])
        print(ml_model)

    out_dir.mkdir(exist_ok=data_config['overwrite'], parents=True)

    # 3. experiment
    exp_config = config['experiment']
    exp_dir = run_insilico_experiment(X_pool=X_pool, y_pool=y_pool,
                                      ml_model=ml_model, out_dir=out_dir,
                                      **exp_config, make_plot=False)

    with open(exp_dir / Path('experiment_config.log'), "w") as f:
        yaml.dump(config, f, default_flow_style=False)