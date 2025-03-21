#!
import toml
import argparse
import numpy as np
from activereg.sampling import sample_landscape
from activereg.cycle import active_learning_cycle
from activereg.mlmodel import ML_MODELS
from activereg.beauty import plot_2Dcycle
from activereg.format import EXAMPLES_REPO

# - func

def run_2Dexperiment(
        X_pool,
        y_pool,
        ml_model,
        n_batch: int,
        init_sampling_mode: str,
        n_cycles: int,
        acquisition_mode: str,
        percentile: int,
        sampling_mode: str,
        out_dir: str):

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
    ml_model.fit(X_train, y_train)

    # 2. cycles
    for c in range(n_cycles):

        print(f'Cycle: {c+1}')

        X_next, y_next, y_pred, landscape, X_acq_landscape, sampled_new_idx = active_learning_cycle(
            X_candidates=X_candidates,
            y_candidates=y_candidates,
            model=ml_model,
            acquisition_mode=acquisition_mode,
            percentile=percentile,
            n_batch=n_batch,
            sampling_mode=sampling_mode
        )

        # - plt
        plot_2Dcycle(train_set=(X_train,y_train),
                    pred_set=(X_candidates,y_pred),
                    pool_set=(X_pool,y_pool,'coolwarm'),
                    next_set=(X_next,y_next),
                    landscape_set=(landscape,X_acq_landscape,'plasma'),
                    name_set=(out_dir,'fig_',init_sampling_mode_str,acquisition_mode,sampling_mode,c))

        # 5. update the trainig set
        X_train = np.vstack((X_train, X_next))
        y_train = np.append(y_train, y_next)

        # remove selected point from candidates
        X_candidates = np.delete(X_candidates, sampled_new_idx, axis=0)
        y_candidates = np.delete(y_candidates, sampled_new_idx, axis=0)

        # 6. retrain ALmodel with new points
        ml_model.fit(X_train, y_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read a TOML config file.")
    parser.add_argument("-c", "--config", required=True, help="Path to the TOML configuration file")
    args = parser.parse_args()
    config = toml.load(args.config)

    X_pool = np.load(EXAMPLES_REPO / 'data' / config['X_pool'])
    y_pool = np.load(EXAMPLES_REPO / 'data' / config['y_pool'])

    out_dir = EXAMPLES_REPO / config['out_dir']
    out_dir.mkdir(exist_ok=config['overwrite'])

    with open(out_dir / 'config.log', "w") as f:
        toml.dump(config, f)

    ml_model_type = config['ml_model']
    ml_model = ML_MODELS[ml_model_type]

    run_2Dexperiment(X_pool=X_pool, 
                     y_pool=y_pool,
                     ml_model=ml_model, 
                     out_dir=out_dir,
                     **config['experiment'])