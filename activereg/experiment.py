#!

import numpy as np
import pandas as pd
import activereg.mlmodel as regmodels
from pathlib import Path
from sklearn.base import BaseEstimator
from activereg.format import DATASETS_REPO
from activereg.hyperparams import get_gp_kernel
from activereg.acquisition import (
    penalize_landscape_fast,
    AcquisitionFunction,
    landscape_sanity_check,
    BatchSelectionStrategy,
    compute_per_property_stats,
    scalarize,
    WeightSampler,
)


def get_gt_dataframes(ground_truth_file: str, experiment_evidence_file: str=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gets the ground truth and evidence dataframes from the config.

    Args:
        ground_truth_file (str): Path to the ground truth dataframe file.
        experiment_evidence_file (str, optional): Path to the experiment evidence dataframe file. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Ground truth dataframe and evidence dataframe.
    """
    gt_df_name = Path(ground_truth_file)
    if gt_df_name is None:
        raise ValueError("Ground truth dataframe name must be provided in the config file.")
    gt_df = pd.read_csv(DATASETS_REPO / gt_df_name)

    exp_evidence_df_name = Path(experiment_evidence_file) if experiment_evidence_file is not None else None
    evidence_df = pd.read_csv(DATASETS_REPO / exp_evidence_df_name) if exp_evidence_df_name is not None else None

    return gt_df, evidence_df


def setup_data_pool(df: pd.DataFrame, search_var: list[str], scaler: BaseEstimator) -> tuple[np.ndarray, BaseEstimator]:
    """Gets the search space and scales it using StandardScaler or MinMaxScaler.

    Args:
        df (pd.DataFrame): pool dataframe.
        search_var (list[str]): list of search variable names.
        scaler (BaseEstimator): pre-initialized scaler instance

    Returns:
        tuple[np.ndarray, BaseEstimator]: scaled search space as a numpy array and the scaler instance
    """
    df = df.copy()
    search_var = search_var or df.columns.tolist()
    
    if not all(var in df.columns for var in search_var):
        raise ValueError("Some search variables are not in the dataframe.")

    # Scale the dataframe
    X = df[search_var].to_numpy()
    X_scaled_array = scaler.fit_transform(X)

    return X_scaled_array, scaler


def remove_evidence_from_gt(gt: pd.DataFrame, evidence: pd.DataFrame, search_vars: list[str]) -> pd.DataFrame:
    if evidence is not None:
        # If evidence dataframe is provided, save it as the training set
        assert all(var in evidence.columns for var in search_vars), \
            f"Search space variables {search_vars} not found in evidence dataframe columns."
        assert all(var in evidence.columns for var in gt.columns), \
            f"Ground truth variables {gt.columns.tolist()} not found in evidence dataframe columns."

        # Remove evidence points from the ground truth dataframe
        evidence_set = set(evidence[search_vars].apply(tuple, axis=1))
        candidates_df = gt[~gt[search_vars].apply(tuple, axis=1).isin(evidence_set)]

    elif evidence is None:
        # If evidence dataframe is not provided, the candidates are
        # the same as the gt for the first cycle
        candidates_df = gt.copy()

    return candidates_df


def setup_experiment_variables(
    config: dict,
) -> tuple[str, str, int, int, str, list[str], list[str]]:
    """Sets up the experiment parameters from the config dictionary.
    The configuration dictionary must contain the following keys:
    - experiment_name (str): Name of the experiment.
    - experiment_notes (str): Additional notes for the experiment.
    - n_cycles (int): Number of active learning cycles.
    - init_batch_size (int): Initial batch size.
    - init_sampling (str): Initial sampling method.
    - search_space_variables (list[str]): Feature column names.
    - target_variables (list[str]): Target property names. Guaranteed non-empty.

    Args:
        config (dict): Configuration dictionary containing experiment parameters.

    Returns:
        tuple: (experiment_name, additional_notes, n_cycles, init_batch_size,
                init_sampling, search_space_variables, target_variables).
               ``target_variables`` is always a non-empty list[str] — it is the
               authoritative list of target property names for the experiment.
    """
    exp_name = config.get('experiment_name', None)
    additional_notes = config.get('experiment_notes', '')
    n_cycles = config.get('n_cycles', 3)
    init_batch = config.get('init_batch_size', 8)
    init_sampling = config.get('init_sampling', 'fps')

    search_space_vars = config.get('search_space_variables', [])
    assert len(search_space_vars) > 0, "Search space variables must be defined in the config file."

    target_vars = config.get('target_variables', None)
    assert isinstance(target_vars, list) and len(target_vars) > 0, (
        "target_variables must be a non-empty list in the config file."
    )

    return (exp_name,
            additional_notes,
            n_cycles,
            init_batch,
            init_sampling,
            search_space_vars,
            target_vars)


def setup_ml_model(ml_model_type: str, ml_model_params: dict) -> regmodels.MLModel:
    """Sets up the machine learning model based on the configuration.

    Args:
        ml_model_type (str): Type of machine learning model to use.
        ml_model_params (dict): Parameters for the machine learning model.

    Returns:
        regmodels.MLModel: The configured machine learning model.
    """
    if ml_model_type == 'GPR':
        return create_gpr_instance(ml_model_params)
    elif ml_model_type == 'AnchoredEnsembleMLP':
        return create_anchored_ensemble_mlp(ml_model_params)
    elif ml_model_type == 'kNNRegressorAL':
        return create_knn_instance(ml_model_params)
    elif ml_model_type == 'BayesianNN':
        return create_bnn_instance(ml_model_params)
    else:
        raise ValueError(f"Unknown ML model type: {ml_model_type}. Supported types are: ['GPR', 'AnchoredEnsembleMLP', 'kNNRegressorAL', 'BayesianNN'].")


def setup_multi_property_ml_model(
    config: dict,
    target_names: list[str],
) -> regmodels.IndependentMultiPropertyModel:
    """Build a MultiPropertyMLModel from a model config dict and a list of target names.

    Supports two config layouts (D6):

    **Global spec** — ``config["ml_model"]`` is a string; the same model type
    and ``config["model_parameters"]`` apply to every target. One independent
    model instance is created per target::

        ml_model: "GPR"
        model_parameters:
          kernel: "MATERN_W"
          ...

    **Per-property spec** — ``config["ml_model"]`` is a dict keyed by target
    name; ``config["model_parameters"]`` is also a dict keyed by target name::

        ml_model:
          y1: "GPR"
          y2: "AnchoredEnsembleMLP"
        model_parameters:
          y1: {kernel: "MATERN_W", ...}
          y2: {n_models: 10, ...}

    For single-property experiments, the global-spec path produces an
    ``IndependentMultiPropertyModel`` with a single entry, which is the
    canonical multi-property interface even for single-target runs.

    Args:
        config (dict): Model configuration dict (typically loaded from a
            ``*_config.yaml`` file).
        target_names (list[str]): Ordered list of target property names.
            Must be non-empty.

    Returns:
        IndependentMultiPropertyModel: Wrapped collection of single-output
            models, one per target property.
    """
    ml_model_spec = config["ml_model"]
    model_params = config.get("model_parameters", {})

    if isinstance(ml_model_spec, str):
        # Global spec: same type and parameters for every target.
        models = {
            name: setup_ml_model(ml_model_spec, model_params)
            for name in target_names
        }
    elif isinstance(ml_model_spec, dict):
        # Per-property spec: each target may have a different model type/params.
        missing = [n for n in target_names if n not in ml_model_spec]
        if missing:
            raise ValueError(
                f"Per-property ml_model spec is missing entries for: {missing}. "
                f"Keys present: {list(ml_model_spec.keys())}"
            )
        if not isinstance(model_params, dict) or any(
            n not in model_params for n in target_names
        ):
            missing_p = [n for n in target_names if n not in model_params]
            raise ValueError(
                f"Per-property model_parameters must be a dict keyed by target name. "
                f"Missing entries for: {missing_p}"
            )
        models = {
            name: setup_ml_model(ml_model_spec[name], model_params[name])
            for name in target_names
        }
    else:
        raise ValueError(
            f"config['ml_model'] must be a string (global spec) or a dict "
            f"(per-property spec), got {type(ml_model_spec).__name__}."
        )

    return regmodels.IndependentMultiPropertyModel(models)


def validate_acquisition_params(
    acquisition_params: list[dict],
    target_names: list[str],
) -> None:
    """Validate acquisition parameter entries against the declared target names.

    Raises ``ValueError`` with a descriptive message on the first malformed
    entry.  Designed to be called once at experiment setup time, before the AL
    cycle loop, so config mistakes surface immediately rather than mid-run.

    Rules enforced:

    * ``target_variable`` and ``target_variables`` are mutually exclusive.
    * If ``target_variable`` is present, the name must exist in ``target_names``.
    * If ``target_variables`` is present:
      - Must be a non-empty list.
      - All referenced names must exist in ``target_names``.
      - Must have exactly one of ``weight_sampler`` or ``weights`` (not both,
        not neither).
      - If ``weights`` is a fixed vector, its length must equal
        ``len(target_variables)``.
    * Entries with neither key are allowed (legacy single-property mode where
      ``sampling_block`` falls back to ``prop_idx=0``).

    Args:
        acquisition_params: List of acquisition entry dicts (from the
            acquisition_mode_settings YAML after loading).
        target_names: Authoritative list of target property names for the
            experiment (from the benchmark config YAML).

    Raises:
        ValueError: If any entry violates the rules above.
    """
    for i, entry in enumerate(acquisition_params):
        label = (
            f"acquisition entry {i} "
            f"(name={entry.get('name', entry.get('acquisition_mode', '<unknown>'))!r})"
        )

        has_tv  = 'target_variable'  in entry
        has_tvs = 'target_variables' in entry

        if has_tv and has_tvs:
            raise ValueError(
                f"{label}: 'target_variable' and 'target_variables' are mutually "
                "exclusive. Use 'target_variable' for a single-property entry, "
                "'target_variables' for a joint scalarized entry."
            )

        if has_tv:
            tv = entry['target_variable']
            if tv not in target_names:
                raise ValueError(
                    f"{label}: target_variable={tv!r} is not in "
                    f"target_names={target_names}."
                )

        if has_tvs:
            tvs = entry['target_variables']
            if not isinstance(tvs, list) or len(tvs) == 0:
                raise ValueError(
                    f"{label}: 'target_variables' must be a non-empty list, "
                    f"got {tvs!r}."
                )
            missing = [t for t in tvs if t not in target_names]
            if missing:
                raise ValueError(
                    f"{label}: 'target_variables' references names not in "
                    f"target_names: {missing}. target_names={target_names}."
                )
            has_ws = 'weight_sampler' in entry
            has_w  = 'weights' in entry
            if has_ws and has_w:
                raise ValueError(
                    f"{label}: 'weight_sampler' and 'weights' are mutually "
                    "exclusive. Use 'weight_sampler' for per-cycle random weights "
                    "or 'weights' for a fixed vector."
                )
            if not has_ws and not has_w:
                raise ValueError(
                    f"{label}: joint entry (target_variables) must specify either "
                    "'weight_sampler' or 'weights'."
                )
            if has_w:
                w = entry['weights']
                if len(w) != len(tvs):
                    raise ValueError(
                        f"{label}: len(weights)={len(w)} does not match "
                        f"len(target_variables)={len(tvs)}."
                    )


def create_gpr_instance(model_parameters: dict) -> regmodels.MLModel:
    """Creates a Gaussian Process Regressor instance.
    Dictionary must contain the following keys:
        - kernel (str or list): Recipe for the GP kernel. See `get_gp_kernel` and `KernelFactory` for details.
        - alpha (float): Value added to the diagonal of the kernel matrix during fitting. Default is 1e-10.
        - optimizer (str or callable): Optimizer to use for kernel hyperparameter optimization. Default is 'fmin_l_bfgs_b'.
        - n_restarts_optimizer (int): Number of restarts for the optimizer. Default is 0.
        - normalize_y (bool): Whether to normalize the target values. Default is False

    Args:
        model_parameters (dict): Configuration dictionary containing model parameters.

    Returns:
        regmodels.MLModel: The created Gaussian Process Regressor instance.
    """
    model_parameters_copy = model_parameters.copy()
    kernel_function = model_parameters_copy.pop('kernel', None)

    # Assert that kernel_function is provided
    assert kernel_function is not None, \
        "'kernel' must be provided in the model parameters."
    
    kernel_recipe = get_gp_kernel(kernel_function)

    if 'alpha' in model_parameters_copy:
        model_parameters_copy['alpha'] = float(model_parameters_copy['alpha'])
    # else use the default value from the regmodels.GPR class
    return regmodels.GPR(kernel=kernel_recipe, **model_parameters_copy)


def create_anchored_ensemble_mlp(model_parameters: dict) -> regmodels.MLModel:
    """Creates an Anchored Ensemble MLP instance.
    Follows the implementation of a traditional MLP with additional anchoring regularization.
    Dictionary must contain the following keys:
        - n_models (int): Number of models in the ensemble.
        - in_feats (int): Number of input features.
        - out_feats (int): Number of output features.
        - hidden_layers (list): List of hidden layer sizes. Default is [64].
        - activation (str): Activation function to use. Default is 'relu'.
        - lambda_anchor (float): Anchoring regularization strength. Default is 1e-4.
        - lr (float): Learning rate for training. Default is 1e-3.
        - n_epochs (int): Number of training epochs. Default is 100.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        regmodels.MLModel: The created Anchored Ensemble MLP instance.
    """
    return regmodels.AnchoredEnsembleMLP(**model_parameters)


def create_knn_instance(model_parameters: dict) -> regmodels.MLModel:
    """Creates a k-Nearest Neighbors Regressor instance with Active Learning capabilities.
    Dictionary must contain the following keys:
        - k (int): Number of neighbors to use.
        - length_scale (float): Length scale for distance metric. Default is 1.0.
        - noise_floor (float): Noise floor to add to distance metric. Default is 1e-6.
        - distance_penalty (float): Penalty factor for distance in acquisition function. Default is 0.0.

    Args:
        model_parameters (dict): Configuration dictionary containing model parameters.

    Returns:
        regmodels.MLModel: The created k-Nearest Neighbors Regressor instance.
    """
    return regmodels.kNNRegressorAL(**model_parameters)


def create_bnn_instance(model_parameters: dict) -> regmodels.MLModel:
    """Creates a Bayesian Neural Network instance.
    Dictionary must contain the following keys:
        - in_feats (int): Number of input features.
        - out_feats (int): Number of output features.
        - hidden_layers (list): List of hidden layer sizes. Default is [32, 32].
        - activation (str): Activation function to use. Default is 'relu'.
        - seed (int): Random seed for reproducibility. Default is 42.
        - lr (float): Learning rate for training. Default is 1e-3.
        - epochs (int): Number of training epochs. Default is 100.
        - to_gpu (bool): Whether to use GPU for training. Default is True.

    For more details, see the `regmodels.BayesianNN` class.

    Args:
        model_parameters (dict): Configuration dictionary containing model parameters.

    Returns:
        regmodels.MLModel: The created Bayesian Neural Network instance.
    """
    return regmodels.BayesianNN(**model_parameters)


def sampling_block(
        X_candidates: np.ndarray,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        ml_model: regmodels.IndependentMultiPropertyModel,
        acquisition_params: list[dict],
        batch_selection_method: str,
        batch_selection_params: dict,
        penalization_params: tuple[float, float] = (0.25, 1.0),
        rng: np.random.Generator | None = None,
    ) -> tuple[list[int], np.ndarray, list[dict | None]]:
    """Samples new points from the acquisition landscape.

    A self-contained sampling unit: trains nothing, just takes a model and a
    candidate pool, runs one full cycle of acquisition + batch selection, and
    returns the selected indices, the per-entry padded landscapes, and
    per-entry metadata (resolved weights and ``y_best_z`` for joint entries).

    Args:
        X_candidates (np.ndarray): Candidate points, shape (M, d).
        X_train (np.ndarray): Training inputs, shape (N, d).
        Y_train (np.ndarray): Training targets, shape (N, P). Single-property
            callers should pass a 2-D array of shape (N, 1).
        ml_model (IndependentMultiPropertyModel): Trained multi-property model.
        acquisition_params (list[dict]): Per-entry acquisition parameters.
            Each entry must contain ``acquisition_mode`` and ``n_points``.
            Per-property entries carry ``target_variable``.
            Joint entries carry ``target_variables`` and either
            ``weight_sampler`` or ``weights``.
        batch_selection_method (str): Batch selection strategy
            (``highest_landscape`` | ``constant_liar`` | ``kriging_believer`` |
            ``local_penalization``). Only ``highest_landscape`` is tested with
            multi-property models.
        batch_selection_params (dict): Parameters for the batch selection strategy.
        penalization_params (tuple[float, float] | None): ``(radius, strength)``
            for soft landscape penalization. Pass ``None`` to disable.
        rng (np.random.Generator | None): Random generator for reproducible
            weight sampling in joint entries. Required when any entry contains a
            ``weight_sampler`` key; ignored otherwise.

    Returns:
        tuple:
            - ``list[int]``: Indices into ``X_candidates`` selected this cycle.
            - ``np.ndarray``: Landscape array of shape ``(n_entries, M)`` with
              each row padded to the full pool size.
            - ``list[dict | None]``: Per-entry metadata. For joint entries:
              ``{"_resolved_weights": np.ndarray, "_y_best_z": float,
              "target_variables": list[str]}``. For per-property / fast-path
              entries: ``None``.
    """
    M = len(X_candidates)

    # Y_train must be 2-D: (N, P)
    if Y_train.ndim == 1:
        Y_train = Y_train.reshape(-1, 1)

    X_train_copy = X_train.copy()

    # Per-property stats computed once per cycle (D8); used by joint entries.
    y_stats = compute_per_property_stats(Y_train, ml_model.target_names)

    # candidate_mask: True = available; shrinks across entries (D11).
    candidate_mask = np.ones(M, dtype=bool)

    sampled_new_idx: list[int] = []
    landscape_list: list[np.ndarray] = []
    per_entry_meta: list[dict | None] = []

    for acp in acquisition_params:
        acqui_param = acp.copy()
        n_points_per_style = acqui_param.pop('n_points')
        # target_variable is NOT popped: it flows into AcquisitionFunction.__init__
        # so that internal callers (e.g. batch_highest_landscape) use the right branch.
        target_variable    = acqui_param.get('target_variable', None)
        target_variables_e = acqui_param.get('target_variables', None)

        # ------------------------------------------------------------------
        # Weight resolution and y_best_z for joint entries (P2.4).
        # For per-property entries these stay None.
        # ------------------------------------------------------------------
        resolved_weights: np.ndarray | None = None
        y_best_z_val: float | None = None

        if target_variables_e is not None:
            weight_sampler_cfg = acqui_param.get('weight_sampler', None)
            weights_cfg        = acqui_param.get('weights', None)
            scal_method = acqui_param.get('scalarization', 'augmented_chebyshev')
            rho_val     = float(acqui_param.get('rho', 0.05))

            if weight_sampler_cfg is not None:
                if rng is None:
                    raise ValueError(
                        "rng (np.random.Generator) must be passed to sampling_block "
                        "when acquisition entries use 'weight_sampler'. "
                        "Create one with np.random.default_rng(seed) and pass it."
                    )
                ws = WeightSampler(
                    mode=weight_sampler_cfg['mode'],
                    n_properties=len(target_variables_e),
                    alpha=float(weight_sampler_cfg.get('alpha', 1.0)),
                )
                resolved_weights = ws.sample(rng)
            elif weights_cfg is not None:
                w = np.asarray(weights_cfg, dtype=float)
                resolved_weights = w / w.sum()
            else:
                entry_id = acqui_param.get('name', acqui_param.get('acquisition_mode'))
                raise ValueError(
                    f"Joint acquisition entry '{entry_id}' must have either "
                    "'weight_sampler' or 'weights'."
                )

            # Compute y_best_z: max scalarized observed value in training set.
            prop_indices = [ml_model.target_names.index(t) for t in target_variables_e]
            Y_sub     = Y_train[:, prop_indices]
            y_min_arr = np.array([y_stats[t][0] for t in target_variables_e])
            y_max_arr = np.array([y_stats[t][1] for t in target_variables_e])
            mu_z_obs, _ = scalarize(
                Y_sub, np.zeros_like(Y_sub), resolved_weights, y_min_arr, y_max_arr,
                method=scal_method, rho=rho_val,
            )
            y_best_z_val = float(mu_z_obs.max())

            # Inject resolved params into acqui_param so AcquisitionFunction.__init__
            # stores them and internal callers (batch_highest_landscape) pick them up.
            acqui_param['weights']  = resolved_weights
            acqui_param['y_stats']  = y_stats

        # Metadata returned to caller for logging.
        per_entry_meta.append(
            {
                '_resolved_weights': resolved_weights,
                '_y_best_z':         y_best_z_val,
                'target_variables':  target_variables_e,
            } if target_variables_e is not None else None
        )

        # ------------------------------------------------------------------
        # y_best and y_train_1d for this entry.
        # ------------------------------------------------------------------
        if target_variables_e is not None:
            y_best     = y_best_z_val
            y_train_1d = None       # only highest_landscape supported for joint entries
        elif target_variable is not None:
            prop_idx   = ml_model.target_names.index(target_variable)
            y_best     = float(np.max(Y_train[:, prop_idx]))
            y_train_1d = Y_train[:, prop_idx]
        else:
            prop_idx   = 0
            y_best     = float(np.max(Y_train[:, prop_idx]))
            y_train_1d = Y_train[:, prop_idx]

        # Sub-pool of candidates not yet selected by earlier entries.
        idx_map = np.flatnonzero(candidate_mask)   # sub-pool → full-pool index
        X_sub   = X_candidates[candidate_mask]

        # --- random fast-path ---
        if acqui_param['acquisition_mode'] == 'random':
            random_sub_ndx = np.random.choice(
                len(X_sub), size=n_points_per_style, replace=False
            )
            full_ndx = idx_map[random_sub_ndx]
            sampled_new_idx += list(full_ndx)
            landscape_list.append(np.zeros(M))
            candidate_mask[full_ndx] = False
            X_train_copy = np.concatenate([X_train_copy, X_candidates[full_ndx]])
            continue

        # --- maximum_predicted_value fast-path guard ---
        if acqui_param['acquisition_mode'] == 'maximum_predicted_value':
            assert n_points_per_style == 1, (
                "Number of points must be 1 when using maximum_predicted_value acquisition mode."
            )

        acqui_func = AcquisitionFunction(y_best=y_best, **acqui_param)

        # Compute landscape over the current sub-pool.
        # All dispatch params are already stored on acqui_func; no need to pass explicitly.
        landscape_sub = acqui_func.landscape_acquisition(
            X_candidates=X_sub,
            ml_model=ml_model,
        )
        landscape_sub = landscape_sanity_check(landscape_sub)

        # Pad to full pool size (unselected entries stay 0).
        landscape_full = np.zeros(M)
        landscape_full[candidate_mask] = landscape_sub
        landscape_list.append(landscape_full)

        # --- maximum_predicted_value fast-path ---
        if acqui_func.acquisition_mode == 'maximum_predicted_value':
            acq_mpv_sub_ndx = np.argmax(landscape_sub)
            full_ndx = idx_map[acq_mpv_sub_ndx]
            sampled_new_idx += [int(full_ndx)]
            candidate_mask[full_ndx] = False
            X_train_copy = np.concatenate([X_train_copy, X_candidates[[full_ndx]]])
            continue

        # Generic penalization of the sub-pool landscape.
        if penalization_params:
            radius, strength = penalization_params
            landscape_sub = penalize_landscape_fast(
                landscape=landscape_sub,
                X_candidates=X_sub,
                X_train=X_train_copy,
                radius=radius, strength=strength,
            )

        # Batch selection over the sub-pool.
        batch_selector = BatchSelectionStrategy(
            strategy_mode=batch_selection_method,
            strategy_params=batch_selection_params
        )
        sampled_sub_idx = batch_selector.batch_acquire(
            X_candidates=X_sub,
            model=ml_model,
            acquisition_function=acqui_func,
            batch_size=n_points_per_style,
            X_train=X_train_copy,
            y_train=y_train_1d,
        )

        # Map sub-pool indices back to full-pool and remove from future entries.
        full_idx = idx_map[sampled_sub_idx]
        sampled_new_idx += list(full_idx)
        candidate_mask[full_idx] = False
        X_train_copy = np.concatenate([X_train_copy, X_candidates[full_idx]])

    return sampled_new_idx, np.vstack(landscape_list), per_entry_meta


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


def create_acquisition_params(acquisition_params: list[dict], acquisition_protocol: dict, cycle: int) -> list[dict]:
    """Create acquisition parameters for a specific cycle.
    Each stage last for a number of cycles stated in the `cycles` key,
    e.g. stage_1 lasts for 3 cycles, stage_2 lasts for 3 cycles, etc.
    The function returns the acquisition parameters for the current cycle.

    Args:
        acquisition_params (list[dict]): List of acquisition parameters.
        acquisition_protocol (dict): Acquisition protocol defining stages and cycles.
        cycle (int): Current cycle number.

    Returns:
        list[dict]: Acquisition parameters for the current cycle.
    """
    total_cycles = sum([acquisition_protocol[stage]['cycles'] for stage in acquisition_protocol])
    assert cycle < total_cycles, f"Cycle number {cycle} exceeds total cycles {total_cycles} defined in the acquisition protocol."

    cycle_count = 0
    for stage in acquisition_protocol:
        n_cycles = acquisition_protocol[stage]['cycles']
        n_points = acquisition_protocol[stage]['n_points']
        modes = acquisition_protocol[stage]['acquisition_modes']

        #TODO allow for additional parameters specific for all acquisition modes

        # assert that the n_points tutple is the same length as the acquisition modes list
        assert len(modes) == len(n_points), \
            f"Number of acquisition modes {len(modes)} does not match number of n_points {len(n_points)} in stage {stage}."

        # update the acquisition_params with the n_points for the current stage
        for i, mode in enumerate(modes):
            for acq in acquisition_params:
                if acq['acquisition_mode'] == mode:
                    acq['n_points'] = n_points[i]

        if cycle < cycle_count + n_cycles:
            modes = acquisition_protocol[stage]['acquisition_modes']
            acq_params_for_cycle = [acq for acq in acquisition_params if acq['acquisition_mode'] in modes]

            return acq_params_for_cycle
        
        cycle_count += n_cycles

    return []  # Fallback return, should not reach here if assertions are correct


class AcquisitionParametersGenerator:

    def __init__(self, acquisition_params: list[dict], acquisition_protocol: dict, cycle_start_count: int = 0):
        self.acquisition_params = acquisition_params
        self.acquisition_protocol = acquisition_protocol
        self.total_cycles = sum([acquisition_protocol[stage]['cycles'] for stage in acquisition_protocol])
        self.cycle_start_count = cycle_start_count

        # self.current_stage = None
        # self.stage_cycle_count = 0
        # self.stage_n_cycles = 0
        # self.stage_modes = []
        # self.stage_n_points = []

        # assert that acquisition_params is provided and not empty
        assert acquisition_params and len(acquisition_params) > 0, "Acquisition parameters must be provided and not empty."

    @staticmethod
    def _entry_identifier(entry: dict) -> str:
        """Return the identifier for an acquisition entry.

        If the entry has an explicit ``name`` field, that is used; otherwise
        the ``acquisition_mode`` string is used as a fallback.  This allows
        protocol stages to reference entries by a human-readable ``name``
        (supporting multi-property configs with multiple entries of the same
        acquisition mode) while remaining backwards compatible with existing
        configs that have no ``name`` field.
        """
        return entry.get('name') or entry['acquisition_mode']

    def _protocol_params_for_cycle(self, cycle: int) -> dict:
        """Get acquisition parameters for a specific cycle, indipendently of the state of the class.
        Each stage last for a number of cycles stated in the `cycles` key:
        e.g. stage_1 lasts for 3 cycles, stage_2 lasts for 3 cycles, etc.
        The function returns the acquisition parameters for the current cycle.

        Args:
            cycle (int): Current cycle number.
        Returns:
            list[dict]: Acquisition parameters for the current cycle.
        """
        assert cycle < self.total_cycles, f"Cycle number {cycle} exceeds total cycles {self.total_cycles} defined in the acquisition protocol."

        cycle_count = self.cycle_start_count
        for stage in self.acquisition_protocol:
            n_cycles = self.acquisition_protocol[stage]['cycles']
            n_points = self.acquisition_protocol[stage]['n_points']
            modes = self.acquisition_protocol[stage]['acquisition_modes']

            # assert that the n_points list is the same length as the acquisition modes list
            assert len(modes) == len(n_points), \
                f"Number of acquisition modes {len(modes)} does not match number of n_points {len(n_points)} in stage {stage} (starting count: {self.cycle_start_count})."

            # update the acquisition_params with the n_points for the current stage,
            # matching by identifier (name or acquisition_mode fallback).
            for i, mode in enumerate(modes):
                for acq in self.acquisition_params:
                    if self._entry_identifier(acq) == mode:
                        acq['n_points'] = n_points[i]

            if cycle < cycle_count + n_cycles:
                modes = self.acquisition_protocol[stage]['acquisition_modes']
                acq_params_for_cycle = [
                    acq for acq in self.acquisition_params
                    if self._entry_identifier(acq) in modes
                ]
                return acq_params_for_cycle

            cycle_count += n_cycles

        return []


    def get_params_for_cycle(self, cycle: int) -> list[dict]:
        """Get acquisition parameters for a specific cycle.
        Follows the acquisition protocol if provided, otherwise returns the acquisition parameters as is.

        Args:
            cycle (int): Current cycle number.
        Returns:
            list[dict]: Acquisition parameters for the current cycle.
        """

        if self.acquisition_protocol and len(self.acquisition_protocol) > 0:
            return self._protocol_params_for_cycle(cycle)
        else:
            return self.acquisition_params
