#!

import numpy as np
import pandas as pd
import activereg.mlmodel as regmodels
from pathlib import Path
from sklearn.base import BaseEstimator
from activereg.format import DATASETS_REPO
from activereg.sampling import sample_landscape
from activereg.hyperparams import get_gp_kernel
from activereg.acquisition import (
    penalize_landscape_fast,
    highest_landscape_selection,
    AcquisitionFunction,
    landscape_sanity_check,
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


def setup_experiment_variables(config: dict) -> tuple[str, str, int, int, str, str, list[dict]]:
    """Sets up the experiment parameters from the config dictionary.
    The configuration dictionary must contain the following keys:
    - experiment_name (str): Name of the experiment.
    - experiment_notes (str): Additional notes for the experiment.
    - n_cycles (int): Number of active learning cycles.
    - init_batch_size (int): Initial batch size.
    - init_sampling (str): Initial sampling method.
    - cycle_sampling (str): Cycle sampling method.
    - acquisition_parameters (list[dict]): Acquisition function parameters.

    Args:
        config (dict): Configuration dictionary containing experiment parameters.

    Returns:
        tuple: Contains experiment name, additional notes, number of cycles, initial batch size,
               initial sampling method, cycle sampling method, and acquisition parameters.
    """
    exp_name = config.get('experiment_name', None)
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
    The configuration dictionary must contain the following keys:
        - ml_model (str): Type of machine learning model ('GPR', 'AnchoredEnsembleMLP', 'BayesianNN').
        - model_parameters (dict): Parameters specific to the chosen model.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        regmodels.MLModel: The configured machine learning model.
    """
    ml_model_type = config.get('ml_model', None)
    ml_model_params = config.get('model_parameters', {})
    assert ml_model_type is not None, "ML model type must be specified in the config file."

    if ml_model_type == 'GPR':
        return create_gpr_instance(ml_model_params)
    elif ml_model_type == 'AnchoredEnsembleMLP':
        return create_anchored_ensemble_mlp(ml_model_params)
    elif ml_model_type == 'BayesianNN':
        return create_bnn_instance(ml_model_params)
    else:
        raise ValueError(f"Unknown ML model type: {ml_model_type}. Supported types are: ['GPR', 'AnchoredEnsembleMLP', 'BayesianNN'].")


def create_gpr_instance(model_parameters: dict) -> regmodels.MLModel:
    """Creates a Gaussian Process Regressor instance.
    Dictionary must contain the following keys:
        - kernel_recipe (str or list): Recipe for the GP kernel. See `get_gp_kernel` and `KernelFactory` for details.
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
    kernel_recipe = model_parameters_copy.pop('kernel_recipe', None)
    assert kernel_recipe is not None, "Kernel recipe must be specified for GPR in the config file."

    kernel_recipe = get_gp_kernel(kernel_recipe)

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
        y_best: float, 
        ml_model: regmodels.MLModel, 
        acquisition_params: list[dict],
        penalization_params: tuple[float, float] = (0.25, 1.0),
        sampling_mode: str = 'voronoi'
    ) -> tuple[list[int], np.ndarray]:
    """Samples new points from the landscape of the acquisition function.

    Args:
        X_candidates (np.ndarray): candidates points for the cycle.
        X_train (np.ndarray): training points from the cycle.
        y_best (float): best value of the target variable.
        ml_model (regmodels.MLModel): machine learning model used for the experiment.
        acquisition_params (list[dict]): acquisition function parameters for the cycle.
        penalization_params (tuple[float, float], optional): penalization parameters for the landscape. Defaults to (0.25, 1.0).
        sampling_mode (str, optional): sampling mode for the landscape. Defaults to 'voronoi'.

    Returns:
        tuple[list[int], np.ndarray]: new sampled indexes and the landscapes of the acquisition function.
    """

    # init variables
    X_candidates_indexes = np.arange(0,len(X_candidates))
    X_train_copy = X_train.copy()

    sampled_new_idx = []
    landscape_list = []

    # loop over acquisition function types
    for acp in acquisition_params:
        acqui_param = acp.copy()
        n_points_per_style = acqui_param['n_points']
        percentile = acqui_param.pop('percentile')

        # if acquisition mode is "random", sample random points and continue
        if acqui_param['acquisition_mode'] == 'random':
            random_ndx = np.random.choice(
                X_candidates_indexes, 
                size=n_points_per_style, 
                replace=False
            )
            sampled_new_idx += list(random_ndx)
            landscape_list.append(np.zeros(len(X_candidates)))  # append a zero landscape for consistency
            X_train_copy = np.concatenate([X_train_copy, X_candidates[random_ndx]])
            continue

        # assert that if the acquisition mode is mpv the number of points is 1
        if acqui_param['acquisition_mode'] == 'maximum_predicted_value':
            assert n_points_per_style == 1, "Number of points must be 1 when using maximum_predicted_value acquisition mode."

        acqui_func = AcquisitionFunction(y_best=y_best, **acqui_param)
        # Compute pure landscape for possible output/analysis (batch methods can modify it)
        landscape = acqui_func.landscape_acquisition(X_candidates=X_candidates, ml_model=ml_model)
        landscape = landscape_sanity_check(landscape)
        landscape_list.append(landscape)

        # skip if acquisition mode is maximum_predicted_value
        if acqui_func.acquisition_mode == 'maximum_predicted_value':
            acq_mpv_ndx = np.argmax(landscape)
            sampled_new_idx += [X_candidates_indexes[acq_mpv_ndx]]
            landscape_list.append(landscape)
            X_train_copy = np.concatenate([X_train_copy, X_candidates[[acq_mpv_ndx]]])
            continue

        # Generic penalization of the landscape to add soft avoidance of sampled points
        # Generally can be skipped for batch methods that have their own penalization
        if penalization_params:
            radius, strength = penalization_params
            landscape = penalize_landscape_fast(
                landscape=landscape,
                X_candidates=X_candidates,
                X_train=X_train_copy,
                radius=radius, strength=strength,
            )

        # TODO: add here option for batch selection methods (e.g., CL, etc.)
        # TODO: write functional form for batch selection methods to clean up the code
        acq_landscape_ndx = highest_landscape_selection(landscape=landscape, percentile=percentile)
        X_acq_landscape = X_candidates[acq_landscape_ndx]
        X_acq_landscape_indexes = X_candidates_indexes[acq_landscape_ndx]

        sampled_hls_idx = sample_landscape(
            X_landscape=X_acq_landscape, 
            n_points=n_points_per_style, 
            sampling_mode=sampling_mode
        )

        sampled_new_idx += list(X_acq_landscape_indexes[sampled_hls_idx])

        X_train_copy = np.concatenate([X_train_copy, X_candidates[sampled_new_idx]])

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

            #TODO allow for additional parameters specific for all acquisition modes

            # assert that the n_points list is the same length as the acquisition modes list
            assert len(modes) == len(n_points), \
                f"Number of acquisition modes {len(modes)} does not match number of n_points {len(n_points)} in stage {stage} (starting count: {self.cycle_start_count})."
            
            # update the acquisition_params with the n_points for the current stage
            for i, mode in enumerate(modes):
                for acq in self.acquisition_params:
                    if acq['acquisition_mode'] == mode:
                        acq['n_points'] = n_points[i]

            if cycle < cycle_count + n_cycles:
                modes = self.acquisition_protocol[stage]['acquisition_modes']
                acq_params_for_cycle = [acq for acq in self.acquisition_params if acq['acquisition_mode'] in modes]

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
