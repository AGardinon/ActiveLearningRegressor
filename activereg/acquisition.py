#!

import numpy as np
from scipy.stats import norm
from scipy.spatial import cKDTree
from activereg.mlmodel import MLModel
from activereg.sampling import sample_landscape

# --------------------------------------------------------------
# Generic landscape sampling and penalization functions
# --------------------------------------------------------------

def highest_landscape_selection(
    landscape: np.ndarray, 
    percentile: int = 80,
    min_points: int = 50,
    max_points: int = 500
    ) -> np.ndarray:
    """Select the top percentile of the landscape with bounds on number of points.

    Args:
        landscape (np.ndarray): Input landscape array.
        percentile (int, optional): Percentile threshold for selection. Defaults to 80.
        min_points (int, optional): Minimum number of points to select. Defaults to 50.
        max_points (int, optional): Maximum number of points to select. Defaults to 500.

    Returns:
        np.ndarray: Indices of the selected points.
    """
    threshold = np.percentile(landscape, percentile)
    indices = np.where(landscape >= threshold)[0]
    
    # Adjust if too few/many points
    if len(indices) < min_points:
        # Lower threshold to get more points
        indices = np.argpartition(landscape, -min_points)[-min_points:]
    elif len(indices) > max_points:
        # Take top max_points from the selected region
        top_in_region = np.argpartition(landscape[indices], -max_points)[-max_points:]
        indices = indices[top_in_region]
    
    return indices


def penalize_landscape_fast(
    landscape: np.ndarray,
    X_candidates: np.ndarray,
    X_train: np.ndarray,
    radius: float = 0.1,
    strength: float = 1.0
    ) -> np.ndarray:
    """Fast version for large candidate sets using cKDTree.

    Args:
        landscape (np.ndarray): 
        X_candidates (np.ndarray): 
        X_train (np.ndarray): 
        radius (float, optional): . Defaults to 0.1.
        strength (float, optional): . Defaults to 1.0.

    Returns:
        np.ndarray: Corrected landscape.
    """
    if X_train.shape[0] == 0:
        return landscape

    # Build KDTree for screened points
    tree = cKDTree(X_train)

    # Query nearest distance for each candidate
    min_dists, _ = tree.query(X_candidates, k=1)

    # Gaussian penalty
    penalties = np.exp(- (min_dists**2) / (2 * radius**2))
    corrected = landscape * (1 - strength * penalties)
    
    return corrected

# --------------------------------------------------------------
# Acquisition function implementations
# --------------------------------------------------------------

class AcquisitionFunction:
    """
    Acquisition function class.
    """
    def __init__(self, acquisition_mode: str, y_best: float, **kwargs) -> None:
        """Initialize the acquisition function.

        Args:
            acquisition_mode (str): The mode of the acquisition function.
            y_best (float): The best observed value.
        kwargs: Additional parameters for specific acquisition functions.
        """
        self.acquisition_mode = acquisition_mode
        self.modes = ['upper_confidence_bound', 
                      'uncertainty_landscape', 
                      'maximum_predicted_value',
                      'expected_improvement',
                      'target_expected_improvement',
                      'percentage_target_expected_improvement',
                      'exploration_mutual_info']
        # check for numbered modes (e.g., expected_improvement_1, expected_improvement_2, etc.)
        mode_split = acquisition_mode.split('_')
        if mode_split[-1].isdigit() and '_'.join(mode_split[:-1]) in self.modes:
            self.acquisition_mode = '_'.join(mode_split[:-1])
        else:
            assert acquisition_mode in self.modes, f'Function "{acquisition_mode}" not implemented, choose from {self.modes.keys()}'
        # additional parameters
        self.y_best = y_best
        self.y_target = kwargs.get('y_target', y_best)
        self.kappa = kwargs.get('kappa', 2.0)
        self.xi = kwargs.get('xi', 1.e-2)
        self.dist = kwargs.get('dist', None)
        self.epsilon = kwargs.get('epsilon', None)
        self.tei_percentage = kwargs.get('percentage', None)
        # assertions
        if self.acquisition_mode == 'expected_improvement':
            assert self.xi >= 0, "`xi` must be non-negative."
        if self.acquisition_mode == 'target_expected_improvement':
            assert (self.dist is None) != (self.epsilon is None), "Provide exactly one of `d` (best closeness) or `epsilon` (band width)."
        if self.acquisition_mode == 'percentage_target_expected_improvement':
            assert self.tei_percentage is not None, "`percentage` must be provided for percentage_target_expected_improvement."


    def landscape_acquisition(self, X_candidates: np.ndarray, ml_model: MLModel) -> np.ndarray:
        """Generate an acquisition landscape for the given candidate points.
        The methods assume one of the activereg.mlmodel is used, where the predict statement
        automatically returns mean and standard dev.

        Args:
            X_candidates (np.ndarray): Candidate points for evaluation.
            ml_model (MLModel): Machine learning model for predictions (trained).

        Raises:
            ValueError: If the acquisition mode is not supported.

        Returns:
            np.ndarray: Acquisition landscape.
        """
        _, mu, sigma = ml_model.predict(X_candidates)

        if self.acquisition_mode == 'upper_confidence_bound':
            return upper_confidence_bound(mu=mu, sigma=sigma, kappa=self.kappa)
        
        elif self.acquisition_mode == 'uncertainty_landscape':
            return uncertainty_landscape(sigma=sigma)

        elif self.acquisition_mode == 'expected_improvement':
            return expected_improvement(mu=mu, sigma=sigma, y_best=self.y_best, xi=self.xi)
        
        elif self.acquisition_mode == 'target_expected_improvement':
            return target_expected_improvement(mu=mu, sigma=sigma, y_target=self.y_target, dist=self.dist, epsilon=self.epsilon)
        
        elif self.acquisition_mode == 'percentage_target_expected_improvement':
            return percentage_target_expected_improvement(mu=mu, sigma=sigma, y_best=self.y_best, percentage=self.tei_percentage)
        
        elif self.acquisition_mode == 'exploration_mutual_info':
            try:
                noise_var = ml_model.model.kernel_.k2.noise_level
            except AttributeError:
                raise ValueError('noise_level not found, GPR needs to be trained with a WhiteKernel().')
            return exploration_mutual_info(sigma=sigma, noise_var=noise_var)

        elif self.acquisition_mode == 'maximum_predicted_value':
            return maximum_predicted_value(X_candidates=X_candidates, ml_model=ml_model)
        

def upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray, kappa: float=2.0) -> np.ndarray:
    """Acquisition function: Upper Confidence Bound (UCB) function.

    Args:
        mu (np.ndarray): Mean predictions.
        sigma (np.ndarray): Standard deviation of the predictions.
        kappa (float, optional): Exploration-exploitation tradeoff parameter. Defaults to 2.0.

    Returns:
        np.ndarray: UCB scores.
    """
    return mu + kappa * sigma


def uncertainty_landscape(sigma: np.ndarray) -> np.ndarray:
    """Explore using the model uncertainty landscape

    Args:
        sigma (np.ndarray): Standard deviation of the predictions.

    Returns:
        np.ndarray: Uncertainty scores.
    """
    return sigma


def exploration_mutual_info(sigma: np.ndarray, noise_var: float) -> np.ndarray:
    """Score(x) = 0.5 * log(1 + sigma_f^2 / noise_var).
    `noise_var` = observational noise variance (e.g., WhiteKernel(noise_level) from sklearn GPR).

    Args:
        sigma (np.ndarray): Standard deviation of the predictions.
        noise_var (float): Observational noise variance.

    Returns:
        np.ndarray: Exploration scores.
    """
    sigma_y = np.maximum(sigma, 1e-12)
    sigma_f2 = np.maximum(sigma_y**2 - noise_var, 0.0)
    score = 0.5 * np.log1p(sigma_f2 / np.maximum(noise_var, 1e-12))
    return score


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    """Acquisition function: Expected Improvement (EI).

    Args:
        mu (np.ndarray): Mean predictions.
        sigma (np.ndarray): Standard deviation of the predictions.
        y_best (float): Best observed value.
        xi (float, optional): Exploration-exploitation tradeoff parameter. Defaults to 0.01.

    Returns:
        np.ndarray: EI scores.
    """
    # Avoid division by zero
    sigma = sigma.clip(min=1e-9)

    # Compute Z-score
    Z = (mu - y_best - xi) / sigma
    
    # Compute EI using the normal CDF (\Phi) and PDF (\phi)
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    # Ensure non-negative values (since EI is max(0, ...))
    return np.maximum(ei, 0)


def target_expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_target: float, *,
    dist: float=None,        # use this for best-closeness TEI: d = current best distance to target
    epsilon: float=None,     # use this for band TEI: epsilon = tolerance
    clip_sigma: float=1e-12
    ) -> np.ndarray:
    """Mathematically derived Targeted Expected Improvement (TEI).
    Improvement: (d - |Y - t|)_+ where d = epsilon (band TEI) OR d = best_closeness (best-TEI).

    Args:
        mu (np.ndarray): Mean predictions.
        sigma (np.ndarray): Standard deviation of the predictions.
        y_target (float): Target value t.
        epsilon (float, optional): Tolerance half-width. Defaults to None.
        dist (float, optional): Current best distance to target. Defaults to None.

    Raises:
        ValueError: If neither `d` nor `epsilon` is provided.

    Returns:
        np.ndarray: TEI scores.
    """
    if (dist is None) == (epsilon is None):
        raise ValueError("Provide exactly one of `d` (best closeness) or `epsilon` (band width).")

    d_val = float(epsilon if dist is None else dist)
    sigma = np.clip(sigma, clip_sigma, None)

    t = float(y_target)
    a = t - d_val
    b = t + d_val

    z1 = (a - mu) / sigma
    z0 = (t - mu) / sigma
    z2 = (b - mu) / sigma

    Phi = norm.cdf
    phi = norm.pdf

    dPhi1 = Phi(z0) - Phi(z1)
    dPhi2 = Phi(z2) - Phi(z0)

    tei = ( (d_val - t + mu) * dPhi1
          + (d_val + t - mu) * dPhi2
          + sigma * (phi(z1) - 2.0 * phi(z0) + phi(z2)) )

    # numerical safety
    tei = np.maximum(tei, 0.0)
    return tei


def percentage_target_expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, percentage: float) -> np.ndarray:
    """Percentage Targeted Expected Improvement (TEI) acquisition function.

    Args:
        mu (np.ndarray): Mean predictions.
        sigma (np.ndarray): Standard deviation of the predictions.
        y_best (float): Best observed value.
        percentage (float): Percentage for target value adjustment.

    Returns:
        np.ndarray: TEI scores.
    """
    y_target = y_best * (1 - percentage / 100)
    return target_expected_improvement(mu, sigma, y_target, dist=abs(y_best - y_target))


def maximum_predicted_value(mu: np.ndarray) -> np.ndarray:
    """Acquisition function: Maximum Predicted Value (MPV).
    Returns a landscape that is zero everywhere except at the maximum predicted value.

    Args:
        mu (np.ndarray): Mean predictions.

    Returns:
        np.ndarray: MPV scores.
    """
    mpv = np.zeros_like(mu)
    mpv[mu.argmax()] = 1.0
    return mpv

# --------------------------------------------------------------
# Batch acquisition strategies
# --------------------------------------------------------------

def landscape_sanity_check(landscape: np.ndarray) -> np.ndarray:
    """Check and adjust the shape of the acquisition landscape.

    Args:
        landscape (np.ndarray): Input landscape array.
    Returns:
        np.ndarray: Adjusted landscape array.
    """
    if len(landscape.shape) > 1 and landscape.shape[1] == 1:
        landscape = landscape.ravel()
        return landscape
    
    elif len(landscape.shape) > 1 and landscape.shape[1] > 1:
        raise ValueError("Landscape shape is multi dimensional. "
        "Expected 1D array or 2D array with single column. "
        "Multi output acquisition functions are not supported.")
    
    return landscape


class BatchSelectionStrategy:
    """
    Batch selection strategy class.
    """
    def __init__(self, strategy_mode: str, **kwargs) -> None:
        """Initialize the batch selection strategy.

        Args:
            strategy_mode (str): The mode of the batch selection strategy.

        kwargs: Additional parameters for specific strategies.
        Parameters for different strategies:
            - Highest Landscape Sampling:
                - percentile (int): Percentile for highest landscape selection.
                - sampling_method (str): Sampling method ('random', 'kmeans', 'voronoi').
            - Constant Liar:
                - lie_type (str): Type of lie value ('max', 'min', 'mean').
                - lie_value (float): Specific lie value to use.
            - Local Penalization:
                - L (float): Lipschitz constant.
                - alpha (float): Penalization strength.
        """
        self.strategy_mode = strategy_mode
        self.modes = ['highest_landscape', 
                      'constant_liar', 
                      'kriging_believer',
                      'local_penalization']
        assert strategy_mode in self.modes, f'Function "{strategy_mode}" not implemented, choose from {self.modes}'
        # additional parameters
        self.percentile = kwargs.get('percentile', 95)
        self.sampling_method = kwargs.get('sampling_method', 'voronoi')
        self.lie_type = kwargs.get('lie_type', 'max')
        self.lie_value = kwargs.get('lie_value', None)
        self.L = kwargs.get('L', 1.0)
        self.alpha = kwargs.get('alpha', 1.0)

    def batch_acquire(
        self,
        X_candidates: np.ndarray,
        model: MLModel,
        acquisition_function: AcquisitionFunction,
        batch_size: int,
        #
        X_train: np.ndarray = None,
        y_train: np.ndarray = None
    ) -> np.ndarray:
        """Perform batch acquisition based on the selected strategy.

        Args:
            X_candidates (np.ndarray): Candidate points.
            model (MLModel): Machine learning model.
            acquisition_function (AcquisitionFunction): Acquisition function instance.
            batch_size (int): Number of points to acquire.
            X_train (np.ndarray, optional): Training input points. Defaults to None.
            y_train (np.ndarray, optional): Training target values. Defaults to None.

        Returns:
            np.ndarray: Indices of selected points.
        """
        if self.strategy_mode == 'highest_landscape':
            return batch_highest_landscape(
                X_candidates=X_candidates,
                model=model,
                acquisition_function=acquisition_function,
                batch_size=batch_size,
                percentile=self.percentile,
                sampling_method=self.sampling_method
            )
        
        elif self.strategy_mode == 'constant_liar':
            if self.lie_value is None:
                # Determine lie value based on lie type
                if self.lie_type == 'max':
                    _, mu, _ = model.predict(X_candidates)
                    self.lie_value = np.max(mu)
                elif self.lie_type == 'min':
                    _, mu, _ = model.predict(X_candidates)
                    self.lie_value = np.min(mu)
                elif self.lie_type == 'mean':
                    _, mu, _ = model.predict(X_candidates)
                    self.lie_value = np.mean(mu)
                else:
                    raise ValueError(f'Lie type "{self.lie_type}" not recognized. Choose from "max", "min", "mean".')
                
            return batch_constant_liar(
                X_candidates=X_candidates,
                model=model,
                acquisition_function=acquisition_function,
                batch_size=batch_size,
                X_train=X_train,
                y_train=y_train,
                lie_value=self.lie_value
            )
        
        elif self.strategy_mode == 'kriging_believer':
            return batch_kriging_believer(
                X_candidates=X_candidates,
                model=model,
                acquisition_function=acquisition_function,
                batch_size=batch_size,
                X_train=X_train,
                y_train=y_train
            )
        
        elif self.strategy_mode == 'local_penalization':
            return batch_local_penalization(
                X_candidates=X_candidates,
                model=model,
                acquisition_function=acquisition_function,
                batch_size=batch_size,
                L=self.L,
                alpha=self.alpha
            )
        
    def __str__(self) -> str:
        return f'BatchSelectionStrategy(strategy_mode={self.strategy_mode})'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def get_params(self) -> dict:
        param_dict = {
            'strategy_mode': self.strategy_mode,
            'percentile': self.percentile,
            'sampling_method': self.sampling_method,
            'lie_type': self.lie_type,
            'lie_value': self.lie_value,
            'L': self.L,
            'alpha': self.alpha
        }
        if self.strategy_mode != 'highest_landscape':
            param_dict.pop('percentile')
            param_dict.pop('sampling_method')
        if self.strategy_mode != 'constant_liar':
            param_dict.pop('lie_type')
            param_dict.pop('lie_value')
        if self.strategy_mode != 'local_penalization':
            param_dict.pop('L')
            param_dict.pop('alpha')
        return param_dict
    

# Highest landscape hypersurface sampling
def batch_highest_landscape(
    X_candidates: np.ndarray,
    model: MLModel,
    acquisition_function: AcquisitionFunction,
    batch_size: int,
    #
    percentile: int,
    sampling_method: str = 'voronoi',
) -> np.ndarray:
    """Batch acquisition using Highest Landscape Sampling strategy.

    Args:
        X_candidates (np.ndarray): Candidate points.
        model (MLModel): Machine learning model.
        acquisition_function (AcquisitionFunction): Acquisition function instance.
        batch_size (int): Number of points to acquire.
        percentile (int): Percentile for highest landscape selection.
        sampling_method (str, optional): Sampling method. Defaults to 'voronoi'.
    Returns:
        np.ndarray: Indices of selected points, referred to the original candidate set.
    """
    # Make a copy to avoid modifying original data
    X_candidates_tmp = X_candidates.copy()

    # Track original candidate indexes
    X_candidates_indexes = np.arange(X_candidates.shape[0])

    # Compute landscape
    landscape = acquisition_function.landscape_acquisition(X_candidates_tmp, model)
    landscape = landscape_sanity_check(landscape)

    # Select top percentile points
    candidate_indices = highest_landscape_selection(landscape=landscape, percentile=percentile)
    X_candidates_selected = X_candidates_tmp[candidate_indices]
    X_candidates_selected_indices = X_candidates_indexes[candidate_indices]

    # Select points from the selected candidates
    sampled_indices = sample_landscape(
        X_landscape=X_candidates_selected,
        n_points=batch_size,
        method=sampling_method
    )
    sampled_new_indices = X_candidates_selected_indices[sampled_indices]

    return sampled_new_indices


# Batch acquisition using Constant Liar strategy
# faithful to the CL literature (Ginsbourger et al.)
def batch_constant_liar(
    X_candidates: np.ndarray,
    model: MLModel,
    acquisition_function: AcquisitionFunction,
    batch_size: int,
    #
    X_train: np.ndarray,
    y_train: np.ndarray,
    lie_value: float = None
) -> np.ndarray:
    """Batch acquisition using Constant Liar strategy.

    Args:
        X_candidates (np.ndarray): Candidate points.
        model (MLModel): Machine learning model.
        acquisition_function (AcquisitionFunction): Acquisition function instance.
        batch_size (int): Number of points to acquire.
        X_train (np.ndarray): Training input points.
        y_train (np.ndarray): Training target values.
        lie_value (float, optional): Lie value to use. Defaults to None.
    Returns:
        np.ndarray: Indices of selected points, referred to the original candidate set.
    """
    selected_indices = []
    
    # Make local copies to avoid modifying original data
    X_train_copy = X_train.copy()
    y_train_copy = y_train.copy()
    X_candidates_copy = X_candidates.copy()
    model_copy = model

    # Track original candidate indexes
    X_candidates_indexes = np.arange(X_candidates.shape[0])

    for _ in range(batch_size):
        # Compute landscape (the methods uses MLModel.predict that returns mu,sigma internally)
        landscape_tmp = acquisition_function.landscape_acquisition(X_candidates_copy, model_copy)
        landscape_tmp = landscape_sanity_check(landscape_tmp)

        # Select the best candidate
        best_idx = np.argmax(landscape_tmp)

        # Map back to original candidate index
        original_idx = X_candidates_indexes[best_idx]
        selected_indices.append(original_idx)

        # Determine the lie value
        if lie_value is None:
            # Use the predicted mean at the selected point as the lie value
            _, mu_best, _ = model_copy.predict(X_candidates_copy[best_idx].reshape(1, -1))
            y_lie = mu_best[0]
        else:
            y_lie = np.array([lie_value], dtype=float)

        # Update the temporary training set with the selected point and lie value
        X_train_copy = np.vstack([X_train_copy, X_candidates_copy[best_idx]])
        y_train_copy = np.hstack([y_train_copy, y_lie])

        # Retrain the model with the updated training set
        model_copy.train(X_train_copy, y_train_copy)
        # Remove the selected point from the candidate set
        X_candidates_copy = np.delete(X_candidates_copy, best_idx, axis=0)
        X_candidates_indexes = np.delete(X_candidates_indexes, best_idx, axis=0)

    return np.array(selected_indices)


# Batch acquisition using Kriging Believer strategy
def batch_kriging_believer(
    X_candidates: np.ndarray,
    model: MLModel,
    acquisition_function: AcquisitionFunction,
    batch_size: int,
    #
    X_train: np.ndarray,
    y_train: np.ndarray
) -> np.ndarray:
    """Batch acquisition using Kriging Believer strategy.

    Args:
        X_candidates (np.ndarray): Candidate points.
        model (MLModel): Machine learning model.
        acquisition_function (AcquisitionFunction): Acquisition function instance.
        batch_size (int): Number of points to acquire.
        X_train (np.ndarray): Training input points.
        y_train (np.ndarray): Training target values.

    Returns:
        np.ndarray: Indices of selected points, referred to the original candidate set.
    """
    sampled_new_idx = []
    
    # Make local copies to avoid modifying original data
    X_train_copy = X_train.copy()
    y_train_copy = y_train.copy()
    X_candidates_copy = X_candidates.copy()
    model_copy = model

    # Track original candidate indexes
    X_candidates_indexes = np.arange(X_candidates.shape[0])

    for _ in range(batch_size):
        # Compute landscape (the methods uses MLModel.predict that returns mu,sigma internally)
        landscape_tmp = acquisition_function.landscape_acquisition(X_candidates_copy, model_copy)
        landscape_tmp = landscape_sanity_check(landscape_tmp)

        # Select the best candidate
        sampled_hls_idx = np.argmax(landscape_tmp)
        sampled_new_idx.append(X_candidates_indexes[sampled_hls_idx])

        # Predict the value at the selected point
        _, mu_best, _ = model_copy.predict(X_candidates_copy[sampled_hls_idx].reshape(1, -1))
        y_believe = mu_best[0]

        # Update the temporary training set with the selected point and believed value
        X_train_copy = np.vstack([X_train_copy, X_candidates_copy[sampled_hls_idx]])
        y_train_copy = np.hstack([y_train_copy, y_believe])

        # Retrain the model with the updated training set
        model_copy.train(X_train_copy, y_train_copy)

        # Remove the selected point from the candidate set
        X_candidates_copy = np.delete(X_candidates_copy, sampled_hls_idx, axis=0)
        X_candidates_indexes = np.delete(X_candidates_indexes, sampled_hls_idx, axis=0)

    return np.array(sampled_new_idx)


# Batch acquisition using Local Penalization strategy
def batch_local_penalization(
    X_candidates: np.ndarray,
    model: MLModel,
    acquisition_function: AcquisitionFunction,
    batch_size: int,
    #
    L: float = 1.0,
    alpha: float = 1.0
) -> np.ndarray:
    """Batch acquisition using Local Penalization strategy.

    Args:
        X_candidates (np.ndarray): Candidate points.
        model (MLModel): Machine learning model.
        acquisition_function (AcquisitionFunction): Acquisition function instance.
        batch_size (int): Number of points to acquire.
        L (float, optional): Lipschitz constant. Defaults to 1.0.
        alpha (float, optional): Penalization strength. Defaults to 1.0.

    Returns:
        np.ndarray: Indices of selected points, referred to the original candidate set.
    """
    selected_indices = []
    selected_points = []

    X_candidates_copy = X_candidates.copy()
    X_candidates_indexes = np.arange(len(X_candidates))
    model_copy = model

    # Compute initial acquisition landscape
    landscape = acquisition_function.landscape_acquisition(X_candidates_copy, model_copy)
    landscape = landscape_sanity_check(landscape)

    for _ in range(batch_size):
        # Start from baseline landscape
        landscape_tmp = landscape.copy()

        # Apply penalization for each already selected point
        # Simple linear penalization based on distance and Lipschitz constant
        for xp in selected_points:
            d = np.linalg.norm(X_candidates_copy - xp, axis=1)
            R = alpha / L  # radius based on Lipschitz constant
            phi = np.minimum(1.0, d / R)  # linear decay
            landscape_tmp *= phi  # apply penalization

        # Select the best candidate from the penalized landscape
        best_idx = np.argmax(landscape_tmp)
        xp_best = X_candidates_copy[best_idx]

        original_idx = X_candidates_indexes[best_idx]
        selected_indices.append(original_idx)

        selected_points.append(xp_best)

        # Remove the selected point from the candidate set and landscape for next iteration
        X_candidates_copy = np.delete(X_candidates_copy, best_idx, axis=0)
        X_candidates_indexes = np.delete(X_candidates_indexes, best_idx, axis=0)
        landscape = np.delete(landscape, best_idx, axis=0)

    return np.array(selected_indices)