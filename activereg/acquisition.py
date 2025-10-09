#!

import torch
import numpy as np
from scipy.stats import norm
from scipy.spatial import cKDTree
# from sklearn.metrics import pairwise_distances
from typing import Tuple

def highest_landscape_selection(landscape: np.ndarray, percentile: int=80):
    """
    Select the top percentile of the landscape (distribution) and return
    their indexes
    """
    threshold = np.percentile(landscape, percentile)
    return np.where(landscape >= threshold)[0]


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


class AcquisitionFunction:
    """
    Acquisition function class.
    """
    def __init__(self, acquisition_mode: str, y_best: float, **kwargs):
        self.acquisition_mode = acquisition_mode
        self.modes = ['upper_confidence_bound', 
                      'uncertainty_landscape', 
                      'maximum_predicted_value',
                      'expected_improvement',
                      'target_expected_improvement',
                      'percentage_target_expected_improvement',
                      'exploration_mutual_info']
        
        # add possibility to handle an acquisition mode that ends with f'_{N}' for multuple definition of the 
        # same acquisition function with different parameters (e.g. percentage_target_expected_improvement_5)
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

    def landscape_acquisition(self, X_candidates: np.ndarray, ml_model):
        if self.acquisition_mode == 'upper_confidence_bound':
            return upper_confidence_bound(X_candidates=X_candidates, ml_model=ml_model, kappa=self.kappa)
        
        elif self.acquisition_mode == 'uncertainty_landscape':
            return uncertainty_landscape(X_candidates=X_candidates, ml_model=ml_model)

        elif self.acquisition_mode == 'expected_improvement':
            return expected_improvement(X_candidates=X_candidates, 
                                        ml_model=ml_model, 
                                        y_best=self.y_best, 
                                        xi=self.xi)
        
        elif self.acquisition_mode == 'target_expected_improvement':
            return target_expected_improvement(X_candidates=X_candidates, 
                                               ml_model=ml_model, 
                                               y_target=self.y_target, 
                                               dist=self.dist, 
                                               epsilon=self.epsilon)
        
        elif self.acquisition_mode == 'percentage_target_expected_improvement':
            return percentage_target_expected_improvement(X_candidates=X_candidates, 
                                                          ml_model=ml_model, 
                                                          y_best=self.y_best, 
                                                          percentage=self.tei_percentage)
        
        elif self.acquisition_mode == 'exploration_mutual_info':
            return exploration_mutual_info(X_candidates=X_candidates, ml_model=ml_model)
        
        elif self.acquisition_mode == 'maximum_predicted_value':
            return maximum_predicted_value(X_candidates=X_candidates, ml_model=ml_model)

# '''
# The methods assume one of the activereg.mlmodel is used, where the predict statment
# automatically returns mean and standard dev.
# '''

def upper_confidence_bound(X_candidates: np.ndarray, ml_model, kappa: float=2.0) -> Tuple[np.ndarray]:
    """Acquisition function: Upper Confidence Bound (UCB) function.

    Args:
        X_candidates (np.ndarray): Candidate points for evaluation.
        ml_model (_type_): Trained model for predictions.
        kappa (float, optional): Exploration-exploitation tradeoff parameter. Defaults to 2.0.

    Returns:
        Tuple[np.ndarray]: Mean predictions and UCB scores.
    """
    _, mu, sigma = ml_model.predict(X_candidates)
    return mu, mu + kappa * sigma


def uncertainty_landscape(X_candidates: np.ndarray, ml_model) -> Tuple[np.ndarray]:
    """Explore using the model uncertainty landscape

    Args:
        X_candidates (np.ndarray): Candidate points for evaluation.
        ml_model (_type_): Trained model for predictions.

    Returns:
        Tuple[np.ndarray]: Mean predictions and uncertainty scores.
    """
    _, mu, sigma = ml_model.predict(X_candidates)
    return mu, sigma


def exploration_mutual_info(X_candidates: np.ndarray, ml_model) -> Tuple[np.ndarray]:
    """Score(x) = 0.5 * log(1 + sigma_f^2 / noise_var).
    `noise_var` = observational noise variance (e.g., WhiteKernel(noise_level) from sklearn GPR).

    Args:
        X_candidates (np.ndarray): Candidate points for evaluation.
        ml_model (_type_): Trained model for predictions.

    Raises:
        ValueError: If noise_level is not found.

    Returns:
        Tuple[np.ndarray]: Mean predictions and exploration scores.
    """
    try:
        noise_var = ml_model.model.kernel_.k2.noise_level
    except AttributeError:
        raise ValueError('noise_level not found, GPR needs to be trained with a WhiteKernel().')

    _, mu, sigma_y = ml_model.predict(X_candidates)
    sigma_y = np.maximum(sigma_y, 1e-12)
    sigma_f2 = np.maximum(sigma_y**2 - noise_var, 0.0)
    score = 0.5 * np.log1p(sigma_f2 / np.maximum(noise_var, 1e-12))
    return mu, score


def expected_improvement(X_candidates: np.ndarray, ml_model, y_best: float, xi: float = 0.01) -> Tuple[np.ndarray]:
    """Acquisition function: Expected Improvement (EI).

    Args:
        X_candidates (np.ndarray): Candidate points for evaluation.
        ml_model (_type_): Trained model for predictions.
        y_best (float): Best observed value.
        xi (float, optional): Exploration-exploitation tradeoff parameter. Defaults to 0.01.

    Returns:
        Tuple[np.ndarray]: Mean predictions and EI scores.
    """

    # Get mean and standard deviation from the GP model
    _, mu, sigma = ml_model.predict(X_candidates)
    
    # Avoid division by zero
    sigma = sigma.clip(min=1e-9)

    # Compute Z-score
    Z = (mu - y_best - xi) / sigma
    
    # Compute EI using the normal CDF (\Phi) and PDF (\phi)
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    # Ensure non-negative values (since EI is max(0, ...))
    return mu, np.maximum(ei, 0)


def target_expected_improvement(X_candidates: np.ndarray, ml_model, y_target: float, *,
                dist: float=None,        # use this for best-closeness TEI: d = current best distance to target
                epsilon: float=None,     # use this for band TEI: epsilon = tolerance
                clip_sigma: float=1e-12) -> Tuple[np.ndarray]:
    """Mathematically derived Targeted Expected Improvement (TEI).
    Improvement: (d - |Y - t|)_+ where d = epsilon (band TEI) OR d = best_closeness (best-TEI).

    Args:
        X_candidates (np.ndarray): Candidate points for evaluation.
        ml_model (_type_): Trained model for predictions.
        y_target (float): Target value t
        epsilon (float, optional): Tolerance half-width. Defaults to None.
        dist (float, optional): Current best distance to target. Defaults to None.

    Raises:
        ValueError: If neither `d` nor `epsilon` is provided.

    Returns:
        Tuple[np.ndarray]: Mean predictions and TEI scores.
    """
    if (dist is None) == (epsilon is None):
        raise ValueError("Provide exactly one of `d` (best closeness) or `epsilon` (band width).")

    d_val = float(epsilon if dist is None else dist)
    _, mu, sigma = ml_model.predict(X_candidates)
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
    return mu, tei


def percentage_target_expected_improvement(X_candidates: np.ndarray, ml_model, y_best: float, percentage: float) -> Tuple[np.ndarray]:
    """Percentage Targeted Expected Improvement (TEI) acquisition function.

    Args:
        X_candidates (np.ndarray): Candidate points for evaluation.
        ml_model (_type_): Trained model for predictions.
        y_best (float): Best observed value.
        percentage (float): Percentage for target value adjustment.

    Returns:
        Tuple[np.ndarray]: Mean predictions and TEI scores.
    """
    y_target = y_best * (1 - percentage / 100)
    return target_expected_improvement(X_candidates, ml_model, y_target, dist=abs(y_best - y_target))


def maximum_predicted_value(X_candidates: np.ndarray, ml_model) -> Tuple[np.ndarray]:
    """Acquisition function: Maximum Predicted Value (MPV).
    Returns a landscape that is zero everywhere except at the maximum predicted value.

    Args:
        X_candidates (np.ndarray): Candidate points for evaluation.
        ml_model (_type_): Trained model for predictions.

    Returns:
        Tuple[np.ndarray]: Mean predictions and MPV scores.
    """
    _, mu, _ = ml_model.predict(X_candidates)

    # Return a landscape that is zero everywhere except at the maximum predicted value
    mpv = np.zeros_like(mu)
    mpv[mu.argmax()] = 1.0
    return mu, mpv