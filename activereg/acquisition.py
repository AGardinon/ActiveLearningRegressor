#!

import torch
import numpy as np
from scipy.stats import norm
from typing import Tuple

def highest_landscape_selection(landscape: np.ndarray, percentile: int=80):
    """
    Select the top percentile of the landscape (distribution) and return
    their indexes
    """
    threshold = np.percentile(landscape, percentile)
    return np.where(landscape >= threshold)[0]


class AcquisitionFunction:
    """
    Acquisition function class.
    """
    def __init__(self, acquisition_mode: str, y_best: float, **kwargs):
        self.acquisition_mode = acquisition_mode
        self.modes = ['upper_confidence_bound', 
                      'uncertainty_landscape', 
                      'expected_improvement',
                      'target_expected_improvement']
        assert acquisition_mode in self.modes, f'Function "{acquisition_mode}" not implemented, choose from {self.modes.keys()}'

        # additional parameters
        self.y_best = y_best
        # !!!
        self.y_target = y_best
        self.kappa = kwargs.get('kappa', 2.0)
        self.xi = kwargs.get('xi', 1.e-2)
        self.tolerance = kwargs.get('tolerance', 0.05)

    def landscape_acquisition(self, X_candidates: np.ndarray, ml_model):
        if self.acquisition_mode == 'upper_confidence_bound':
            return upper_confidence_bound(X_candidates=X_candidates, ml_model=ml_model, kappa=self.kappa)
        
        elif self.acquisition_mode == 'uncertainty_landscape':
            return uncertainty_landscape(X_candidates=X_candidates, ml_model=ml_model)

        elif self.acquisition_mode == 'expected_improvement':
            return expected_improvement(X_candidates=X_candidates, ml_model=ml_model, y_best=self.y_best, xi=self.xi)
        
        elif self.acquisition_mode == 'target_expected_improvement':
            return target_expected_improvement(X_candidates=X_candidates, ml_model=ml_model, 
                                               y_target=self.y_target, tolerance=self.tolerance, xi=self.xi)

# '''
# The methods assume one of the activereg.mlmodel is used, where the predict statment
# automatically returns mean and standard dev.
# '''

def upper_confidence_bound(X_candidates: np.ndarray, ml_model, kappa: float=2.0) -> Tuple[np.ndarray]:
    """
    Acquisition function: Upper Confidence Bound (UCB) function.

    Parameters:
    - X_candidates (np.ndarray): The candidate points to be evaluated.
    - ml_model: A trained model from the models class (must support .predict with return_std=True).
    - kappa (float): Exploration-exploitation tradeoff parameter (higher kappa favors exploration).

    Returns:
    - np.ndarray: predicted values.
    - np.ndarray: UCB values for each candidate point.
    """
    _, mu, sigma = ml_model.predict(X_candidates)
    return mu, mu + kappa * sigma


def uncertainty_landscape(X_candidates: np.ndarray, ml_model) -> Tuple[np.ndarray]:
    """
    Explore using the model uncertainty landscape

    Parameters:
    - X_candidates (np.ndarray): The candidate points to be evaluated.
    - gp_model: A trained model from the models class (must support .predict with return_std=True).

    Returns:
    - np.ndarray: predicted values.
    - np.ndarray: model uncertainty per candidate points.
    """
    _, mu, sigma = ml_model.predict(X_candidates)
    return mu, sigma


def expected_improvement(X_candidates: np.ndarray, ml_model, y_best: float, xi: float = 0.01) -> Tuple[np.ndarray]:
    """
    Acquisition function: Expected Improvement (EI).

    Parameters:
    - X_candidates (np.ndarray): The candidate points where EI will be evaluated.
    - ml_model: A trained model from the models class (must support .predict with return_std=True).
    - y_best (float): The best observed function value so far (e.g., max(y_train)).
    - xi (float): Exploration-exploitation tradeoff parameter (higher xi favors exploration).

    Returns:
    - np.ndarray: predicted values.
    - np.ndarray: EI values for each candidate point.
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

def target_expected_improvement(X_candidates: np.ndarray, ml_model, y_target: float, 
                              tolerance: float = 0.1, xi: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Acquisition function: Expected Improvement toward a target value.
    
    Parameters:
    - X_candidates (np.ndarray): The candidate points where EI will be evaluated.
    - ml_model: A trained model from the models class (must support .predict with return_std=True).
    - y_target (float): The target value you want to achieve.
    - tolerance (float): Acceptable deviation from target (defines "success region").
    - xi (float): Exploration-exploitation tradeoff parameter (higher xi = more exploration).
    
    Returns:
    - np.ndarray: predicted values.
    - np.ndarray: Target-oriented EI values for each candidate point.
    """
    # Get mean and standard deviation from the model
    _, mu, sigma = ml_model.predict(X_candidates)
    
    # Avoid division by zero
    sigma = sigma.clip(min=1e-9)
    
    # Method 1: Add xi as exploration bonus to uncertainty
    # Higher xi increases the value of high-uncertainty regions
    exploration_bonus = xi * sigma
    
    # Define the target region bounds
    target_lower = y_target - tolerance
    target_upper = y_target + tolerance
    
    # Calculate probability of being in target region
    prob_in_target = (norm.cdf((target_upper - mu) / sigma) - 
                     norm.cdf((target_lower - mu) / sigma))
    
    # Expected distance from target (penalize being far from target)
    expected_distance = np.abs(mu - y_target)
    
    # Combined acquisition with exploration bonus
    target_ei = (prob_in_target / (1 + expected_distance)) + exploration_bonus
    
    return mu, target_ei