#!

import numpy as np
from typing import Tuple

def highest_landscape_selection(landscape: np.ndarray, percentile: int=80):
    """
    Select the top percentile of the landscape (distribution) and return
    their indexes
    """
    threshold = np.percentile(landscape, percentile)
    return np.where(landscape >= threshold)[0]


def landscape_acquisition(X_candidates: np.ndarray, gp_model, acquisition_mode: str, **kwargs):

    modes = {
        'upper_confidence_bound' : upper_confidence_bound,
        'explore_uncertainty' : explore_uncertainty
    }

    if acquisition_mode not in modes:
        raise ValueError(f"Invalid acquisition mode: {acquisition_mode}. Choose from {list(modes.keys())}.")

    return modes[acquisition_mode](X_candidates, gp_model, **kwargs)


def upper_confidence_bound(X_candidates: np.ndarray, gp_model, kappa: float=2.0) -> Tuple[np.ndarray]:
    """
    Acquisition function: Upper Confidence Bound (UCB)
    """
    mu, sigma = gp_model.predict(X_candidates, return_std=True)
    return mu, mu + kappa * sigma


def explore_uncertainty(X_candidates: np.ndarray, gp_model) -> Tuple[np.ndarray]:
    """
    Explore using the model uncertainty landscape
    """
    mu, sigma = gp_model.predict(X_candidates, return_std=True)
    return mu, sigma