#!

import json
import numpy as np
from datetime import datetime
from scipy.stats import norm
from pathlib import Path
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, List, Dict, Any


# METRICS


def evaluate_regression_metrics(y_true, y_mean, y_std, confidence=0.95) -> Dict:
    """
    Evaluate regression predictions with uncertainty estimates.

    Args:
        y_true (np.ndarray): True target values. Shape (N,)
        y_mean (np.ndarray): Predicted mean values. Shape (N,)
        y_std (np.ndarray): Predicted standard deviations. Shape (N,)
        confidence (float): Confidence level for interval (default: 0.95)

    Returns:
        dict: Dictionary with RMSE, MAE, NLL, PICP, and MPIW
    """
    y_true = np.asarray(y_true)
    y_mean = np.asarray(y_mean)
    y_std = np.asarray(y_std)

    # Basic error metrics
    rmse = np.sqrt(np.mean((y_true - y_mean)**2))
    mae = np.mean(np.abs(y_true - y_mean))

    # Negative Log-Likelihood (Gaussian assumption)
    nll = -np.mean(norm.logpdf(y_true, loc=y_mean, scale=y_std + 1e-6))  # add epsilon for numerical stability

    # Prediction Interval Coverage Probability (PICP)
    lower, upper = norm.interval(confidence, loc=y_mean, scale=y_std + 1e-6)
    picp = np.mean((y_true >= lower) & (y_true <= upper))

    # Mean Prediction Interval Width (MPIW)
    z = norm.ppf(0.5 + confidence / 2)
    mpiw = 2 * z * np.mean(y_std)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "NLL": nll,
        f"PICP@{int(confidence*100)}%": picp,
        f"MPIW@{int(confidence*100)}%": mpiw
    }


# EXPERIMENTS


def numpy_to_dataloader(x: np.ndarray, y: np.ndarray = None, **kwargs) -> DataLoader:
    """
    Example:
    data_loader = numpy_to_dataloader(x, y, batch_size=batch_size)
    """
    if y is None:
        return DataLoader(TensorDataset(Tensor(x)),  **kwargs)
    else:
        return DataLoader(TensorDataset(Tensor(x), Tensor(y)),  **kwargs)
    

def create_experiment_name(name_set: Tuple) -> str:
    """
    Concatenates a set of string into a unique string
    """
    exp_name_list = [str(i) for i in name_set]
    exp_name = exp_name_list[0]+"_".join(exp_name_list[1:])
    return exp_name


# SYNTHETIC DATA CREATION


def generate_uniform_grid(n_dim: int, limits: List[Tuple[float]], spacing: List[int]) -> np.ndarray:
    """
    Generates a uniform grid of points in N dimensions.

    Parameters:
    - n_dim: int -> Number of dimensions.
    - limits: list of tuples -> [(min_1, max_1), (min_2, max_2), ..., (min_n, max_n)].
    - spacing: list of floats -> Grid spacing in each dimension.

    Returns:
    - np.ndarray -> Array of shape (num_points, n_dim) with grid points.
    """
    assert len(limits) == n_dim, "limits must have the same length as n_dim"
    assert len(spacing) == n_dim, "spacing must have the same length as n_dim"

    # Generate 1D arrays for each dimension
    grid_axes = [np.arange(lim[0], lim[1] + spacing[i], spacing[i]) for i, lim in enumerate(limits)]

    # Generate N-dimensional meshgrid and flatten
    grid = np.array(np.meshgrid(*grid_axes, indexing="ij")).T.reshape(-1, n_dim)

    return grid


def gaussian_landscape(X: np.ndarray, centers: List[Tuple[float]], scales: List[float], noise_level: float=0.0) -> np.ndarray:
    """
    Creates a 2D synthetic landscape using a sum of Gaussians.

    Parameters:
    - X: (N, 2) array of points.
    - centers: List of Gaussian peak centers [(x1, y1), (x2, y2), ...].
    - scales: List of Gaussian width scales [s1, s2, ...].

    Returns:
    - (N,) array with function values at each X point.
    """
    f_values = np.zeros(X.shape[0])
    for center, scale in zip(centers, scales):
        dist_sq = np.sum((X - center) ** 2, axis=1)
        f_values += np.exp(-dist_sq / (2 * scale**2))
    
    noise = noise_level * np.random.randn(X.shape[0])
    return f_values + noise


def sinusoidal_landscape(X: np.ndarray, noise_level: float=0.1) -> np.ndarray:
    """
    Multi-dimensional sinusoidal function.

    f(X) = sum(sin(X_i)) + noise

    Parameters:
    - X: (N, d) array of N points in d-dimensional space.
    - noise_level: float, intensity of the added random noise.

    Returns:
    - (N,) array with function values.
    """
    noise = noise_level * np.random.randn(X.shape[0])  # Optional noise
    return np.sum(np.sin(X), axis=1) + noise


# FOLDERS & FILES

def create_strict_folder(path_str: str) -> None:
    """
    Create a folder from a path string, no owerwrite.
    """
    path = Path(path_str)
    if path.exists():
        raise FileExistsError(f"Directory '{path}' already exists.")
    path.mkdir(parents=True)


def save_to_json(dictionary: Dict[Any, Any], fout_name: str, timestamp: bool=True, verbose: bool=False) -> None:
    """
    Saves a dictionary to a JSON file with a timestamp appended to the file name.

    Parameters:
    - dictionary (Dict[Any, Any]): The dictionary to save.
    - fout_name (str): The base name of the output file (without extension).

    Returns:
    - None
    """
    if isinstance(fout_name, Path):
        fout_name = str(fout_name)

    if timestamp:
        timestamp_str = datetime.now().strftime('%b_%d_%Y')
        fout_name = f"{fout_name}_{timestamp_str}"

    if not str(fout_name).endswith('.json'):
        fout_name += '.json'
    
    fout_name = Path(fout_name)

    with open(fout_name, 'w') as f:
        json.dump(dictionary, f, indent=4)
    if verbose:
        print(f"JSON saved: {fout_name}")

