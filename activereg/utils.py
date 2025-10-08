#!

import ot
import json
import shutil
import numpy as np
from scipy.stats import norm, entropy, gaussian_kde
from sklearn.metrics.pairwise import rbf_kernel
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from datetime import datetime
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

EPS = 1e-12

def js_divergence_from_probs(p, q, base=2.0):
    """Jensen-Shannon divergence between two probability vectors."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # ensure same shape
    assert p.shape == q.shape
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=base) + entropy(q, m, base=base))


def jsd_histogram(samples_p, samples_q, nbins=100, range=None, base=2.0):
    """
    Compute JSD between two sample sets by histogramming them with the SAME bins.
    - range: (min, max) or None -> automatic from combined samples.
    """
    samples_p = np.asarray(samples_p).ravel()
    samples_q = np.asarray(samples_q).ravel()
    if range is None:
        mn = min(samples_p.min(), samples_q.min())
        mx = max(samples_p.max(), samples_q.max())
        # optionally pad a bit
        pad = 1e-6 * (mx - mn + 1.0)
        range = (mn - pad, mx + pad)
    p_hist, edges = np.histogram(samples_p, bins=nbins, range=range, density=False)
    q_hist, _     = np.histogram(samples_q, bins=nbins, range=range, density=False)
    # convert counts to probabilities
    p_prob = p_hist.astype(float) + EPS
    q_prob = q_hist.astype(float) + EPS
    p_prob /= p_prob.sum()
    q_prob /= q_prob.sum()
    return js_divergence_from_probs(p_prob, q_prob, base=base)


def jsd_kde(samples_p, samples_q, grid_points=512, bandwidth=None, base=2.0):
    """
    KDE-based JSD: fit KDE to each sample set, evaluate on shared grid, compute JSD.
    Less sensitive to bin edge issues.
    """
    samples_p = np.asarray(samples_p).ravel()
    samples_q = np.asarray(samples_q).ravel()
    mn = min(samples_p.min(), samples_q.min())
    mx = max(samples_p.max(), samples_q.max())
    grid = np.linspace(mn, mx, grid_points)
    kde_p = gaussian_kde(samples_p, bw_method=bandwidth)
    kde_q = gaussian_kde(samples_q, bw_method=bandwidth)
    p_vals = kde_p(grid) + EPS
    q_vals = kde_q(grid) + EPS
    # make probabilities (sum to 1) using Riemann sum approximation
    p_prob = p_vals / np.trapz(p_vals, grid)
    q_prob = q_vals / np.trapz(q_vals, grid)
    return js_divergence_from_probs(p_prob, q_prob, base=base)


def prepare_measure(
    X: np.ndarray, 
    Y: np.ndarray, 
    normalize: bool=True, 
    shift_nonneg: bool=False, 
    clip_min_to_zero: bool=True
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare the input features and target masses for the metric computation.

    Args:
        X (np.ndarray): Input features of shape (N, d).
        Y (np.ndarray): Input target values of shape (N,).
        normalize (bool, optional): Whether to normalize the masses. Defaults to True.
        shift_nonneg (bool, optional): Whether to shift the masses to be non-negative. Defaults to False.
        clip_min_to_zero (bool, optional): Whether to clip the masses to be non-negative. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Prepared input features and target masses.
    """
    assert not (shift_nonneg and clip_min_to_zero), "Only one of shift_nonneg and clip_min_to_zero can be True"

    masses = Y.astype(float)
    if clip_min_to_zero:
        masses = np.clip(masses, 0, None)
    if shift_nonneg and masses.min() < 0 and not clip_min_to_zero:
        masses = masses - masses.min()
    masses[masses < 0] = 0.0
    if normalize:
        s = masses.sum()
        if s > 0:
            masses /= s
    return X, masses


def mmd_from_coords(
    X1: np.ndarray, 
    Y1: np.ndarray, 
    X2: np.ndarray, 
    Y2: np.ndarray, 
    sigma: float=1.0, 
    normalize: bool=True, 
    shift_nonneg: bool=False, 
    clip_min_to_zero: bool=True
) -> float:
    """Compute the Maximum Mean Discrepancy (MMD) between two distributions.

    Args:
        X1 (np.ndarray): Input features of shape (N, d).
        Y1 (np.ndarray): Input target values of shape (N,).
        X2 (np.ndarray): Input features of shape (M, d).
        Y2 (np.ndarray): Input target values of shape (M,).
        sigma (float, optional): Bandwidth parameter for the RBF kernel. Defaults to 1.0.
        normalize (bool, optional): Whether to normalize the distributions. Defaults to True.
        shift_nonneg (bool, optional): Whether to shift the masses to be non-negative. Defaults to False.
        clip_min_to_zero (bool, optional): Whether to clip the masses to be non-negative

    Returns:
        float: The computed MMD value.
    """
    coords1, a = prepare_measure(X1, Y1, normalize=normalize, shift_nonneg=shift_nonneg, clip_min_to_zero=clip_min_to_zero)
    coords2, b = prepare_measure(X2, Y2, normalize=normalize, shift_nonneg=shift_nonneg, clip_min_to_zero=clip_min_to_zero)
    K11 = rbf_kernel(coords1, coords1, gamma=1.0/(2*sigma**2))
    K22 = rbf_kernel(coords2, coords2, gamma=1.0/(2*sigma**2))
    K12 = rbf_kernel(coords1, coords2, gamma=1.0/(2*sigma**2))
    return a @ K11 @ a + b @ K22 @ b - 2 * a @ K12 @ b


def emd_from_coords(
    X1: np.ndarray,
    Y1: np.ndarray,
    X2: np.ndarray,
    Y2: np.ndarray,
    normalize=True,
    shift_nonneg: bool=False,
    clip_min_to_zero: bool=True
) -> float:
    """Compute the Earth Mover's Distance (EMD) between two distributions.

    Args:
        X1 (np.ndarray): Input features of shape (N, d).
        Y1 (np.ndarray): Input target values of shape (N,).
        X2 (np.ndarray): Input features of shape (M, d).
        Y2 (np.ndarray): Input target values of shape (M,).
        normalize (bool, optional): Whether to normalize the distributions. Defaults to True.
        shift_nonneg (bool, optional): Whether to shift the masses to be non-negative. Defaults to False.
        clip_min_to_zero (bool, optional): Whether to clip the masses to be non-negative. Defaults to True.

    Returns:
        float: The computed EMD value.
    """
    coords1, a = prepare_measure(X1, Y1, normalize=normalize, shift_nonneg=shift_nonneg, clip_min_to_zero=clip_min_to_zero)
    coords2, b = prepare_measure(X2, Y2, normalize=normalize, shift_nonneg=shift_nonneg, clip_min_to_zero=clip_min_to_zero)
    # pairwise Euclidean distance matrix
    M = ot.dist(coords1, coords2, metric='euclidean')
    cost = ot.emd2(a, b, M, numItermax=1e6)  # total transport cost
    return cost


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

def create_strict_folder(path_str: str, overwrite: bool = False) -> None:
    """
    Create a folder from a path string, with optional overwrite.
    
    Args:
        path_str: str - Path to the folder to create
        overwrite: bool - If True, allows overwriting existing folder (default: False)
    """
    path = Path(path_str)
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
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

