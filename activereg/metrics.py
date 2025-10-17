#!

import ot
import numpy as np
from scipy.stats import norm, entropy, gaussian_kde
from sklearn.metrics.pairwise import rbf_kernel

# TODO: add more metrics to the evaluation function

# CONSTANTS
EPS = 1e-12  # small constant for numerical stability
CONFIDENCE = 0.95  # default confidence level

# METRICS

# def evaluate_regression_metrics(y_true, y_mean, y_std, confidence=0.95) -> Dict:
#     """
#     Evaluate regression predictions with uncertainty estimates.

#     Args:
#         y_true (np.ndarray): True target values. Shape (N,)
#         y_mean (np.ndarray): Predicted mean values. Shape (N,)
#         y_std (np.ndarray): Predicted standard deviations. Shape (N,)
#         confidence (float): Confidence level for interval (default: 0.95)

#     Returns:
#         dict: Dictionary with RMSE, MAE, NLL, PICP, and MPIW
#     """
#     y_true = np.asarray(y_true)
#     y_mean = np.asarray(y_mean)
#     y_std = np.asarray(y_std)

#     # Basic error metrics
#     rmse = np.sqrt(np.mean((y_true - y_mean)**2))
#     mae = np.mean(np.abs(y_true - y_mean))

#     # Negative Log-Likelihood (Gaussian assumption)
#     nll = -np.mean(norm.logpdf(y_true, loc=y_mean, scale=y_std + 1e-6))  # add epsilon for numerical stability

#     # Prediction Interval Coverage Probability (PICP)
#     lower, upper = norm.interval(confidence, loc=y_mean, scale=y_std + 1e-6)
#     picp = np.mean((y_true >= lower) & (y_true <= upper))

#     # Mean Prediction Interval Width (MPIW)
#     z = norm.ppf(0.5 + confidence / 2)
#     mpiw = 2 * z * np.mean(y_std)

#     return {
#         "RMSE": rmse,
#         "MAE": mae,
#         "NLL": nll,
#         f"PICP@{int(confidence*100)}%": picp,
#         f"MPIW@{int(confidence*100)}%": mpiw
#     }

# --------------------------------------------------------------------------------
# Uncertainty & regression metrics


def nll_gauss(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, eps=EPS) -> float:
    """Negative log-likelihood under Gaussian assumption.

    Args:
        y_true (np.ndarray): True values.
        y_mean (np.ndarray): Predicted mean values.
        y_std (np.ndarray): Predicted standard deviations.
        eps (float, optional): Small constant for numerical stability. Defaults to EPS=1e-12.

    Returns:
        float: Negative log-likelihood.
    """
    y_std = np.clip(y_std, eps, None)
    return 0.5 * np.mean(np.log(2 * np.pi * y_std**2) + ((y_true - y_mean)**2) / (y_std**2))
           

def picp(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, confidence=CONFIDENCE) -> float:
    """Prediction Interval Coverage Probability (PICP).

    Args:
        y_true (np.ndarray): True values.
        y_mean (np.ndarray): Predicted mean values.
        y_std (np.ndarray): Predicted standard deviations.
        confidence (float, optional): Confidence level. Defaults to CONFIDENCE=0.95.

    Returns:
        float: Prediction Interval Coverage Probability (PICP).
    """
    z = norm.ppf(0.5 + confidence / 2.0)
    lower = y_mean - z * y_std
    upper = y_mean + z * y_std
    inside = (y_true >= lower) & (y_true <= upper)
    return np.mean(inside)


def mpiw(y_mean: np.ndarray, y_std: np.ndarray, confidence=CONFIDENCE) -> float:
    """Mean Prediction Interval Width (MPIW).

    Args:
        y_mean (np.ndarray): Predicted mean values.
        y_std (np.ndarray): Predicted standard deviations.
        confidence (float, optional): Confidence level. Defaults to CONFIDENCE=0.95.

    Returns:
        float: Mean Prediction Interval Width (MPIW).
    """
    z = norm.ppf(0.5 + confidence / 2.0)
    interval_widths = 2 * z * y_std
    return np.mean(interval_widths)

# --------------------------------------------------------------------------------
# JSD metrics


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

# --------------------------------------------------------------------------------
# MMD and EMD


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


def prepare_measure(
    X: np.ndarray, 
    Y: np.ndarray, 
    normalize: bool=True, 
    shift_nonneg: bool=False, 
    clip_min_to_zero: bool=True
) -> tuple[np.ndarray, np.ndarray]:
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
