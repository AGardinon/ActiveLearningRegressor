#!

import numpy as np
import pandas as pd
from activereg.data import DatasetGenerator
from activereg.utils import compute_knn_distance
from scipy.spatial import cKDTree
from typing import Literal

# --------------------------------------------------------------
# Adaptive refinement functions

# TODO: ?
# class AdaptiveRefinement:

# ---
# Select a subset of the AL batch to serve as hypercube centers for refinement.

def select_centers_from_batch(
    candidate_points: np.ndarray,
    n_centers: int = 4,
    min_centers: int = 2,
) -> np.ndarray:
    """Reduce AL batch selection to a smaller diverse set of hypercube centers.
    Assumes sampled_new_idx already has acquisition-informed diversity.

    Args:
        sampled_new_idx (np.ndarray): Indices of the newly sampled points in the pool.
        pool_scaled (np.ndarray): Scaled pool of points from which the samples are drawn.
        n_centers (int, optional): Number of centers to select. Defaults to 4.
        min_centers (int, optional): Minimum number of centers to select. Defaults to 2.

    Returns:
        np.ndarray: Indices of the selected hypercube centers.
    """
    
    n_centers = max(min_centers, min(n_centers, len(candidate_points)))
    
    # No reduction needed if batch is already at or below target
    if len(candidate_points) <= n_centers:
        return candidate_points
    
    # Otherwise greedy maximin reduction
    selected = [0]
    for _ in range(n_centers - 1):
        selected_points = candidate_points[selected]
        dists = np.min(
            np.linalg.norm(candidate_points[:, None] - selected_points[None], axis=2),
            axis=1
        )
        dists[selected] = -np.inf
        selected.append(np.argmax(dists))
    
    return candidate_points[selected]

# ---
# Compute half-side length for the hypercube
# TODO: consider adding a `fixed-decay` option for decreasing the half-side length over iterations

def get_hypercube_half_side(
    centroid: np.ndarray,           # shape (1, n_dims)
    pool_scaled: np.ndarray,        # shape (N, n_dims)
    strategy: Literal['fixed', 'density'] = 'density',
    # fixed strategy
    base_fraction: float = 0.1,     # fraction of space extent to start with for the fixed strategy
    # density strategy
    k_neighbors: int = 10,
    density_scale: float = 0.5,     # multiplier on mean kNN distance to determine half-side length
    # shared
    min_half_side: float = 1e-3,
    max_half_side: float = 0.5,     # in scaled space, 0.5 = half the domain
) -> float:
    """Compute the half-side length of a hypercube centered at a given centroid.

    Args:
        centroid (np.ndarray): Centroid of the hypercube (array of shape (1, n_dimensions) in the reduced space).
        pool_scaled (np.ndarray): Scaled pool of points from which the samples are drawn (shape (N, n_dimensions)).
        strategy (str, optional): Strategy to determine the half-side length of the hypercube. Defaults to 'density'.
         - 'fixed': use a fixed fraction of the space extent as the half-side length.
         - 'density': adapt the half-side length based on the local density of points in the pool, using the mean distance to the k nearest neighbors.
        base_fraction (float, optional): Fraction of space extent to start with for the fixed strategy. Defaults to 0.1.
        density_scale (float, optional): Multiplier on mean kNN distance to determine half-side length. Defaults to 0.5.
        max_half_side (float, optional): Maximum allowed half-side length. Defaults to 0.5.

    Returns:
        float: Computed half-side length of the hypercube.
    """
    assert strategy in ('fixed', 'density'), f"Unknown strategy: {strategy}"
    centroid_flat = centroid.flatten()

    if strategy == 'fixed':
        half_side = base_fraction

    elif strategy == 'density':
        # Mean distance to k nearest neighbors in the pool
        dists = np.linalg.norm(pool_scaled - centroid_flat, axis=1)
        knn_dists = np.sort(dists)[1:k_neighbors+1]  # exclude self if centroid is in pool
        mean_knn_dist = np.mean(knn_dists)
        global_mean_knn_dist = compute_knn_distance(pool_scaled, k_neighbors=k_neighbors)
        half_side = density_scale * min(mean_knn_dist, 2.0 * global_mean_knn_dist)

    # Clip to reasonable range
    half_side = float(np.clip(half_side, min_half_side, max_half_side))
    return half_side

# ---
# Create hypercube bounds and generate points within it for refinement

def pointwise_hypercube_refinement(
    refine_generator: DatasetGenerator,
    design_bounds: np.ndarray,
    design_dimensions: int,
    refine_centroid: np.ndarray,
    pool_scaled: np.ndarray,
    n_points: int,
    refine_noise_std: float,
    scaler,
    refine_function,
    half_side_length_strategy: Literal['fixed', 'density'] = 'density',
    hsl_strategy_params: dict = None,
    refine_method: str = 'lhs'
) -> pd.DataFrame:
    """Generate new candidate points within a hypercube centered at a given centroid.

    Args:
        refine_generator (DatasetGenerator): DatasetGenerator instance for generating points.
        refine_centroid (np.ndarray): Centroid of the hypercube (array of shape (1, n_dimensions) in the reduced space).
        half_side_length_strategy (str, optional): Strategy to determine the half-side length of the hypercube. Defaults to 'density'.
        n_points (int): Number of points to generate within the hypercube.
        refine_noise_std (float): Standard deviation of noise to add to the function evaluations.
        scaler (_type_): Scaler used to inverse transform the hypercube bounds.
        refine_function (_type_): Function to evaluate at the generated points.
        refine_method (str, optional): Sampling method to generate points within the hypercube. Defaults to 'lhs'.

    Returns:
        pd.DataFrame: DataFrame containing the generated points and their evaluations.
    """    
    assert refine_centroid.shape == (1, design_dimensions), "refine_centroid must be of shape (1, n_dimensions)"

    # Inverse transform the centroid to the original space
    refine_centroid_orig = scaler.inverse_transform(refine_centroid)
    
    # Compute the half-side length of the hypercube based on the selected strategy
    half_side_length = get_hypercube_half_side(
        centroid=refine_centroid,
        pool_scaled=pool_scaled,
        strategy=half_side_length_strategy,
        **(hsl_strategy_params or {})
    )

    # Scale the half_side_length to each dimension based on the scaler
    half_side_length_dimwise = half_side_length * scaler.scale_

    # Define the hypercube bounds in the original space, centered at the refine_centroid
    hyper_cube_bounds = np.zeros((design_dimensions, 2))
    for d in range(design_dimensions):
        low = refine_centroid_orig[0, d] - half_side_length_dimwise[d]
        high = refine_centroid_orig[0, d] + half_side_length_dimwise[d]

        # clip within design space
        low = max(low, design_bounds[d, 0])
        high = min(high, design_bounds[d, 1])

        hyper_cube_bounds[d] = [low, high]

    # Generate n_points random samples within the hypercube in the original space
    refine_generator.set_bounds = hyper_cube_bounds

    # Generate the refinement dataframe containing a train and validation set
    refined_df = refine_generator.generate_dataset(
        function=refine_function,
        n_samples=n_points,
        method=refine_method,
        val_size=1.0,
        noise_std=refine_noise_std,
    )

    return refined_df

# ---
# Filter out points that are too close to existing points in the pool and optionally filter internal duplicates
# TODO: possibility to used adaptive min_distance based on points-to-keep thresholding

def filter_refined_additions(
    X_addition: np.ndarray,
    X_existing: np.ndarray,
    min_distance: float,
    filter_internal: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Filter out points from X_addition that are too close to the X_existing points and optionally filter internal duplicates.

    Args:
        X_addition (np.ndarray): Points proposed for addition.
        X_existing (np.ndarray): Existing points to compare against.
        min_distance (float): Minimum allowable distance from existing points.
        filter_internal (bool, optional): If True, also filter out points that are too close to each other within X_addition. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: Filtered points and their indices.
    """

    # Filter against existing pool
    tree = cKDTree(X_existing)
    dists, _ = tree.query(X_addition, k=1)
    mask = dists >= min_distance
    candidates = X_addition[mask]
    candidate_indices = np.where(mask)[0]

    if not filter_internal or len(candidates) == 0:
        return candidates, candidate_indices

    # Greedy filter for intra-addition duplicates
    kept = [0]
    for i in range(1, len(candidates)):
        tree_kept = cKDTree(candidates[kept])
        dist, _ = tree_kept.query(candidates[i], k=1)
        if dist >= min_distance:
            kept.append(i)

    kept = np.array(kept)
    return candidates[kept], candidate_indices[kept]
