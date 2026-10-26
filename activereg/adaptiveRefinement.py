#!

import numpy as np
import pandas as pd
from activereg.data import DatasetGenerator
from activereg.utils import compute_knn_distance
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import Literal

# --------------------------------------------------------------
# Adaptive refinement functions

# ---
# Select a subset of the AL batch to serve as hypercube centers for refinement.

def select_centers_from_batch(
    candidate_points: np.ndarray,
    n_centers: int = 4,
    min_centers: int = 2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Reduce AL batch selection to a smaller diverse set of hypercube centers.
    Assumes sampled_new_idx already has acquisition-informed diversity.

    Args:
        sampled_new_idx (np.ndarray): Indices of the newly sampled points in the pool.
        pool_scaled (np.ndarray): Scaled pool of points from which the samples are drawn.
        n_centers (int, optional): Number of centers to select. Defaults to 4.
        min_centers (int, optional): Minimum number of centers to select. Defaults to 2.
        rng (np.random.Generator, optional): Random generator for the first center. If None,
            numpy's global random state is used (legacy behaviour). Defaults to None.

    Returns:
        np.ndarray: Indices of the selected hypercube centers.
    """

    n_centers = max(min_centers, min(n_centers, len(candidate_points)))

    # No reduction needed if batch is already at or below target
    if len(candidate_points) <= n_centers:
        return candidate_points

    # Otherwise greedy maximin reduction, starting from a random point
    if rng is None:
        selected = np.random.choice(len(candidate_points), size=1, replace=False).tolist()
    else:
        selected = [int(rng.integers(len(candidate_points)))]
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
    refine_method: str = 'lhs',
    return_cube: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray, float]:
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
        return_cube (bool, optional): If True, also return the hypercube bounds and half-side length. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing the generated points and their evaluations.
            If `return_cube` is True, a tuple `(df, bounds, half_side)` with the clipped hypercube
            bounds in the original space, shape (D, 2), and the half-side length in the scaled space.
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

    # Fresh seed for each hypercube, so successive refinements do not repeat the same pattern
    call_seed = refine_generator.next_seed()

    # Generate the refinement dataframe containing a train and validation set
    if refine_function is not None:
        refined_df = refine_generator.generate_dataset(
            function=refine_function,
            n_samples=n_points,
            method=refine_method,
            val_size=None,
            noise_std=refine_noise_std,
            seed=call_seed,
        )

    if refine_function is None:
        dimension_names = refine_generator.get_dimension_names
        refine_array = refine_generator.sample_space(
            n_samples=n_points,
            method=refine_method,
            seed=call_seed,
        )
        refined_df = pd.DataFrame(refine_array, columns=[dimension_names[i] for i in range(design_dimensions)])

    if return_cube:
        return refined_df, hyper_cube_bounds, half_side_length
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

# --------------------------------------------------------------
# Adaptive refinement orchestration (paper: alg:adaptive_refinement)

@dataclass
class RefinementResult:
    """Points added to the candidate pool by one refinement step.

    Attributes:
        X_new (np.ndarray): New points in the scaled space, after filtering, shape (n, D).
        y_new (np.ndarray | None): Oracle values at `X_new`, shape (n, P); None without an oracle.
        step (int): Refinement step counter within the current repetition (1-based).
        n_proposed (int): Number of points proposed before filtering.
        cubes (pd.DataFrame): One row per hypercube: `refinement_step`, `cube`, `half_side`
            (scaled space), `n_proposed`, `n_added`, and per search variable `<var>_centroid`,
            `<var>_low`, `<var>_high` (original space, bounds clipped to the design space).
    """
    X_new: np.ndarray
    y_new: np.ndarray | None
    step: int
    n_proposed: int
    cubes: pd.DataFrame


class AdaptiveRefinement:
    """Adaptive hypercube refinement of the candidate pool.

    Every `step` acquired points, a few centroids are selected among the training
    points, a hypercube is built around each of them and filled with new points, which
    are filtered against the existing pool and returned to the caller. The caller owns
    the pool arrays and appends the returned points to them.

    Trigger: refinement fires when at least `step` points were acquired since the last
    refinement (initial points included, i.e. the count starts from 0).
    Centroid window: centroids are drawn from the last `centroid_window` training points,
    or from the whole training history when `centroid_window` is None.
    """

    def __init__(
        self,
        generator: DatasetGenerator,
        design_bounds: np.ndarray,
        scaler,
        search_var: list[str],
        target_var: list[str],
        step: int,
        n_centroids: int,
        points_per_centroid: int,
        centroid_window: int | None = None,
        half_side_strategy: Literal['fixed', 'density'] = 'density',
        half_side_params: dict = None,
        method: str = 'lhs',
        noise_std: float = 0.0,
        min_distance: float = 1e-4,
        oracle=None,
    ) -> None:
        """Initialize the adaptive refinement.

        Args:
            generator (DatasetGenerator): Generator used to sample (and label) the hypercubes.
            design_bounds (np.ndarray): Design space bounds in the original space, shape (D, 2).
            scaler: Fitted scaler mapping the original space to the scaled space.
            search_var (list[str]): Names of the search variables.
            target_var (list[str]): Names of the target variables.
            step (int): Number of acquired points between refinements.
            n_centroids (int): Number of hypercube centroids per refinement.
            points_per_centroid (int): Number of points sampled in each hypercube.
            centroid_window (int | None, optional): Number of latest training points from
                which the centroids are selected. None uses the whole training history.
                Defaults to None.
            half_side_strategy (str, optional): 'fixed' or 'density'. Defaults to 'density'.
            half_side_params (dict, optional): Parameters of the half-side strategy. Defaults to None.
            method (str, optional): Sampling method within the hypercube. Defaults to 'lhs'.
            noise_std (float, optional): Noise std added to the oracle values. Defaults to 0.0.
            min_distance (float, optional): Minimum distance (scaled space) of new points
                from the pool and from each other. Defaults to 1e-4.
            oracle (optional): Function that labels the new points (benchmark mode). If None,
                only the points are generated and `y_new` is None (no labels available).
        """
        self.generator = generator
        self.design_bounds = design_bounds
        self.scaler = scaler
        self.search_var = search_var
        self.target_var = target_var
        self.step = step
        self.n_centroids = n_centroids
        self.points_per_centroid = points_per_centroid
        if centroid_window is not None and centroid_window < 1:
            raise ValueError(f"centroid_window must be a positive integer or None, got {centroid_window}.")
        self.centroid_window = centroid_window
        self.half_side_strategy = half_side_strategy
        self.half_side_params = half_side_params or {}
        self.method = method
        self.noise_std = noise_std
        self.min_distance = min_distance
        self.oracle = oracle
        # Own random generator for the centroid selection, seeded from the generator so runs are
        # reproducible, and kept apart from numpy's global state (used e.g. by GPR restarts)
        self._rng = np.random.default_rng(generator.next_seed())
        self.reset()

    @classmethod
    def from_config(
        cls,
        refinement_config: dict | None,
        *,
        generator: DatasetGenerator,
        design_bounds: np.ndarray,
        scaler,
        search_var: list[str],
        target_var: list[str],
        oracle=None,
    ) -> "AdaptiveRefinement | None":
        """Build from the `adaptive_refinement` block of the benchmark config.

        Keys and defaults are the same as the original scripts. Returns None when the
        block is null (refinement disabled).
        """
        if refinement_config is None:
            return None

        refiner = cls(
            generator=generator,
            design_bounds=design_bounds,
            scaler=scaler,
            search_var=search_var,
            target_var=target_var,
            step=refinement_config.get('refinement_step', 20),
            n_centroids=refinement_config.get('refinement_centroids', 5),
            points_per_centroid=refinement_config.get('refinement_batch_size', 500),
            centroid_window=refinement_config.get('refinement_centroid_window', None),
            half_side_strategy=refinement_config.get('half_side_length_strategy', 'density'),
            half_side_params=refinement_config.get('half_side_length_strategy_params', {}),
            method=refinement_config.get('method', 'lhs'),
            noise_std=refinement_config.get('refinement_noise_std', 0.0),
            min_distance=refinement_config.get('refinement_filtering_min_distance', 0.0001),
            oracle=oracle,
        )
        window = 'whole history' if refiner.centroid_window is None else f'last {refiner.centroid_window} points'
        print(f"Adaptive refinement activated every {refiner.step} points acquired, "
              f"with {refiner.n_centroids} centroids and adding {refiner.points_per_centroid} points per centroids "
              f"(centroid window: {window}).")
        return refiner

    def reset(self) -> None:
        """Reset the per-repetition state: step counter, last trigger point and logs."""
        self.counter = 0
        self.last_refined_at = 0
        self._log = pd.DataFrame()
        self._cubes_log = pd.DataFrame()

    @property
    def log(self) -> pd.DataFrame:
        """All points added in the current repetition (original space), with a `refinement_step` column."""
        return self._log

    @property
    def cubes_log(self) -> pd.DataFrame:
        """All hypercubes of the current repetition, one row per cube (see `RefinementResult.cubes`)."""
        return self._cubes_log

    def step_if_due(
        self,
        n_acquired: int,
        X_train: np.ndarray,
        X_pool: np.ndarray,
    ) -> RefinementResult | None:
        """Run one refinement step if the trigger is reached.

        Args:
            n_acquired (int): Number of training points acquired so far.
            X_train (np.ndarray): Current training set (scaled space).
            X_pool (np.ndarray): Current pool (scaled space), used for density and filtering.

        Returns:
            RefinementResult | None: The filtered new points, or None if no refinement is due.
        """
        if n_acquired - self.last_refined_at < self.step:
            return None

        self.counter += 1
        dims = self.design_bounds.shape[0]

        # Select the hypercube centres among the training points in the window
        X_window = X_train if self.centroid_window is None else X_train[-self.centroid_window:]
        X_centroids = select_centers_from_batch(
            candidate_points=X_window,
            n_centers=self.n_centroids,
            min_centers=2,
            rng=self._rng,
        )

        # Generate new points around each centroid (original space)
        refined_df = pd.DataFrame()
        cube_rows, cube_sizes = [], []
        for i_cube, centroid in enumerate(X_centroids):
            centroid_df, cube_bounds, half_side = pointwise_hypercube_refinement(
                refine_generator=self.generator,
                design_bounds=self.design_bounds,
                design_dimensions=dims,
                refine_centroid=np.reshape(centroid, (1, -1)),
                pool_scaled=X_pool,
                half_side_length_strategy=self.half_side_strategy,
                hsl_strategy_params=self.half_side_params,
                n_points=self.points_per_centroid,
                refine_noise_std=self.noise_std,
                scaler=self.scaler,
                refine_function=self.oracle,
                refine_method=self.method,
                return_cube=True,
            )
            refined_df = pd.concat([refined_df, centroid_df], ignore_index=True)

            # Describe the hypercube (original space, bounds clipped to the design space)
            centroid_orig = self.scaler.inverse_transform(np.reshape(centroid, (1, -1)))[0]
            row = {'refinement_step': self.counter, 'cube': i_cube, 'half_side': half_side,
                   'n_proposed': len(centroid_df)}
            for d, var in enumerate(self.search_var):
                row[f'{var}_centroid'] = centroid_orig[d]
                row[f'{var}_low'], row[f'{var}_high'] = cube_bounds[d]
            cube_rows.append(row)
            cube_sizes.append(len(centroid_df))

        # Filter out points too close to the pool or to each other (scaled space)
        X_refined_scaled = self.scaler.transform(refined_df[self.search_var].to_numpy())
        X_new, kept_indexes = filter_refined_additions(
            X_addition=X_refined_scaled,
            X_existing=X_pool,
            min_distance=self.min_distance,
            filter_internal=True
        )
        y_new = refined_df[self.target_var].to_numpy()[kept_indexes] if self.oracle is not None else None

        n_proposed = self.n_centroids * self.points_per_centroid
        print(f"-> Adaptive refinement step {self.counter} at {n_acquired} points acquired, "
              f"added {len(X_new)}/{n_proposed} points.")

        # Keep a log of the added points (original space)
        refined_df = refined_df.iloc[kept_indexes].copy()
        refined_df['refinement_step'] = self.counter
        self._log = pd.concat([self._log, refined_df], ignore_index=True)

        # Log the hypercubes, with the number of points each one kept after filtering
        cubes_df = pd.DataFrame(cube_rows)
        cube_of_point = np.repeat(np.arange(len(cube_sizes)), cube_sizes)
        cubes_df['n_added'] = np.bincount(cube_of_point[kept_indexes], minlength=len(cube_sizes))
        self._cubes_log = pd.concat([self._cubes_log, cubes_df], ignore_index=True)

        self.last_refined_at = n_acquired
        return RefinementResult(X_new=X_new, y_new=y_new, step=self.counter, n_proposed=n_proposed,
                                cubes=cubes_df)
