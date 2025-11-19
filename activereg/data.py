#!

import torch
import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------------------
# Dataset generation functions

# TODO: remove y_scaling as it might lead to errors in the overall process and in the refinement steps

class DatasetGenerator:

    def __init__(
        self, 
        n_dimensions: int, 
        bounds: np.ndarray, 
        negate: bool=False,
        seed: int=None, 
        dim_labels: list[str]=None, 
        target_label: list[str]=None
    ) -> None:
        """Initialize the DatasetGenerator.

        Args:
            n_dimensions (int): Number of dimensions for the input space.
            bounds (np.ndarray): Array of shape (n_dimensions, 2) specifying the (min, max) bounds for each dimension.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            dim_labels (list[str], optional): List of labels for each dimension. Defaults to None.
        """
        self.n_dimensions = n_dimensions
        self.bounds = bounds
        self.negate_y = negate
        self.available_methods = ['lhs', 'sobol', 'random']
        self.dimension_names = dim_labels if dim_labels is not None else [f'x{i+1}' for i in range(n_dimensions)]
        self.target_label = target_label if target_label is not None else ['y']
        self.seed = seed

        assert len(bounds) == self.n_dimensions, "Length of bounds must match n_dimensions"
        assert all(isinstance(b, np.ndarray) and len(b) == 2 for b in bounds), "Each bound must be a tuple of (min, max)"
        assert all(b[0] < b[1] for b in bounds), "Each bound's min must be less than max"
        assert len(self.dimension_names) == self.n_dimensions, "Length of dimension names must match n_dimensions"

    @property
    def get_dimension_names(self):
        return self.dimension_names
    
    @get_dimension_names.setter
    def set_dimension_names(self, dim_labels):
        assert len(dim_labels) == self.n_dimensions, "Length of dimension names must match n_dimensions"
        self.dimension_names = dim_labels

    @property
    def get_bounds(self):
        return self.bounds

    @get_bounds.setter
    def set_bounds(self, bounds):
        assert len(bounds) == self.n_dimensions, "Length of bounds must match n_dimensions"
        assert all(isinstance(b, np.ndarray) and len(b) == 2 for b in bounds), "Each bound must be a tuple of (min, max)"
        assert all(b[0] < b[1] for b in bounds), "Each bound's min must be less than max"
        self.bounds = bounds

    def __repr__(self):
        return (f"DatasetGenerator(n_dimensions={self.n_dimensions}, "
                f"bounds={self.bounds}, "
                f"dimension_names={self.dimension_names}), "
                f"seed={self.seed})")

    def sample_space(self, n_samples: int, method: str='lhs', seed: int=None, **kwargs) -> np.ndarray:
        """Generate samples using the specified sampling method. For 'custom' method, 
        a `custom_sampler` function must be provided in kwargs. The function should
        accept parameters (n_samples, n_dimensions, bounds, seed, **kwargs) and return 
        an array of shape (n_samples, n_dimensions).

        Args:
            n_samples (int): Number of samples to generate.
            method (str, optional): Sampling method to use. Defaults to 'lhs'.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the sampling functions.

        Raises:
            ValueError: If an unknown sampling method is requested.
            ValueError: If the custom sampler is not provided for 'custom' method.

        Returns:
            np.ndarray: Array of shape (n_samples, n_dimensions) containing the generated samples.
        """
        if self.seed:
            seed = self.seed

        if method == 'lhs':
            return lhs_sampling(n_samples=n_samples, n_dimensions=self.n_dimensions, bounds=self.bounds, seed=seed)
        elif method == 'sobol':
            return sobol_sampling(n_samples=n_samples, n_dimensions=self.n_dimensions, bounds=self.bounds, seed=seed)
        elif method == 'random':
            return random_sampling(n_samples=n_samples, n_dimensions=self.n_dimensions, bounds=self.bounds, rounding=self.rounding, seed=seed)
        elif method == 'custom':
            if 'custom_sampler' in kwargs:
                custom_sampler = kwargs['custom_sampler']
                return custom_sampler(n_samples=n_samples, n_dimensions=self.n_dimensions, bounds=self.bounds, seed=seed, **kwargs)
            else:
                raise ValueError("For 'custom' method, please provide a 'custom_sampler' function in kwargs.")
        else:
            raise ValueError(f"Unknown sampling method: {method}.\nAvailable methods: {self.available_methods}")

    def compute_function_values(self, X: np.ndarray, function, noise_std: float=0.0, **kwargs) -> np.ndarray:
        """Compute function values for the given input samples.

        Args:
            X (np.ndarray): Input samples of shape (n_samples, n_dimensions).
            function (_type_): Function to compute the output values.
            noise_std (float, optional): Standard deviation of the Gaussian noise to add to the output
            **kwargs: Additional keyword arguments to pass to the function.
        Returns:
            np.ndarray: Computed function values of shape (n_samples,).
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        f = function(dim=self.n_dimensions, noise_std=noise_std, negate=self.negate_y, **kwargs)
        y = f(X_tensor).numpy()
        return y

    def generate_dataset(
        self, 
        function, 
        n_samples: int, 
        method: str='lhs', 
        val_size: float=0.2, 
        noise_std: float=0.0, 
        **kwargs) -> pd.DataFrame:
        """Generate a dataset by sampling the input space and computing function values.

        Args:
            function (_type_): Function to compute the output values.
            n_samples (int): Number of samples to generate.
            method (str, optional): Sampling method to use. Defaults to 'lhs'.
            val_size (float, optional): Proportion of the dataset to include in the validation split. Defaults to 0.2.
            noise_std (float, optional): Standard deviation of the Gaussian noise to add to the output. Defaults to 0.0.
            **kwargs: Additional keyword arguments to pass to the sampling ('sampling') and function ('function') computation.

        Returns:
            pd.DataFrame: Generated dataset with input features and corresponding output values.
        """
        kwargs_sampling = kwargs.get('sampling', {})
        kwargs_function = kwargs.get('function', {})

        X_train = self.sample_space(n_samples, method=method, n_dimensions=self.n_dimensions, bounds=self.bounds, seed=self.seed, **kwargs_sampling)
        y_train = self.compute_function_values(X_train, function, noise_std=noise_std, **kwargs_function)

        if val_size > 0.0:
            X_val = self.sample_space(int(n_samples * val_size), method=method, n_dimensions=self.n_dimensions, bounds=self.bounds, seed=self.seed+13, **kwargs_sampling)
            y_val = self.compute_function_values(X_val, function, noise_std=noise_std, **kwargs_function)

        # Add the possibility of handling multi-target outputs in the future
        if len(y_train.shape) > 1:
            if len(self.target_label) > 1:
                assert y_train.shape[1] == len(self.target_label), "Number of target labels must match output dimensions"
            else:
                self.target_label = [f'y{i+1}' for i in range(y_train.shape[1])]

        # create a complete dataset dataframe with train/val splits as additional columns
        train_data = pd.DataFrame(X_train, columns=self.dimension_names)
        for i, label in enumerate(self.target_label):
            train_data[label] = y_train[:, i] if len(y_train.shape) > 1 else y_train
        train_data['set'] = 'train'

        if val_size > 0.0:
            val_data = pd.DataFrame(X_val, columns=self.dimension_names)
            for i, label in enumerate(self.target_label):
                val_data[label] = y_val[:, i] if len(y_val.shape) > 1 else y_val
            val_data['set'] = 'val'
        elif val_size == 0.0:
            val_data = pd.DataFrame(columns=train_data.columns)

        dataset = pd.concat([train_data, val_data], ignore_index=True)
        return dataset


def lhs_sampling(
    n_samples: int, 
    n_dimensions: int, 
    bounds: np.ndarray, 
    rounding: list[int]|int=None, 
    seed=None
) -> np.ndarray:
    """Generate samples using Latin Hypercube Sampling (LHS).

    Args:
        n_samples (int): Number of samples to generate.
        n_dimensions (int): Number of dimensions in the search space.
        bounds (np.ndarray): Array of shape (n_dimensions, 2) specifying the (min, max) bounds for each dimension.
        rounding (list[int] | int, optional): List specifying the number of decimal places to round each dimension. Defaults to None.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Array of shape (n_samples, n_dimensions) containing the generated samples.
    """
    assert len(bounds) == n_dimensions, "Length of bounds must match n_dimensions"

    sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)
    sample = sampler.random(n=n_samples)
    l_bounds = [b[0] for b in bounds]
    u_bounds = [b[1] for b in bounds]
    scaled_sample = qmc.scale(sample, l_bounds, u_bounds)

    if rounding is not None:
        if isinstance(rounding, int):
            rounding = [rounding] * n_dimensions
        for dim in range(n_dimensions):
            scaled_sample[:, dim] = np.round(scaled_sample[:, dim], rounding[dim])

    return scaled_sample


def sobol_sampling(
    n_samples: int, 
    n_dimensions: int, 
    bounds: np.ndarray, 
    rounding: list[int]|int=None, 
    seed=None
) -> np.ndarray:
    """Generate samples using Sobol sequence.

    Args:
        n_samples (int): Number of samples to generate.
        n_dimensions (int): Number of dimensions in the search space.
        bounds (np.ndarray): Array of shape (n_dimensions, 2) specifying the (min, max) bounds for each dimension.
        rounding (list[int] | int, optional): List specifying the number of decimal places to round each dimension. Defaults to None.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Array of shape (n_samples, n_dimensions) containing the generated samples.
    """
    assert len(bounds) == n_dimensions, "Length of bounds must match n_dimensions"

    sampler = qmc.Sobol(d=n_dimensions, scramble=True, seed=seed)
    sample = sampler.random(n=n_samples)
    l_bounds = [b[0] for b in bounds]
    u_bounds = [b[1] for b in bounds]
    scaled_sample = qmc.scale(sample, l_bounds, u_bounds)

    if rounding is not None:
        if isinstance(rounding, int):
            rounding = [rounding] * n_dimensions
        for dim in range(n_dimensions):
            scaled_sample[:, dim] = np.round(scaled_sample[:, dim], rounding[dim])

    return scaled_sample


def random_sampling(
    n_samples: int, 
    n_dimensions: int, 
    bounds: np.ndarray, 
    rounding: list[int]|int=None, 
    seed=None
) -> np.ndarray:
    """Generate samples using random uniform sampling.

    Args:
        n_samples (int): Number of samples to generate.
        n_dimensions (int): Number of dimensions in the search space.
        bounds (np.ndarray): Array of shape (n_dimensions, 2) specifying the (min, max) bounds for each dimension.
        rounding (list[int] | int, optional): List specifying the number of decimal places to round each dimension. Defaults to None.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Array of shape (n_samples, n_dimensions) containing the generated samples.
    """
    if seed is not None:
        np.random.seed(seed)

    assert len(bounds) == n_dimensions, "Length of bounds must match n_dimensions"

    samples = np.zeros((n_samples, n_dimensions))
    for dim in range(n_dimensions):
        min_bound, max_bound = bounds[dim]
        samples[:, dim] = np.random.uniform(min_bound, max_bound, n_samples)
        if rounding is not None:
            if isinstance(rounding, int):
                round_val = rounding
            else:
                round_val = rounding[dim]
            samples[:, dim] = np.round(samples[:, dim], round_val)

    return samples


# --------------------------------------------------------------
# Adaptive refinement functions


def compute_reference_distance(
    X: np.ndarray,
    method: str = "median_nn",
    gp_lengthscale: np.ndarray = None
) -> float:
    """
    Compute a reference distance scale for threshold setting.

    Args:
        X_train: (N, d) array of existing training points (can be candidates too).
        method: "median_nn" | "mean_nn" | "gp_lengthscale"
        gp_lengthscale: None or scalar or (d,) array of kernel lengthscales.

    Returns:
        ref: scalar reference distance (or geometric mean of lengthscales if gp_lengthscale provided)
    """
    if X is None or len(X) < 2:
        # fallback: use 1.0
        if gp_lengthscale is not None:
            ls = np.asarray(gp_lengthscale)
            return float(np.exp(np.mean(np.log(np.abs(ls) + 1e-12))))
        return 1.0

    if method in ("median_nn", "mean_nn"):
        tree = cKDTree(X)
        # query k=2 because first neighbor is the point itself
        dists, _ = tree.query(X, k=2)
        # dists[:,0] is zero (self), dists[:,1] is nearest neighbor
        nn = dists[:, 1]
        if method == "median_nn":
            return float(np.median(nn))
        else:
            return float(np.mean(nn))
        
    elif method == "gp_lengthscale":
        if gp_lengthscale is None:
            raise ValueError("gp_lengthscale must be provided for method 'gp_lengthscale'")
        ls = np.asarray(gp_lengthscale)
        # convert per-dim lengthscales to a scalar reference, use geometric mean
        return float(np.exp(np.mean(np.log(np.abs(ls) + 1e-12))))
    else:
        raise ValueError(f"Unknown method {method}")
    

def pointwise_hypercube_refinement(
    refine_generator: DatasetGenerator,
    design_bounds: np.ndarray,
    design_dimensions: int,
    refine_centroid: np.ndarray, 
    half_side_length: float, 
    n_points: int,
    refine_noise_std: float,
    scaler,
    refine_function,
    refine_method: str = 'lhs'
) -> pd.DataFrame:
    """Generate new candidate points within a hypercube centered at a given centroid.

    Args:
        refine_generator (DatasetGenerator): DatasetGenerator instance for generating points.
        refine_centroid (np.ndarray): Centroid of the hypercube (array of shape (1, n_dimensions) in the reduced space).
        half_side_length (float): Half the length of the hypercube's side.
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


# TODO: possibility to used adaptive min_distance based on points-to-keep thresholding
def filter_refined_additions(
    X_addition: np.ndarray,
    X_existing: np.ndarray,
    min_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Filter out points from X_addition that are too close to the X_existing points and 
    return the filtered points with the indices of the points that were kept.

    Args:
        X_addition (np.ndarray): Points proposed for addition.
        X_existing (np.ndarray): Existing points to compare against.
        min_distance (float): Minimum allowable distance from existing points.

    Returns:
        tuple[np.ndarray, np.ndarray]: Filtered points and their indices.
    """
    tree = cKDTree(X_existing)
    filtered_points = []
    filtered_indices = []

    for idx, point in enumerate(X_addition):
        dist, _ = tree.query(point, k=1)
        if dist >= min_distance:
            filtered_points.append(point)
            filtered_indices.append(idx)

    return np.array(filtered_points), np.array(filtered_indices)