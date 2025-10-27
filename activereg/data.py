#!

import torch
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.preprocessing import MinMaxScaler

# TODO: create the DataSampler class to handle all sampling methods and to allow the creation of
# a training and a validation set for BO/AL loops

class DatasetGenerator:

    def __init__(self, n_dimensions: int, bounds: np.ndarray, seed: int=None, dim_labels: list[str]=None, target_label: list[str]=None) -> None:
        """Initialize the DatasetGenerator.

        Args:
            n_dimensions (int): Number of dimensions for the input space.
            bounds (np.ndarray): Array of shape (n_dimensions, 2) specifying the (min, max) bounds for each dimension.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            dim_labels (list[str], optional): List of labels for each dimension. Defaults to None.
        """
        self.n_dimensions = n_dimensions
        self.bounds = bounds
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

    def compute_function_values(self, X: np.ndarray, function, noise_std: float=0.0, negate: bool=False, **kwargs) -> np.ndarray:
        """Compute function values for the given input samples.

        Args:
            X (np.ndarray): Input samples of shape (n_samples, n_dimensions).
            function (_type_): Function to compute the output values.
            noise_std (float, optional): Standard deviation of the Gaussian noise to add to the output
            negate (bool, optional): Whether to negate the function values. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the function.
        Returns:
            np.ndarray: Computed function values of shape (n_samples,).
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        f = function(dim=self.n_dimensions, noise_std=noise_std, negate=negate, **kwargs)
        y = f(X_tensor).numpy()
        return y

    def generate_dataset(self, function, n_samples: int, method: str='lhs', val_size: float=0.2, noise_std: float=0.0, negate: bool=False, scale_y: bool=False, **kwargs) -> pd.DataFrame:
        """Generate a dataset by sampling the input space and computing function values.

        Args:
            function (_type_): Function to compute the output values.
            n_samples (int): Number of samples to generate.
            method (str, optional): Sampling method to use. Defaults to 'lhs'.
            val_size (float, optional): Proportion of the dataset to include in the validation split. Defaults to 0.2.
            noise_std (float, optional): Standard deviation of the Gaussian noise to add to the output. Defaults to 0.0.
            negate (bool, optional): Whether to negate the function values. Defaults to False.
            scale_y (bool, optional): Whether to scale the output values to [0, 1] range. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the sampling ('sampling') and function ('function') computation.

        Returns:
            pd.DataFrame: Generated dataset with input features and corresponding output values.
        """
        kwargs_sampling = kwargs.get('sampling', {})
        kwargs_function = kwargs.get('function', {})

        X_train = self.sample_space(n_samples, method=method, seed=self.seed, **kwargs_sampling)
        y_train = self.compute_function_values(X_train, function, noise_std=noise_std, negate=negate, **kwargs_function)
        if scale_y:
            y_scaler = MinMaxScaler()
            y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

        if val_size > 0.0:
            X_val = self.sample_space(int(n_samples * val_size), method=method, seed=self.seed+13, **kwargs_sampling)
            y_val = self.compute_function_values(X_val, function, noise_std=noise_std, negate=negate, **kwargs_function)
            if scale_y:
                y_scaler = MinMaxScaler()
                y_val = y_scaler.fit_transform(y_val.reshape(-1, 1)).flatten()

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


def lhs_sampling(n_samples: int, n_dimensions: int, bounds: np.ndarray, rounding: list[int]|int=None, seed=None) -> np.ndarray:
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


def sobol_sampling(n_samples: int, n_dimensions: int, bounds: np.ndarray, rounding: list[int]|int=None, seed=None) -> np.ndarray:
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


def random_sampling(n_samples: int, n_dimensions: int, bounds: np.ndarray, rounding: list[int]|int=None, seed=None) -> np.ndarray:
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
