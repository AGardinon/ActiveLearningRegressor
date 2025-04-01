#!
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from typing import Tuple, List, Dict, Union

# - GAUSSIAN PROCESS REGRESSOR

class GPR:
    """
    Gaussian process regressors
    """
    def __init__(self, log_transform: bool = False, **kwargs) -> None:
        self.model = GaussianProcessRegressor(**kwargs)
        self.log_transform = log_transform

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.log_transform:
            y = np.log10(y)
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_hat_mean, y_hat_uncertainty = self.model.predict(x, return_std=True)

        if self.log_transform:
            # Convert mean back to original scale
            y_hat_mean = 10 ** y_hat_mean
            # Apply delta method to transform standard deviation
            variance_original = (10 ** y_hat_mean * np.log(10)) ** 2 * y_hat_uncertainty ** 2
            # Calculate the standard deviation in the original scale
            y_hat_uncertainty = np.sqrt(variance_original)

        # Dummy variable as they are identical
        # in Ensamble methods y_hat is the set of predictions
        y_hat = y_hat_mean

        return y_hat, y_hat_mean, y_hat_uncertainty
    
    def __repr__(self) -> str:
        try:
            return f"GPR(log_transform={self.log_transform}, kernel={self.model.kernel_}, trained=True)"
        except:
            return f"GPR(log_transform={self.log_transform}, kernel={self.model.kernel}, trained=False)"


class KernelFactory:
    '''
    KernelFactory usage.

    Example:
    kernel_recipe = ['+', {'type': 'C', 'constant_value': 2.0}, ['*', 'RBF', {'type': 'W', 'noise_level': 1.0}]]
    kernel_factory = KernelFactory(kernel_recipe)
    kernel = kernel_factory.get_kernel()
    -> kernel = 1.41**2 + RBF(length_scale=1) * WhiteKernel(noise_level=1)

    '''
    def __init__(self, kernel_recipe: List[Union[str,Dict]]):
        self.kernel_recipe = kernel_recipe

    def get_kernel(self):
        kernel_map = {
            'RBF': RBF,
            'Matern': Matern,
            'RationalQuadratic': RationalQuadratic,
            'C': ConstantKernel,
            'W': WhiteKernel,
        }
        
        return self._parse_kernel(self.kernel_recipe, kernel_map)
    
    def _parse_kernel(self, kernel_recipe, kernel_map):
        # Simple string case
        if isinstance(kernel_recipe, str):
            return kernel_map[kernel_recipe]()

        # Dictionary with parameters
        elif isinstance(kernel_recipe, dict):
            kernel_type = kernel_recipe.pop('type')
            return kernel_map[kernel_type](**kernel_recipe)

        # Composite kernel
        elif isinstance(kernel_recipe, list):
            operator = kernel_recipe[0]
            first_kernel = self._parse_kernel(kernel_recipe[1], kernel_map)
            second_kernel = self._parse_kernel(kernel_recipe[2], kernel_map)
            
            if operator == '*':
                return first_kernel * second_kernel
            elif operator == '+':
                return first_kernel + second_kernel
            else:
                raise ValueError(f"Unknown operator: {operator}")

        else:
            raise ValueError(f"Invalid kernel definition: {kernel_recipe}")
        
