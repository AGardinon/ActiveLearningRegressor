#!
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, ConstantKernel, WhiteKernel
)

from typing import List, Dict, Union


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
