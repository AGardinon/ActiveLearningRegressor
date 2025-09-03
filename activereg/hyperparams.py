#!

from typing import Union
from activereg.mlmodel import KernelFactory
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel


GPR = {
    'RBF_W': ConstantKernel() * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.),
    'MATERN_W': ConstantKernel() * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.),
}

def get_gp_kernel(kernel_recipe: Union[str, list]):
    """Gets the Gaussian Process kernel from a recipe.

    Args:
        kernel_recipe (str or list): Name of the kernel or list of kernel components.

    Returns:
        _type_: The corresponding Gaussian Process kernel.
    """
    if isinstance(kernel_recipe, str):
        return get_default_gp_kernel(kernel_recipe)
    
    elif isinstance(kernel_recipe, list):
        return get_custom_gp_kernel(kernel_recipe)
    
    else:
        raise ValueError("kernel_recipe must be either a string or a list.")


def get_default_gp_kernel(kernel_recipe: str):
    """Gets the Gaussian Process kernel by name.

    Args:
        name (str): Name of the kernel.

    Returns:
        _type_: The corresponding Gaussian Process kernel.
    """
    if kernel_recipe not in GPR:
        raise ValueError(f"Kernel '{kernel_recipe}' is not defined. Choose from {list(GPR.keys())}.")
    return GPR[kernel_recipe]


def get_custom_gp_kernel(kernel_recipe: list):
    """Gets a custom Gaussian Process kernel from a recipe.

    Args:
        kernel_recipe (list): List of kernel components.

    Returns:
        _type_: The corresponding custom Gaussian Process kernel.
    """
    kernel_factory = KernelFactory(kernel_recipe)
    return kernel_factory.get_kernel()
