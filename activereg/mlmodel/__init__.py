#!
import importlib

from activereg.mlmodel._base import MLModel
from activereg.mlmodel._gpr import GPR
from activereg.mlmodel._knn import kNNRegressorAL
from activereg.mlmodel._kernels import KernelFactory
from activereg.mlmodel._multi_property import (
    MultiPropertyMLModel,
    IndependentMultiPropertyModel,
    wrap_single_property,
)

# Neural-network models depend on torch/pyro, which are optional extras.
# They are imported on first attribute access so that the core package
# (GPR, kNN, acquisition, cycle, plotting) works without them installed.
_OPTIONAL_MODELS = {
    'MLP': ('activereg.mlmodel._mlp', 'nn'),
    'AnchoredEnsembleMLP': ('activereg.mlmodel._mlp', 'nn'),
    'BayesianNN': ('activereg.mlmodel._bnn', 'nn'),
}


def __getattr__(name: str):
    if name in _OPTIONAL_MODELS:
        module_name, extra = _OPTIONAL_MODELS[name]
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"'{name}' requires the optional '{extra}' dependencies. "
                f"Install them with: pip install 'activereg[{extra}]'"
            ) from exc
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


__all__ = [
    'MLModel',
    'GPR',
    'kNNRegressorAL',
    'AnchoredEnsembleMLP',
    'MLP',
    'BayesianNN',
    'KernelFactory',
    'MultiPropertyMLModel',
    'IndependentMultiPropertyModel',
    'wrap_single_property',
]
