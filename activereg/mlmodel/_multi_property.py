#!
import numpy as np
from typing import Tuple, Protocol, runtime_checkable

from activereg.mlmodel._base import MLModel


@runtime_checkable
class MultiPropertyMLModel(Protocol):
    """Protocol for multi-property (multi-output) ML models used in active learning."""

    target_names: list[str]

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Train on X (N, d) and Y (N, P)."""
        ...

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (y_hat, mu, sigma), each of shape (N, P)."""
        ...

    def predict_property(
        self, X: np.ndarray, name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (y_hat, mu, sigma) for a single named property, each of shape (N,)."""
        ...

    def __repr__(self) -> str: ...


class IndependentMultiPropertyModel:
    """Dict of independent single-output MLModels, one per target property.

    ``target_names`` ordering is fixed at construction time and determines
    the column order of Y in ``train`` and the stacked outputs of ``predict``.
    """

    def __init__(self, models: dict[str, MLModel]) -> None:
        if not models:
            raise ValueError("models dict must not be empty")
        self._models: dict[str, MLModel] = dict(models)  # preserve insertion order
        self.target_names: list[str] = list(self._models.keys())

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        if Y.ndim != 2 or Y.shape[1] != len(self.target_names):
            raise ValueError(
                f"Y must be 2D with shape (N, {len(self.target_names)}), got {Y.shape}"
            )
        for i, name in enumerate(self.target_names):
            self._models[name].train(X, Y[:, i])

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mus, sigmas = [], []
        for name in self.target_names:
            _, mu_i, sigma_i = self._models[name].predict(X)
            mus.append(mu_i)
            sigmas.append(sigma_i)
        mu = np.column_stack(mus)      # (N, P)
        sigma = np.column_stack(sigmas)
        return mu, mu, sigma           # y_hat == mu for independent models

    def predict_property(
        self, X: np.ndarray, name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if name not in self._models:
            raise KeyError(f"Unknown property '{name}'. Available: {self.target_names}")
        return self._models[name].predict(X)

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{name}: {self._models[name]!r}" for name in self.target_names
        )
        return f"{self.__class__.__name__}({{{parts}}})"


def wrap_single_property(
    model: MLModel, target_name: str
) -> IndependentMultiPropertyModel:
    """Wrap a single-property MLModel so it satisfies MultiPropertyMLModel."""
    return IndependentMultiPropertyModel({target_name: model})
