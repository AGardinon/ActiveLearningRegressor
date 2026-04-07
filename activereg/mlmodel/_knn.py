#!
import numpy as np
from scipy.spatial.distance import cdist

from typing import Tuple


class kNNRegressorAL:
    """
    kNN regressor with geometry-driven uncertainty for Active Learning.

    Uncertainty is defined as the weighted local variance of neighbor targets.
    This is a non-Bayesian uncertainty proxy suitable for AL benchmarks.
    """
    def __init__(
        self,
        k: int = 10,
        length_scale: float = 1.0,
        noise_floor: float = 1e-6,
        distance_penalty: float = 0.0
    ):
        self.k = k
        self.length_scale = length_scale
        self.noise_floor = noise_floor
        self.distance_penalty = distance_penalty

        self.X_train = None
        self.y_train = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).ravel()

    def _weights(self, distances: np.ndarray) -> np.ndarray:
        w = np.exp(-(distances ** 2) / (self.length_scale ** 2))
        return w / np.sum(w, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.X_train is None:
            raise RuntimeError("Model must be fitted before prediction.")

        X = np.asarray(X)

        # Pairwise distances to training data
        distances = cdist(X, self.X_train)

        # Explicit effective k (in case k > n_train)
        k_eff = min(self.k, self.X_train.shape[0])

        # Extract neighbor distances and values
        knn_idx = np.argsort(distances, axis=1)[:, :k_eff]

        knn_distances = np.take_along_axis(distances, knn_idx, axis=1)
        knn_targets = self.y_train[knn_idx]

        # Compute weights
        weights = self._weights(knn_distances)

        # Predictive mean
        mean = np.sum(weights * knn_targets, axis=1)

        # Weighted local variance for uncertainty
        var = np.sum(
            weights * (knn_targets - mean[:, None]) ** 2,
            axis=1,
        )

        # Add noise floor
        var += self.noise_floor

        # Optional distance-based inflation
        if self.distance_penalty > 0.0:
            avg_dist = np.mean(knn_distances, axis=1)
            var *= (1.0 + self.distance_penalty * avg_dist)

        std = np.sqrt(var)

        return mean, mean, std

    def __repr__(self) -> str:
        return (f"kNNRegressorAL(k={self.k}, length_scale={self.length_scale}, "
                f"noise_floor={self.noise_floor}, distance_penalty={self.distance_penalty})")
