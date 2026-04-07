#!
import warnings
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning

from typing import Tuple, Dict, Any


class GPR:
    """
    Gaussian Process Regressor wrapper with log-transform support
    and a unified predict interface for Active Learning.
    """
    def __init__(self,
                 log_transform: bool = False,
                 **kwargs) -> None:

        self.log_transform = log_transform
        self.eps = 1e-8

        # Initialize base model — all sklearn GPR params forwarded via kwargs
        self.model = GaussianProcessRegressor(**kwargs)

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train the Gaussian Process model."""
        if self.log_transform:
            y = np.log10(y + self.eps)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        y_hat_mean, y_hat_uncertainty = self.model.predict(x, return_std=True)

        if self.log_transform:
            # Store the log-scale predictions
            y_hat_log = y_hat_mean
            # Convert mean back to original scale
            y_hat_mean = 10 ** y_hat_log
            # Apply delta method using log-scale values
            # Derivative of 10^x is 10^x * ln(10)
            derivative = y_hat_mean * np.log(10)  # This is 10^y_hat_log * ln(10)
            variance_original = (derivative ** 2) * (y_hat_uncertainty ** 2)
            y_hat_uncertainty = np.sqrt(variance_original)

        # Dummy variable as they are identical
        # in Ensemble methods y_hat is the set of predictions
        y_hat = y_hat_mean
        return y_hat, y_hat_mean, y_hat_uncertainty

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test set."""
        _, y_pred, _ = self.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }

    def __repr__(self) -> str:
        try:
            return f"GPR(log_transform={self.log_transform}, kernel={self.model.kernel_}, trained=True)"
        except AttributeError:
            return f"GPR(log_transform={self.log_transform}, trained=False)"
