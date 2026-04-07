#!
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import pyro
import pyro.distributions as dist
from pyro.nn import PyroSample, PyroModule
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.optim import Adam

from typing import Tuple, List, Union, Optional


#  - BAYESIAN NN
# TODO: check problem with the dimension of y as (N, 1) gives problems  and (N,) not, but then (N,) is not compatible with the rest
# of the analysis and/or other models
# TODO: add output in numpy form not torch

class BayesianNN:
    """
    Bayesian Neural Network wrapper with train() and predict() functions.
    """
    def __init__(
        self,
        in_feats: int=5,
        hidden_layers: Union[List[int], int]=[32, 32],
        out_feats: int=1,
        activation: str='relu',
        seed: int=42,
        lr: float=1e-3,
        to_gpu: bool=True,
        epochs: int=10000,
        weight_mu: float=0.0,
        weight_sigma: float=1.0,
        bias_mu: float=0.0,
        bias_sigma: float=1.0,
        noise_prior_low: float=0.01,
        noise_prior_high: float=1.0,
        log_transform: bool=False,
        guide_type: str="diagonal"  # "diagonal" or "multivariate"
    ):
        """
        Initialize Bayesian Neural Network.

        Args:
            in_feats: Input feature dimension
            hidden_layers: Either list of hidden layer sizes or single int for uniform layers
            out_feats: Output dimension
            activation: Activation function name
            seed: Random seed
            lr: Learning rate
            to_gpu: Whether to use GPU
            epochs: Default number of training epochs
            weight_mu: Prior mean for weights
            weight_sigma: Prior std for weights
            bias_mu: Prior mean for biases
            bias_sigma: Prior std for biases
            noise_prior_low: Lower bound for noise prior
            noise_prior_high: Upper bound for noise prior
            log_transform: Whether to apply log transform to targets
            guide_type: Type of variational guide ("diagonal" or "multivariate")
        """
        # Handle backward compatibility for hidden_size parameter
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers, hidden_layers]

        # Define some vars and seed random state
        self.lr = lr
        self.train_losses = []
        self.epochs = epochs
        self.epoch = 0
        self.seed = seed
        self.log_transform = log_transform
        self.device = torch.device("cuda" if torch.cuda.is_available() and to_gpu else "cpu")

        self.config = {
        'in_feats': in_feats,
        'hidden_layers': hidden_layers,
        'out_feats': out_feats,
        'activation': activation,
        'seed': seed,
        'lr': lr,
        'to_gpu': to_gpu,
        'epochs': epochs,
        'weight_mu': weight_mu,
        'weight_sigma': weight_sigma,
        'bias_mu': bias_mu,
        'bias_sigma': bias_sigma,
        'noise_prior_low': noise_prior_low,
        'noise_prior_high': noise_prior_high,
        'log_transform': log_transform,
        'guide_type': guide_type,
        }

        # Set random seeds
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
        np.random.seed(seed)

        # Initialize model
        self.model = pyroFNN(
            in_feats=in_feats,
            hidden_layers=hidden_layers,
            out_feats=out_feats,
            activation=activation,
            weight_prior_mu=weight_mu,
            weight_prior_sigma=weight_sigma,
            bias_prior_mu=bias_mu,
            bias_prior_sigma=bias_sigma,
            noise_prior_low=noise_prior_low,
            noise_prior_high=noise_prior_high,
            device=self.device
        )

        # Initialize Guide model
        if guide_type.lower() == "multivariate":
            self.guide = AutoMultivariateNormal(self.model)
        else:  # diagonal
            self.guide = AutoDiagonalNormal(self.model)

        self.guide = self.guide.to(self.device)

        # Initialize optimizer
        adam = Adam({"lr": lr})

        # Stochastic Variational Inference
        self.svi = SVI(self.model, self.guide, adam, loss=Trace_ELBO())

    def numpy_to_dataloader(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        batch_size: int = 256,
        shuffle: bool = True
    ) -> DataLoader:
        """Convert numpy arrays to PyTorch DataLoader."""

        # Convert to tensors
        x_tensor = torch.as_tensor(x, dtype=torch.float32)

        if y is not None:
            y = y.ravel()
            y_tensor = torch.as_tensor(y, dtype=torch.float32)
            dataset = TensorDataset(x_tensor, y_tensor)
        else:
            dataset = TensorDataset(x_tensor)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: Optional[int]=None,
        batch_size: int=256
    ) -> None:
        """
        Train the Bayesian Neural Network.

        Args:
            X: Input features (numpy array)
            Y: Target values (numpy array)
            epochs: Number of training epochs (uses self.epochs if None)
            batch_size: Batch size for training
        """

        # Apply log transform if specified
        if self.log_transform:
            Y = np.log10(Y)

        # Convert numpy to DataLoader
        data_loader = self.numpy_to_dataloader(X, Y, batch_size=batch_size, shuffle=True)

        # Clear parameter store
        pyro.clear_param_store()

        # Training loop
        num_epochs = self.epochs if epochs is None else epochs

        for epoch in range(num_epochs):
            running_loss = 0.0
            n_samples = 0

            for batch in data_loader:
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)

                # ELBO gradient and add loss to running loss
                loss = self.svi.step(x_batch, y_batch)
                running_loss += loss
                n_samples += x_batch.shape[0]

            # Calculate average loss
            avg_loss = running_loss / n_samples
            self.train_losses.append(avg_loss)
            self.epoch = epoch + 1

    def predict(
        self,
        X: np.ndarray,
        num_samples: int=20,
        return_posterior: bool=False,
        batch_size: int=256
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Make predictions with uncertainty quantification.

        Args:
            X: Input features (numpy array)
            num_samples: Number of posterior samples
            return_posterior: Whether to return posterior samples
            batch_size: Batch size for prediction

        Returns:
            Tuple of (samples, mean, std) or (samples, mean, std, posterior)
        """

        # Convert numpy to DataLoader
        data_loader = self.numpy_to_dataloader(X, batch_size=batch_size, shuffle=False)

        # Construct predictive distribution
        predictive = Predictive(
            self.model,
            guide=self.guide,
            num_samples=num_samples,
            return_sites=("obs", "_RETURN")
        )

        # Initialize result tensors
        y_hat = torch.tensor([], device=self.device)
        y_hat_mu = torch.tensor([], device=self.device)
        y_hat_sigma = torch.tensor([], device=self.device)
        posterior = torch.tensor([], device=self.device)

        # Prediction loop
        with torch.no_grad():
            for batch in data_loader:
                x_batch = batch[0].to(self.device)

                # Reshape if needed (handle 1D case)
                if len(x_batch.size()) == 1:
                    x_batch = x_batch.unsqueeze(0)

                # Get samples from predictive distribution
                samples = predictive(x_batch)

                # Handle reshaping for return samples if needed
                if len(samples['_RETURN'].size()) == 1 and return_posterior:
                    samples['_RETURN'] = samples['_RETURN'].reshape([num_samples, -1])

                # Apply inverse log transform if specified
                preds = 10 ** samples['obs'] if self.log_transform else samples['obs']
                posts = 10 ** samples['_RETURN'] if self.log_transform else samples['_RETURN']

                # Accumulate predictions
                y_hat = torch.cat((y_hat, preds.T), 0)
                y_hat_mu = torch.cat((y_hat_mu, torch.mean(preds, dim=0)), 0)
                y_hat_sigma = torch.cat((y_hat_sigma, torch.std(preds, dim=0)), 0)

                if return_posterior:
                    posterior = torch.cat((posterior, posts.T), 0)

        # Move to CPU for return and to numpy
        y_hat_cpu = y_hat.cpu().numpy()
        y_hat_mu_cpu = y_hat_mu.cpu().numpy()
        y_hat_sigma_cpu = y_hat_sigma.cpu().numpy()

        if return_posterior:
            return y_hat_cpu, y_hat_mu_cpu, y_hat_sigma_cpu, posterior.cpu().numpy()

        return y_hat_cpu, y_hat_mu_cpu, y_hat_sigma_cpu

    def get_model_info(self) -> dict:
        """Get information about the model."""
        return self.config

    def get_training_history(self) -> dict:
        """Get training history."""
        return {
            'losses': self.train_losses,
            'epochs_trained': self.epoch,
            'final_loss': self.train_losses[-1] if self.train_losses else None
        }

    def __repr__(self) -> str:
        return f"BayesianNN(log_transform={self.log_transform})"


class pyroFNN(PyroModule):
    """
    Simple FNN using pyro modules.
    """
    def __init__(
        self,
        in_feats: int=5,
        hidden_layers: List[int]=[32, 32],
        out_feats: int=1,
        activation: str='relu',
        weight_prior_mu: float=0.0,
        weight_prior_sigma: float=1.0,
        bias_prior_mu: float=0.0,
        bias_prior_sigma: float=1.0,
        noise_prior_low: float=0.01,
        noise_prior_high: float=1.0,
        device: Optional[torch.device]=None
    ) -> None:
        super().__init__()

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Network architecture
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_layers = hidden_layers

        # Activation function
        self.activation_name = activation
        self.activation = self._get_activation(activation)

        # Prior parameters
        self.weight_prior_mu = weight_prior_mu
        self.weight_prior_sigma = weight_prior_sigma
        self.bias_prior_mu = bias_prior_mu
        self.bias_prior_sigma = bias_prior_sigma
        self.noise_prior_low = noise_prior_low
        self.noise_prior_high = noise_prior_high

        # Build the network layers
        self.layers = self._build_layers()

        # Move to device
        self.to(self.device)


    def _get_activation(self, activation: str):
        """Get activation function by name."""
        activations = {
            'relu': F.relu,
            'gelu': F.gelu,
            'elu': F.elu,
            'leaky': F.leaky_relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid
        }
        if activation not in activations:
            raise ValueError(f"Activation '{activation}' not supported. Choose from {list(activations.keys())}")
        return activations[activation]


    def _build_layers(self):
        """Build the network layers dynamically."""
        layers = nn.ModuleList()

        # Create layer dimensions
        layer_dims = [self.in_feats] + self.hidden_layers + [self.out_feats]

        # Build each layer
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]

            # Create PyroModule layer
            layer = PyroModule[nn.Linear](in_dim, out_dim)

            # Set Bayesian weights and biases with UNIQUE sample names
            layer.weight = PyroSample(
                lambda layer_self, outer_self=self: dist.Normal(
                    outer_self.weight_prior_mu,
                    torch.tensor(outer_self.weight_prior_sigma, device=outer_self.device)
                ).expand([layer_self.out_features, layer_self.in_features]).to_event(2)
            )

            layer.bias = PyroSample(
                lambda layer_self, outer_self=self: dist.Normal(
                    outer_self.bias_prior_mu,
                    torch.tensor(outer_self.bias_prior_sigma, device=outer_self.device)
                ).expand([layer_self.out_features]).to_event(1)
            )

            # Rename layer module to avoid name collision in PyroModule
            setattr(self, f"layer_{i}", layer)
            layers.append(layer)

        return layers


    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor
            y: Target tensor (for conditioning during training)

        Returns:
            Mean predictions
        """
        # Ensure input is on the correct device
        if x.device != self.device:
            x = x.to(self.device)

        # Forward pass through hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)

        # Output layer (no activation)
        mu = self.layers[-1](x).squeeze(-1) if self.out_feats == 1 else self.layers[-1](x)

        # Sample observation noise
        sigma = pyro.sample(
            "sigma",
            dist.Uniform(
                torch.tensor(self.noise_prior_low, device=self.device),
                torch.tensor(self.noise_prior_high, device=self.device)
            )
        )

        # Likelihood
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)

        return mu
