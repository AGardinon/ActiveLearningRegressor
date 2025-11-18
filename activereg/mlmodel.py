#!
import warnings
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.exceptions import ConvergenceWarning

import pyro
import pyro.distributions as dist
from pyro.nn import PyroSample, PyroModule
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.optim import Adam

from typing import Tuple, List, Dict, Union, Protocol, Optional, Any

# Generic MLModel class

class MLModel(Protocol):
    def train(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def __repr__(self) -> str: ...

# - GAUSSIAN PROCESS REGRESSOR WITH INTEGRATED GRIDSEARCH

class GPR:
    """
    Gaussian process regressors with hyperparameter tuning using GridSearchCV
    """
    def __init__(self, 
                 log_transform: bool = False, 
                 use_gridsearch: bool = True,
                 param_grid: Optional[Dict[str, Any]] = None,
                 cv: int = 5,
                 scoring: str = 'neg_mean_squared_error',
                 n_jobs: int = -1,
                 verbose: int = 0,
                 **kwargs) -> None:
        
        self.log_transform = log_transform
        self.eps = 1e-8
        self.use_gridsearch = use_gridsearch
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Handle different scoring options
        if scoring == 'rmse' or scoring == 'neg_rmse':
            # Create custom RMSE scorer (negated for GridSearchCV maximization)
            self.scoring_func = make_scorer(
                lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)), 
                greater_is_better=False
            )
        elif scoring == 'neg_root_mean_squared_error':
            # Use sklearn's built-in RMSE scorer (available in sklearn >= 0.22)
            self.scoring_func = 'neg_root_mean_squared_error'
        else:
            # Use provided scoring metric
            self.scoring_func = scoring
        
        # Default parameter grid if none provided
        if param_grid is None:
            self.param_grid = {
                'kernel' : [
                    ConstantKernel(1.0) * RBF(lc) + WhiteKernel(noise_level=noise)
                    for lc in [0.01, 0.1, 1.0, 10]
                    for noise in [.75, .5, .25, 0.1]
                ],
                'alpha': [1e-12, 1e-10, 1e-8, 1e-6, 1e-4],
                'normalize_y': [True, False],
                'n_restarts_optimizer': [50]
            }
        else:
            self.param_grid = param_grid
        
        # Initialize base model
        self.base_model = GaussianProcessRegressor(**kwargs)
        
        # Initialize GridSearchCV if requested
        if self.use_gridsearch:
            self.model = GridSearchCV(
                estimator=self.base_model,
                param_grid=self.param_grid,
                cv=self.cv,
                scoring=self.scoring_func,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
        else:
            self.model = self.base_model
    
    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train the model with optional hyperparameter tuning"""
        if self.log_transform:
            y = np.log10(y + self.eps)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            # warnings.filterwarnings("ignore", category=Warning)
            self.model.fit(x, y)
        
        # Store best parameters if using gridsearch
        if self.use_gridsearch:
            self.best_params_ = self.model.best_params_
            self.best_score_ = self.model.best_score_
            self.cv_results_ = self.model.cv_results_
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates"""
        # Get the best estimator if using gridsearch
        if self.use_gridsearch:
            predictor = self.model.best_estimator_
        else:
            predictor = self.model
        
        y_hat_mean, y_hat_uncertainty = predictor.predict(x, return_std=True)
        
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
    
    def get_best_params(self) -> Dict[str, Any]:
        """Return the best parameters found during gridsearch"""
        if self.use_gridsearch and hasattr(self, 'best_params_'):
            return self.best_params_
        else:
            return {}
    
    def get_best_score(self) -> float:
        """Return the best cross-validation score"""
        if self.use_gridsearch and hasattr(self, 'best_score_'):
            return self.best_score_
        else:
            return None
    
    def get_cv_results(self) -> Dict[str, Any]:
        """Return detailed cross-validation results"""
        if self.use_gridsearch and hasattr(self, 'cv_results_'):
            return self.cv_results_
        else:
            return {}
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test set"""
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
            if self.use_gridsearch:
                best_kernel = self.model.best_estimator_.kernel_
                return f"GPR(log_transform={self.log_transform}, gridsearch=True, best_kernel={best_kernel}, trained=True)"
            else:
                return f"GPR(log_transform={self.log_transform}, gridsearch=False, kernel={self.model.kernel_}, trained=True)"
        except AttributeError:
            return f"GPR(log_transform={self.log_transform}, gridsearch={self.use_gridsearch}, trained=False)"


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

# - MLP
# TODO: add device allocation, batch size for training, epochs as a optional parameter in train()

class AnchoredEnsembleMLP:
    def __init__(
        self,
        n_models: int, 
        in_feats: int,
        out_feats: int,
        hidden_layers: List[int] = [64],
        activation: str = 'relu', 
        lambda_anchor: float = 1e-4, 
        lr: float = 1e-3,
        n_epochs: int = 100,
        device: str = None
    ):

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.models = []
        self.optimizers = []
        self.anchors = []
        self.lambda_anchor = lambda_anchor
        self.n_epochs = n_epochs
        
        for _ in range(n_models):
            model = MLP(
                in_feats=self.in_feats, 
                hidden_layers=hidden_layers, 
                out_feats=self.out_feats, 
                activation=activation
            ).to(self.device)

            # Save anchor (initial params)
            anchor = [p.detach().clone().to(self.device) for p in model.parameters()]
            opt = optim.Adam(model.parameters(), lr=lr)
            
            self.models.append(model)
            self.optimizers.append(opt)
            self.anchors.append(anchor)

    def train(self, X, Y, epochs: Optional[int]=None, train_bar: bool=False):
        num_epochs = self.n_epochs if epochs is None else epochs
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.as_tensor(Y, dtype=torch.float32).to(self.device)
        bar = trange(num_epochs, desc="Training Anchored Ensemble") if train_bar else range(num_epochs)

        for _ in bar:
            for model, opt, anchor in zip(self.models, self.optimizers, self.anchors):
                opt.zero_grad()
                pred = model(X)
                mse = torch.mean((pred - Y)**2)
                
                # Anchoring penalty
                anchor_loss = 0.0
                for p, p0 in zip(model.parameters(), anchor):
                    anchor_loss += torch.sum((p - p0)**2)
                loss = mse + self.lambda_anchor * anchor_loss
                loss.backward()
                opt.step()
    
    @torch.no_grad()
    def predict(self, X):
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        preds = []
        for model in self.models:
            preds.append(model(X).detach().cpu().numpy())
        preds = np.stack(preds, axis=0)  # shape (n_models, N, out_dim)
        mean = preds.mean(axis=0)        # (N, out_dim)
        std = preds.std(axis=0)          # epistemic uncertainty, (N, out_dim)
        if self.out_feats == 1:
            mean = mean.squeeze(-1)  # (N,)
            std = std.squeeze(-1)  # (N,)
        return preds, mean, std
    

class MLP(nn.Module):
    def __init__(
        self, 
        in_feats: int, 
        hidden_layers: List[int], 
        out_feats: int = 1, 
        activation: str = "relu"
    ):
        """
        Multi-layer Perceptron (MLP) with flexible hidden layers.
        
        Args:
            in_feats (int): Number of input features.
            hidden_layers (list[int]): List with neurons per hidden layer, e.g. [32, 64].
            out_feats (int): Number of output features (targets).
            activation (str): Activation function ('relu', 'tanh', 'sigmoid').
        """
        super().__init__()
        
        # Select activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unknown activation {activation}")
        
        # Build layers
        layers = []
        prev_dim = in_feats
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_feats))  # final output layer
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)  # output layer without activation (regression)
        return x


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
        # bar = trange(num_epochs, desc="Training BNN")
        
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
            
            # Update progress bar
            # bar.set_postfix(loss=f'{avg_loss:.4f}')

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
            # print(f"Building layer {i}: in_dim={in_dim}, out_dim={out_dim}")

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
            # print(f"[Forward] x shape before layer {i}: {x.shape}")
            # print(layer)
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