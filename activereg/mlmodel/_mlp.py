#!
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import List, Optional


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
