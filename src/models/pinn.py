"""
Neural Network Models for Interface Parameter Identification

Contains:
1. InverseMLPModel - Baseline MLP for inverse problem
2. PINN - Physics-Informed Neural Network with BVP constraints

Author: Sumanth Reddy Settipalli
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict


class InverseMLPModel(nn.Module):
    """
    Baseline MLP for inverse parameter identification.

    Input: (K_eff, G_eff, kappa_inc, mu_inc, kappa_mat, mu_mat, f, R) - 8 features
    Output: (k_bar, lambda_bar, mu_bar, alpha) - 4 parameters
    """

    def __init__(
        self,
        input_dim: int = 8,
        output_dim: int = 4,
        hidden_dims: Tuple[int, ...] = (128, 128, 128),
        activation: str = 'tanh',
        dropout: float = 0.0,
        norm_params: Optional[Dict] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm_params = norm_params

        # Build layers
        layers = []
        prev_dim = input_dim

        # Get activation function
        act_fn = self._get_activation(activation)

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (no activation - we'll apply constraints separately)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Store output parameter bounds
        self.param_bounds = {
            'k_bar': (1.0, 500.0),
            'lambda_bar': (0.01, 5.0),
            'mu_bar': (0.01, 5.0),
            'alpha': (0.0, 1.0),
        }

        # Initialize weights
        self._init_weights()

    def _get_activation(self, name: str):
        activations = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'leaky_relu': nn.LeakyReLU,
        }
        return activations.get(name.lower(), nn.Tanh)

    def _init_weights(self):
        """Xavier initialization for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, apply_constraints: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 8)
            apply_constraints: Whether to enforce parameter bounds

        Returns:
            Output tensor of shape (batch, 4): [k_bar, lambda_bar, mu_bar, alpha]
        """
        # Normalize input if norm_params provided
        if self.norm_params is not None:
            x = self._normalize_input(x)

        # Forward through network
        out = self.network(x)

        if apply_constraints:
            out = self._apply_constraints(out)

        return out

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize inputs using stored parameters."""
        mean = torch.tensor(self.norm_params['input_mean'], device=x.device, dtype=x.dtype)
        std = torch.tensor(self.norm_params['input_std'], device=x.device, dtype=x.dtype)
        std = torch.clamp(std, min=1e-8)  # Avoid division by zero
        return (x - mean) / std

    def _apply_constraints(self, out: torch.Tensor) -> torch.Tensor:
        """
        Apply physical constraints to outputs.

        k_bar: positive (softplus)
        lambda_bar: positive (softplus)
        mu_bar: positive (softplus)
        alpha: [0, 1] (sigmoid)
        """
        k_bar = nn.functional.softplus(out[:, 0]) + self.param_bounds['k_bar'][0]
        lambda_bar = nn.functional.softplus(out[:, 1]) * 0.1 + self.param_bounds['lambda_bar'][0]
        mu_bar = nn.functional.softplus(out[:, 2]) * 0.1 + self.param_bounds['mu_bar'][0]
        alpha = torch.sigmoid(out[:, 3])

        return torch.stack([k_bar, lambda_bar, mu_bar, alpha], dim=1)

    def predict(self, K_eff, G_eff, kappa_inc, mu_inc, kappa_mat, mu_mat, f, R):
        """
        Convenience method for single prediction.
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(
                [[K_eff, G_eff, kappa_inc, mu_inc, kappa_mat, mu_mat, f, R]],
                dtype=torch.float32
            )
            return self.forward(x)[0].numpy()


class PINN(InverseMLPModel):
    """
    Physics-Informed Neural Network for interface parameter identification.

    Extends InverseMLPModel with physics-based loss computation.
    The physics loss enforces that predicted parameters satisfy
    the BVP equations from the Extended Interface model.
    """

    def __init__(
        self,
        input_dim: int = 8,
        output_dim: int = 4,
        hidden_dims: Tuple[int, ...] = (128, 128, 128),
        activation: str = 'tanh',
        dropout: float = 0.0,
        norm_params: Optional[Dict] = None,
        physics_weight: float = 1.0
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_params=norm_params
        )
        self.physics_weight = physics_weight

    def compute_physics_loss(
        self,
        predicted_params: torch.Tensor,
        material_props: torch.Tensor,
        K_target: torch.Tensor,
        G_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute physics loss by verifying predicted parameters produce correct K_eff, G_eff.

        Args:
            predicted_params: (batch, 4) - [k_bar, lambda_bar, mu_bar, alpha]
            material_props: (batch, 6) - [kappa_inc, mu_inc, kappa_mat, mu_mat, f, R]
            K_target: (batch,) - target bulk modulus
            G_target: (batch,) - target shear modulus

        Returns:
            Physics loss (MSE between predicted and target effective properties)
        """
        from extended_interface import compute_effective_properties_torch

        k_bar = predicted_params[:, 0]
        lambda_bar = predicted_params[:, 1]
        mu_bar = predicted_params[:, 2]
        alpha = predicted_params[:, 3]

        kappa_inc = material_props[:, 0]
        mu_inc = material_props[:, 1]
        kappa_mat = material_props[:, 2]
        mu_mat = material_props[:, 3]
        f = material_props[:, 4]
        R = material_props[:, 5]

        # Compute effective properties from predicted parameters
        K_pred, G_pred = compute_effective_properties_torch(
            kappa_inc, mu_inc, kappa_mat, mu_mat,
            k_bar, lambda_bar, mu_bar, alpha, f, R
        )

        # Relative MSE for better scaling
        K_loss = ((K_pred - K_target) / (K_target + 1e-8)) ** 2
        G_loss = ((G_pred - G_target) / (G_target + 1e-8)) ** 2

        return (K_loss.mean() + G_loss.mean()) / 2

    def forward_with_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        compute_physics: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with loss computation.

        Args:
            inputs: (batch, 8) - [K_eff, G_eff, kappa_inc, mu_inc, kappa_mat, mu_mat, f, R]
            targets: (batch, 4) - [k_bar, lambda_bar, mu_bar, alpha]
            compute_physics: Whether to compute physics loss

        Returns:
            predictions, data_loss, physics_loss
        """
        predictions = self.forward(inputs)

        # Data loss (MSE on parameters)
        data_loss = nn.functional.mse_loss(predictions, targets)

        # Physics loss
        if compute_physics:
            K_target = inputs[:, 0]
            G_target = inputs[:, 1]
            material_props = inputs[:, 2:]  # kappa_inc, mu_inc, kappa_mat, mu_mat, f, R

            physics_loss = self.compute_physics_loss(
                predictions, material_props, K_target, G_target
            )
        else:
            physics_loss = torch.tensor(0.0, device=inputs.device)

        return predictions, data_loss, physics_loss


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def get_device():
    """Get the best available device (GPU if available)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


if __name__ == "__main__":
    # Test model creation
    print("Testing models...")

    device = get_device()

    # Test baseline MLP
    mlp = InverseMLPModel().to(device)
    x = torch.randn(32, 8).to(device)
    out = mlp(x)
    print(f"MLP output shape: {out.shape}")
    print(f"Sample output: k_bar={out[0, 0]:.2f}, lambda_bar={out[0, 1]:.4f}, "
          f"mu_bar={out[0, 2]:.4f}, alpha={out[0, 3]:.2f}")

    # Test PINN
    pinn = PINN().to(device)
    targets = torch.randn(32, 4).to(device)
    preds, data_loss, physics_loss = pinn.forward_with_loss(x, targets, compute_physics=False)
    print(f"\nPINN output shape: {preds.shape}")
    print(f"Data loss: {data_loss.item():.4f}")

    # Count parameters
    n_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
