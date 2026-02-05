"""
Physics Loss Functions for PINN Training

Implements the BVP equations from Firooz et al. (2022) as differentiable
PyTorch loss functions.

The physics loss ensures predicted interface parameters satisfy the
governing equations and produce correct effective properties.

Author: Sumanth Reddy Settipalli
"""

import torch
import torch.nn as nn
from typing import Tuple


def compute_volumetric_residual(
    kappa_inc: torch.Tensor,
    mu_inc: torch.Tensor,
    kappa_mat: torch.Tensor,
    mu_mat: torch.Tensor,
    k_bar: torch.Tensor,
    lambda_bar: torch.Tensor,
    mu_bar: torch.Tensor,
    alpha: torch.Tensor,
    R: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute volumetric BVP solution and residual (Eq. 99).

    Returns:
        T_b: Equivalent volumetric strain concentration
        H_b: Equivalent volumetric stress concentration
        residual: BVP residual (should be near zero if params are consistent)
    """
    S_bar = lambda_bar + mu_bar

    # 2x2 system coefficients
    a11 = R + (3 * alpha * kappa_inc) / k_bar
    a12 = -R - (4 * (1 - alpha) * mu_mat) / k_bar
    a21 = 3 * kappa_inc + (4 * (1 - alpha) * S_bar) / R
    a22 = 4 * mu_mat + (4 * alpha * S_bar) / R

    b1 = R - (3 * (1 - alpha) * kappa_mat) / k_bar
    b2 = 3 * kappa_mat - (4 * alpha * S_bar) / R

    # Cramer's rule for 2x2 system
    det = a11 * a22 - a12 * a21
    X1_vol = (b1 * a22 - b2 * a12) / det
    X4_vol = (a11 * b2 - a21 * b1) / det

    # Verify solution (residual should be zero)
    res1 = a11 * X1_vol + a12 * X4_vol - b1
    res2 = a21 * X1_vol + a22 * X4_vol - b2
    residual = res1**2 + res2**2

    # Extract interaction terms (Eq. 101)
    T_b = 1 + X4_vol
    H_b = 3 * kappa_mat - 4 * mu_mat * X4_vol

    return T_b, H_b, residual


def compute_deviatoric_residual(
    kappa_inc: torch.Tensor,
    mu_inc: torch.Tensor,
    kappa_mat: torch.Tensor,
    mu_mat: torch.Tensor,
    k_bar: torch.Tensor,
    lambda_bar: torch.Tensor,
    mu_bar: torch.Tensor,
    alpha: torch.Tensor,
    R: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute deviatoric BVP solution and residual (Eq. 110).

    Returns:
        T_s: Equivalent deviatoric strain concentration
        H_s: Equivalent deviatoric stress concentration
        residual: BVP residual
    """
    batch_size = k_bar.shape[0] if k_bar.dim() > 0 else 1
    device = k_bar.device
    dtype = k_bar.dtype

    S_bar = lambda_bar + mu_bar
    k_t = k_bar  # Same for spherical symmetry

    # Build batch of 4x4 matrices
    A_dev = torch.zeros(batch_size, 4, 4, device=device, dtype=dtype)
    b_dev = torch.zeros(batch_size, 4, device=device, dtype=dtype)

    # Row 1
    A_dev[:, 0, 0] = 1 + (2 * alpha * mu_inc) / (k_bar * R)
    A_dev[:, 0, 1] = 2 - 3 * (kappa_inc / mu_inc) + alpha * (3 * kappa_inc - 2 * mu_inc) / (k_bar * R)
    A_dev[:, 0, 2] = -3 - (24 * (1 - alpha) * mu_mat) / (k_bar * R)
    A_dev[:, 0, 3] = -3 - 3 * (kappa_mat / mu_mat) - (1 - alpha) * (18 * kappa_mat + 8 * mu_mat) / (k_bar * R)

    # Row 2
    A_dev[:, 1, 0] = 1 + (2 * alpha * mu_inc) / (k_t * R)
    A_dev[:, 1, 1] = -(11/3) - 5 * (kappa_inc / mu_inc) - 2 * alpha * (8 * kappa_inc + 5 * mu_inc / 3) / (k_t * R)
    A_dev[:, 1, 2] = 2 + (16 * (1 - alpha) * mu_mat) / (k_t * R)
    A_dev[:, 1, 3] = -2 + (6 * (1 - alpha) * kappa_mat) / (k_t * R)

    # Row 3
    A_dev[:, 2, 0] = -2 * mu_inc + (2 * (1 - alpha) * S_bar) / R
    A_dev[:, 2, 1] = 2 * mu_inc - 3 * kappa_inc - 2 * (1 - alpha) * S_bar * (9 * kappa_inc + 15 * mu_inc) / (mu_inc * R)
    A_dev[:, 2, 2] = -24 * mu_mat - (24 * alpha * S_bar) / R
    A_dev[:, 2, 3] = -18 * kappa_mat - 8 * mu_mat - (12 * alpha * kappa_mat * S_bar) / (mu_mat * R)

    # Row 4
    A_dev[:, 3, 0] = -2 * mu_inc - 2 * (1 - alpha) * (S_bar + 2 * mu_bar) / R
    A_dev[:, 3, 1] = (16 * kappa_inc + (10/3) * mu_inc +
                     2 * (1 - alpha) * (kappa_inc * (27 * S_bar + 30 * mu_bar) +
                                        mu_inc * (45 * S_bar + 22 * mu_bar)) / (3 * mu_inc * R))
    A_dev[:, 3, 2] = 16 * mu_mat + 8 * alpha * (3 * S_bar + mu_bar) / R
    A_dev[:, 3, 3] = 6 * kappa_mat + 2 * (6 * alpha * kappa_mat * S_bar - 4 * alpha * mu_mat * mu_bar) / (mu_mat * R)

    # RHS
    b_dev[:, 0] = 1 - (2 * (1 - alpha) * mu_mat) / (k_bar * R)
    b_dev[:, 1] = 1 - (2 * (1 - alpha) * mu_mat) / (k_t * R)
    b_dev[:, 2] = -2 * mu_mat - (2 * alpha * S_bar) / R
    b_dev[:, 3] = -2 * mu_mat + 2 * alpha * (S_bar + 2 * mu_bar) / R

    # Solve batched system
    X_dev = torch.linalg.solve(A_dev, b_dev)

    # Compute residual: ||Ax - b||^2
    residual = ((A_dev @ X_dev.unsqueeze(-1)).squeeze(-1) - b_dev).pow(2).sum(dim=1)

    X8_dev = X_dev[:, 3]

    # Extract interaction terms (Eq. 113)
    T_s = 0.2 * (5 + 6 * X8_dev * (2 + kappa_mat / mu_mat))
    H_s = 0.2 * (10 * mu_mat - 2 * X8_dev * (9 * kappa_mat + 8 * mu_mat))

    return T_s, H_s, residual


def physics_loss(
    predicted_params: torch.Tensor,
    K_target: torch.Tensor,
    G_target: torch.Tensor,
    kappa_inc: torch.Tensor,
    mu_inc: torch.Tensor,
    kappa_mat: torch.Tensor,
    mu_mat: torch.Tensor,
    f: torch.Tensor,
    R: torch.Tensor,
    include_bvp_residual: bool = True,
    K_weight: float = 1.0,
    G_weight: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute physics-based loss for PINN training.

    The loss has two components:
    1. Effective property matching: |K_pred - K_target|^2 + |G_pred - G_target|^2
    2. BVP residuals: Ensures solution satisfies governing equations

    Args:
        predicted_params: (batch, 4) - [k_bar, lambda_bar, mu_bar, alpha]
        K_target, G_target: Target effective properties
        kappa_inc, mu_inc: Inclusion properties
        kappa_mat, mu_mat: Matrix properties
        f: Volume fraction
        R: Particle radius
        include_bvp_residual: Whether to include BVP residual in loss

    Returns:
        total_loss: Combined physics loss
        property_loss: Effective property matching loss
        bvp_loss: BVP residual loss
    """
    k_bar = predicted_params[:, 0]
    lambda_bar = predicted_params[:, 1]
    mu_bar = predicted_params[:, 2]
    alpha = predicted_params[:, 3]

    # Solve BVPs
    T_b, H_b, vol_residual = compute_volumetric_residual(
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        k_bar, lambda_bar, mu_bar, alpha, R
    )

    T_s, H_s, dev_residual = compute_deviatoric_residual(
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        k_bar, lambda_bar, mu_bar, alpha, R
    )

    # Mori-Tanaka: compute effective properties
    A_b = 1.0 / ((1 - f) + f * T_b)
    A_s = 1.0 / ((1 - f) + f * T_s)

    K_pred = (1/3) * ((1 - f) * 3 * kappa_mat + f * H_b) * A_b
    G_pred = (1/2) * ((1 - f) * 2 * mu_mat + f * H_s) * A_s

    # Property matching loss (relative error for scale invariance)
    # Weighted to prioritize K_eff accuracy (typically has higher error)
    K_rel_error = ((K_pred - K_target) / (K_target.abs() + 1e-8)).pow(2)
    G_rel_error = ((G_pred - G_target) / (G_target.abs() + 1e-8)).pow(2)
    property_loss = (K_weight * K_rel_error + G_weight * G_rel_error).mean() / (K_weight + G_weight)

    # BVP residual loss (optional regularization)
    if include_bvp_residual:
        bvp_loss = (vol_residual + dev_residual).mean()
    else:
        bvp_loss = torch.tensor(0.0, device=predicted_params.device)

    total_loss = property_loss + 0.01 * bvp_loss

    return total_loss, property_loss, bvp_loss


class PhysicsLoss(nn.Module):
    """
    Physics loss module for use in training loop.

    Wraps physics_loss function as nn.Module for cleaner integration.
    """

    def __init__(self, include_bvp_residual: bool = True, property_weight: float = 1.0,
                 K_weight: float = 1.0, G_weight: float = 1.0):
        super().__init__()
        self.include_bvp_residual = include_bvp_residual
        self.property_weight = property_weight
        self.K_weight = K_weight
        self.G_weight = G_weight

    def forward(
        self,
        predicted_params: torch.Tensor,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute physics loss.

        Args:
            predicted_params: (batch, 4) - [k_bar, lambda_bar, mu_bar, alpha]
            inputs: (batch, 8) - [K_eff, G_eff, kappa_inc, mu_inc, kappa_mat, mu_mat, f, R]

        Returns:
            total_loss, property_loss, bvp_loss
        """
        K_target = inputs[:, 0]
        G_target = inputs[:, 1]
        kappa_inc = inputs[:, 2]
        mu_inc = inputs[:, 3]
        kappa_mat = inputs[:, 4]
        mu_mat = inputs[:, 5]
        f = inputs[:, 6]
        R = inputs[:, 7]

        return physics_loss(
            predicted_params,
            K_target, G_target,
            kappa_inc, mu_inc, kappa_mat, mu_mat,
            f, R,
            include_bvp_residual=self.include_bvp_residual,
            K_weight=self.K_weight,
            G_weight=self.G_weight
        )


if __name__ == "__main__":
    # Test physics loss computation
    print("Testing physics loss...")

    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test data
    predicted_params = torch.rand(batch_size, 4, device=device)
    predicted_params[:, 0] = predicted_params[:, 0] * 100 + 1  # k_bar in [1, 101]
    predicted_params[:, 1] = predicted_params[:, 1] * 5 + 0.01  # lambda_bar
    predicted_params[:, 2] = predicted_params[:, 2] * 5 + 0.01  # mu_bar
    # alpha already in [0, 1]

    # Material properties
    kappa_inc = torch.ones(batch_size, device=device) * 1.73
    mu_inc = torch.ones(batch_size, device=device) * 0.8
    kappa_mat = torch.ones(batch_size, device=device) * 17.33
    mu_mat = torch.ones(batch_size, device=device) * 8.0
    f = torch.ones(batch_size, device=device) * 0.3
    R = torch.ones(batch_size, device=device) * 0.00278

    K_target = torch.ones(batch_size, device=device) * 8.17
    G_target = torch.ones(batch_size, device=device) * 4.42

    # Compute loss
    total_loss, prop_loss, bvp_loss = physics_loss(
        predicted_params,
        K_target, G_target,
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        f, R
    )

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Property loss: {prop_loss.item():.4f}")
    print(f"BVP loss: {bvp_loss.item():.6f}")

    # Test module version
    physics_module = PhysicsLoss()
    inputs = torch.stack([
        K_target, G_target, kappa_inc, mu_inc, kappa_mat, mu_mat, f, R
    ], dim=1)

    total_loss2, _, _ = physics_module(predicted_params, inputs)
    print(f"\nModule version loss: {total_loss2.item():.4f}")

    # Test gradient flow
    predicted_params.requires_grad = True
    total_loss, _, _ = physics_loss(
        predicted_params,
        K_target, G_target,
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        f, R
    )
    total_loss.backward()
    print(f"\nGradient computed: {predicted_params.grad is not None}")
    print(f"Gradient norm: {predicted_params.grad.norm().item():.4f}")
