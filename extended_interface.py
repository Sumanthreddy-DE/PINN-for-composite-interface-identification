"""
Extended General Interface Model for Particle-Reinforced Composites (3D/CSA)

Python port of Mori_Tanaka_Extended_3D.m
Based on Firooz et al. (2022) - Extended general interfaces: Mori-Tanaka homogenization

This module computes effective properties (K_eff, G_eff) given:
- Inclusion properties (kappa_inc, mu_inc)
- Matrix properties (kappa_mat, mu_mat)
- Interface parameters (k_bar, lambda_bar, mu_bar, alpha)
- Geometry (volume fraction f, particle radius R)

Author: Sumanth Reddy Settipalli
"""

import numpy as np
from typing import Tuple, Dict, Union
import torch


def compute_effective_properties(
    kappa_inc: float,
    mu_inc: float,
    kappa_mat: float,
    mu_mat: float,
    k_bar: float,
    lambda_bar: float,
    mu_bar: float,
    alpha: float,
    f: float,
    R: float
) -> Tuple[float, float]:
    """
    Compute effective bulk and shear moduli using Extended Mori-Tanaka method.

    Args:
        kappa_inc: Inclusion bulk modulus
        mu_inc: Inclusion shear modulus
        kappa_mat: Matrix bulk modulus
        mu_mat: Matrix shear modulus
        k_bar: Interface normal stiffness [N/m³]
        lambda_bar: Interface tangential Lame parameter [N/m]
        mu_bar: Interface tangential shear modulus [N/m]
        alpha: Interface position parameter [0=cohesive, 1=elastic]
        f: Volume fraction of inclusions
        R: Particle radius

    Returns:
        K_eff: Effective bulk modulus
        G_eff: Effective shear modulus
    """
    # Solve BVP 1: Volumetric (Eq. 99)
    T_b, H_b = solve_volumetric_bvp(
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        k_bar, lambda_bar, mu_bar, alpha, R
    )

    # Solve BVP 2: Deviatoric (Eq. 110)
    T_s, H_s = solve_deviatoric_bvp(
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        k_bar, lambda_bar, mu_bar, alpha, R
    )

    # Mori-Tanaka concentration factors
    A_b = 1.0 / ((1 - f) + f * T_b)
    A_s = 1.0 / ((1 - f) + f * T_s)

    # Effective moduli
    K_eff = (1/3) * ((1 - f) * 3 * kappa_mat + f * H_b) * A_b
    G_eff = (1/2) * ((1 - f) * 2 * mu_mat + f * H_s) * A_s

    return K_eff, G_eff


def solve_volumetric_bvp(
    kappa_inc: float,
    mu_inc: float,
    kappa_mat: float,
    mu_mat: float,
    k_bar: float,
    lambda_bar: float,
    mu_bar: float,
    alpha: float,
    R: float
) -> Tuple[float, float]:
    """
    Solve Volumetric BVP (Eq. 99) for T_b and H_b.

    Returns:
        T_b: Equivalent volumetric strain concentration
        H_b: Equivalent volumetric stress concentration
    """
    S_bar = lambda_bar + mu_bar

    # Construct 2x2 system A_vol * X = b_vol
    A_vol = np.array([
        [R + (3 * alpha * kappa_inc) / k_bar,
         -R - (4 * (1 - alpha) * mu_mat) / k_bar],
        [3 * kappa_inc + (4 * (1 - alpha) * S_bar) / R,
         4 * mu_mat + (4 * alpha * S_bar) / R]
    ])

    b_vol = np.array([
        R - (3 * (1 - alpha) * kappa_mat) / k_bar,
        3 * kappa_mat - (4 * alpha * S_bar) / R
    ])

    # Solve system
    X_vol = np.linalg.solve(A_vol, b_vol)
    X1_vol, X4_vol = X_vol[0], X_vol[1]

    # Extract interaction terms (Eq. 101)
    T_b = 1 + X4_vol
    H_b = 3 * kappa_mat - 4 * mu_mat * X4_vol

    return T_b, H_b


def solve_deviatoric_bvp(
    kappa_inc: float,
    mu_inc: float,
    kappa_mat: float,
    mu_mat: float,
    k_bar: float,
    lambda_bar: float,
    mu_bar: float,
    alpha: float,
    R: float
) -> Tuple[float, float]:
    """
    Solve Deviatoric BVP (Eq. 110) for T_s and H_s.

    Returns:
        T_s: Equivalent deviatoric strain concentration
        H_s: Equivalent deviatoric stress concentration
    """
    S_bar = lambda_bar + mu_bar
    k_t = k_bar  # Use same stiffness for tangential (spherical symmetry)

    # Construct 4x4 system A_dev * X = b_dev
    A_dev = np.zeros((4, 4))

    # Row 1
    A_dev[0, 0] = 1 + (2 * alpha * mu_inc) / (k_bar * R)
    A_dev[0, 1] = 2 - 3 * (kappa_inc / mu_inc) + alpha * (3 * kappa_inc - 2 * mu_inc) / (k_bar * R)
    A_dev[0, 2] = -3 - (24 * (1 - alpha) * mu_mat) / (k_bar * R)
    A_dev[0, 3] = -3 - 3 * (kappa_mat / mu_mat) - (1 - alpha) * (18 * kappa_mat + 8 * mu_mat) / (k_bar * R)

    # Row 2
    A_dev[1, 0] = 1 + (2 * alpha * mu_inc) / (k_t * R)
    A_dev[1, 1] = -(11/3) - 5 * (kappa_inc / mu_inc) - 2 * alpha * (8 * kappa_inc + 5 * mu_inc / 3) / (k_t * R)
    A_dev[1, 2] = 2 + (16 * (1 - alpha) * mu_mat) / (k_t * R)
    A_dev[1, 3] = -2 + (6 * (1 - alpha) * kappa_mat) / (k_t * R)

    # Row 3
    A_dev[2, 0] = -2 * mu_inc + (2 * (1 - alpha) * S_bar) / R
    A_dev[2, 1] = 2 * mu_inc - 3 * kappa_inc - 2 * (1 - alpha) * S_bar * (9 * kappa_inc + 15 * mu_inc) / (mu_inc * R)
    A_dev[2, 2] = -24 * mu_mat - (24 * alpha * S_bar) / R
    A_dev[2, 3] = -18 * kappa_mat - 8 * mu_mat - (12 * alpha * kappa_mat * S_bar) / (mu_mat * R)

    # Row 4
    A_dev[3, 0] = -2 * mu_inc - 2 * (1 - alpha) * (S_bar + 2 * mu_bar) / R
    A_dev[3, 1] = (16 * kappa_inc + (10/3) * mu_inc +
                  2 * (1 - alpha) * (kappa_inc * (27 * S_bar + 30 * mu_bar) +
                                     mu_inc * (45 * S_bar + 22 * mu_bar)) / (3 * mu_inc * R))
    A_dev[3, 2] = 16 * mu_mat + 8 * alpha * (3 * S_bar + mu_bar) / R
    A_dev[3, 3] = 6 * kappa_mat + 2 * (6 * alpha * kappa_mat * S_bar - 4 * alpha * mu_mat * mu_bar) / (mu_mat * R)

    # Construct RHS
    b_dev = np.array([
        1 - (2 * (1 - alpha) * mu_mat) / (k_bar * R),
        1 - (2 * (1 - alpha) * mu_mat) / (k_t * R),
        -2 * mu_mat - (2 * alpha * S_bar) / R,
        -2 * mu_mat + 2 * alpha * (S_bar + 2 * mu_bar) / R
    ])

    # Solve system
    X_dev = np.linalg.solve(A_dev, b_dev)
    X1_dev, X2_dev, _, X8_dev = X_dev

    # Extract interaction terms (Eq. 113)
    T_s = 0.2 * (5 + 6 * X8_dev * (2 + kappa_mat / mu_mat))
    H_s = 0.2 * (10 * mu_mat - 2 * X8_dev * (9 * kappa_mat + 8 * mu_mat))

    return T_s, H_s


# ============================================================================
# PyTorch-compatible versions for differentiable physics loss
# ============================================================================

def compute_effective_properties_torch(
    kappa_inc: torch.Tensor,
    mu_inc: torch.Tensor,
    kappa_mat: torch.Tensor,
    mu_mat: torch.Tensor,
    k_bar: torch.Tensor,
    lambda_bar: torch.Tensor,
    mu_bar: torch.Tensor,
    alpha: torch.Tensor,
    f: torch.Tensor,
    R: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch-differentiable version of compute_effective_properties.
    Supports batched inputs of shape (batch_size,).
    """
    T_b, H_b = solve_volumetric_bvp_torch(
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        k_bar, lambda_bar, mu_bar, alpha, R
    )

    T_s, H_s = solve_deviatoric_bvp_torch(
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        k_bar, lambda_bar, mu_bar, alpha, R
    )

    # Mori-Tanaka concentration factors
    A_b = 1.0 / ((1 - f) + f * T_b)
    A_s = 1.0 / ((1 - f) + f * T_s)

    # Effective moduli
    K_eff = (1/3) * ((1 - f) * 3 * kappa_mat + f * H_b) * A_b
    G_eff = (1/2) * ((1 - f) * 2 * mu_mat + f * H_s) * A_s

    return K_eff, G_eff


def solve_volumetric_bvp_torch(
    kappa_inc: torch.Tensor,
    mu_inc: torch.Tensor,
    kappa_mat: torch.Tensor,
    mu_mat: torch.Tensor,
    k_bar: torch.Tensor,
    lambda_bar: torch.Tensor,
    mu_bar: torch.Tensor,
    alpha: torch.Tensor,
    R: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch-differentiable volumetric BVP solver.
    Uses Cramer's rule for 2x2 system to maintain differentiability.
    """
    S_bar = lambda_bar + mu_bar

    # 2x2 system coefficients
    a11 = R + (3 * alpha * kappa_inc) / k_bar
    a12 = -R - (4 * (1 - alpha) * mu_mat) / k_bar
    a21 = 3 * kappa_inc + (4 * (1 - alpha) * S_bar) / R
    a22 = 4 * mu_mat + (4 * alpha * S_bar) / R

    b1 = R - (3 * (1 - alpha) * kappa_mat) / k_bar
    b2 = 3 * kappa_mat - (4 * alpha * S_bar) / R

    # Cramer's rule
    det = a11 * a22 - a12 * a21
    X1_vol = (b1 * a22 - b2 * a12) / det
    X4_vol = (a11 * b2 - a21 * b1) / det

    # Extract interaction terms
    T_b = 1 + X4_vol
    H_b = 3 * kappa_mat - 4 * mu_mat * X4_vol

    return T_b, H_b


def solve_deviatoric_bvp_torch(
    kappa_inc: torch.Tensor,
    mu_inc: torch.Tensor,
    kappa_mat: torch.Tensor,
    mu_mat: torch.Tensor,
    k_bar: torch.Tensor,
    lambda_bar: torch.Tensor,
    mu_bar: torch.Tensor,
    alpha: torch.Tensor,
    R: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch-differentiable deviatoric BVP solver.
    Uses torch.linalg.solve for batched 4x4 systems.
    """
    batch_size = k_bar.shape[0] if k_bar.dim() > 0 else 1
    device = k_bar.device
    dtype = k_bar.dtype

    S_bar = lambda_bar + mu_bar
    k_t = k_bar  # Same for spherical

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
    X8_dev = X_dev[:, 3]

    # Extract interaction terms
    T_s = 0.2 * (5 + 6 * X8_dev * (2 + kappa_mat / mu_mat))
    H_s = 0.2 * (10 * mu_mat - 2 * X8_dev * (9 * kappa_mat + 8 * mu_mat))

    return T_s, H_s


# ============================================================================
# Validation function
# ============================================================================

def validate_against_matlab():
    """
    Validate Python implementation against MATLAB reference values.
    Uses the same parameters as Model_Results.md
    """
    # Reference parameters from Model_Results.md
    kappa_mat = 17.33
    mu_mat = 8.0
    kappa_inc = 1.73
    mu_inc = 0.8

    f = 0.3
    sz = 0.0067  # From Model_Results.md
    R = ((3 * f) / (4 * np.pi)) ** (1/3) * sz

    k_bar = 10.0
    lambda_bar = 0.1
    mu_bar = 0.1
    alpha = 0.0  # From Model_Results.md (cohesive-like)

    # Compute
    K_eff, G_eff = compute_effective_properties(
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        k_bar, lambda_bar, mu_bar, alpha, f, R
    )

    print("Extended Interface Model Validation")
    print("=" * 50)
    print(f"Inputs:")
    print(f"  Matrix: kappa={kappa_mat:.4f}, mu={mu_mat:.4f}")
    print(f"  Inclusion: kappa={kappa_inc:.4f}, mu={mu_inc:.4f}")
    print(f"  Interface: k_bar={k_bar}, lambda_bar={lambda_bar}, mu_bar={mu_bar}, alpha={alpha}")
    print(f"  Geometry: f={f}, R={R:.6e}")
    print()
    print(f"Results:")
    print(f"  K_eff = {K_eff:.15e}")
    print(f"  G_eff = {G_eff:.15e}")
    print()
    print("Expected (from MATLAB):")
    print("  K_eff ~ 8.17")
    print("  G_eff ~ 4.42")

    return K_eff, G_eff


if __name__ == "__main__":
    validate_against_matlab()
