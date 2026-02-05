"""
PINN Demonstration Test Script

Tests the trained PINN model with custom parameters:
- kappa_inc = 15.0, mu_inc = 5.0 (inclusion)
- kappa_mat = 17.33, mu_mat = 8.0 (matrix)
- f = 0.3 (volume fraction)
- phi_1=0.2, phi_2=0.3, phi_3=0.4 (interphase geometry)

Workflow:
1. Compute K_eff, G_eff from 3-Layer Interphase model
2. Use PINN to predict interface parameters (k_bar, lambda_bar, mu_bar, alpha)
3. Verify predicted parameters produce same K_eff, G_eff in Extended Interface model

Usage:
    python scripts/demo.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch

from src.models.pinn import PINN
from src.physics.extended_interface import compute_effective_properties
from src.physics.three_layer_interphase import compute_3layer_properties


def main():
    print("=" * 70)
    print("  PINN DEMONSTRATION TEST")
    print("  Testing Neural Network Predictions for Composite Homogenization")
    print("=" * 70)

    # =========================================================================
    # USER-SPECIFIED PARAMETERS
    # =========================================================================
    # Inclusion (particle) properties
    kappa_inc = 15.0   # Bulk modulus
    mu_inc = 5.0       # Shear modulus

    # Matrix properties
    kappa_mat = 17.33  # Bulk modulus
    mu_mat = 8.0       # Shear modulus

    # Geometry
    f = 0.3            # Volume fraction
    phi_1 = 0.2        # Layer 1 volume ratio
    phi_2 = 0.3        # Layer 2 volume ratio
    phi_3 = 0.4        # Layer 3 volume ratio
    sz = 0.00278       # Size parameter

    # Compute particle radius
    R = ((3 * f) / (4 * np.pi)) ** (1/3) * sz

    print("\nInput Parameters:")
    print("-" * 70)
    print(f"  Inclusion:  kappa_inc = {kappa_inc}, mu_inc = {mu_inc}")
    print(f"  Matrix:     kappa_mat = {kappa_mat}, mu_mat = {mu_mat}")
    print(f"  Geometry:   f = {f}, R = {R:.6e}")
    print(f"  Interphase: phi_1 = {phi_1}, phi_2 = {phi_2}, phi_3 = {phi_3}")
    print(f"  Stiffness Ratio (SR): {kappa_inc/kappa_mat:.2f}")

    # =========================================================================
    # STEP 1: Compute target properties from 3-Layer Interphase model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Computing target K_eff, G_eff from 3-Layer Interphase model")
    print("=" * 70)

    K_target, G_target = compute_3layer_properties(
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        phi_1, phi_2, phi_3, f
    )

    print(f"\n  Target K_eff = {K_target:.6f}")
    print(f"  Target G_eff = {G_target:.6f}")

    # =========================================================================
    # STEP 2: Load and use PINN to predict interface parameters
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Predicting interface parameters using trained PINN")
    print("=" * 70)

    # Load model
    model_path = str(Path(__file__).parent.parent / 'checkpoints' / 'v2' / 'pinn_best.pt')
    print(f"\n  Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    norm_params = checkpoint.get('norm_params', None)

    model = PINN(hidden_dims=(256, 256, 256), norm_params=norm_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("  Model loaded successfully!")

    # Predict interface parameters
    inputs = torch.tensor([[K_target, G_target, kappa_inc, mu_inc, kappa_mat, mu_mat, f, R]],
                          dtype=torch.float32)

    with torch.no_grad():
        params = model(inputs)
        k_bar, lambda_bar, mu_bar, alpha = params[0].numpy()

    print("\n  Predicted Interface Parameters:")
    print(f"    k_bar      = {k_bar:.4f}     (normal stiffness [N/m^3])")
    print(f"    lambda_bar = {lambda_bar:.6f}  (tangential Lame parameter [N/m])")
    print(f"    mu_bar     = {mu_bar:.6f}  (tangential shear modulus [N/m])")
    print(f"    alpha      = {alpha:.6f}  (interface position: 0=cohesive, 1=elastic)")

    # =========================================================================
    # STEP 3: Verify with Extended Interface model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Verifying with Extended Interface (Mori-Tanaka) model")
    print("=" * 70)

    K_recon, G_recon = compute_effective_properties(
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        k_bar, lambda_bar, mu_bar, alpha,
        f, R
    )

    print(f"\n  Reconstructed K_eff = {K_recon:.6f}")
    print(f"  Reconstructed G_eff = {G_recon:.6f}")

    # =========================================================================
    # STEP 4: Error Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Error Analysis")
    print("=" * 70)

    K_error = abs(K_recon - K_target) / K_target * 100
    G_error = abs(G_recon - G_target) / G_target * 100

    print(f"\n  K_eff: Target = {K_target:.6f}, Reconstructed = {K_recon:.6f}")
    print(f"         Error = {K_error:.2f}%")

    print(f"\n  G_eff: Target = {G_target:.6f}, Reconstructed = {G_recon:.6f}")
    print(f"         Error = {G_error:.2f}%")

    # Pass/Fail threshold
    passed = K_error < 5 and G_error < 5

    print("\n" + "=" * 70)
    if passed:
        print("  RESULT: PASS")
        print("  Both K_eff and G_eff errors are below 5% threshold")
    else:
        print("  RESULT: FAIL")
        print("  One or both errors exceed 5% threshold")
    print("=" * 70)

    # Summary table
    print("\n\nSUMMARY TABLE")
    print("-" * 70)
    print(f"{'Property':<20} {'3-Layer Model':<18} {'Extended Interface':<18} {'Error':<10}")
    print("-" * 70)
    print(f"{'K_eff':<20} {K_target:<18.6f} {K_recon:<18.6f} {K_error:.2f}%")
    print(f"{'G_eff':<20} {G_target:<18.6f} {G_recon:<18.6f} {G_error:.2f}%")
    print("-" * 70)

    return {
        'K_target': K_target,
        'G_target': G_target,
        'K_recon': K_recon,
        'G_recon': G_recon,
        'K_error': K_error,
        'G_error': G_error,
        'k_bar': k_bar,
        'lambda_bar': lambda_bar,
        'mu_bar': mu_bar,
        'alpha': alpha,
        'passed': passed
    }


if __name__ == "__main__":
    main()
