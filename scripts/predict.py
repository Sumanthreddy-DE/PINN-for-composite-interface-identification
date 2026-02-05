"""
Neural Network Prediction Test Script

This script tests the PINN model by:
1. Computing target K_eff, G_eff using the 3-Layer Interphase model
2. Using the PINN to predict interface parameters (k_bar, lambda_bar, mu_bar, alpha)
3. Verifying the predicted parameters produce the same K_eff, G_eff in Extended Interface model

Usage:
    python scripts/predict.py

Author: System Test Script
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
from typing import Tuple, Dict, List

from src.models.pinn import PINN
from src.physics.extended_interface import compute_effective_properties
from src.physics.three_layer_interphase import compute_3layer_properties


def load_pinn_model(model_path: str = None) -> PINN:
    """Load the trained PINN model (V2 - best model)."""
    if model_path is None:
        # Default to V2 model
        model_path = str(Path(__file__).parent.parent / 'checkpoints' / 'v2' / 'pinn_best.pt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Get norm_params if available
    norm_params = checkpoint.get('norm_params', None)
    
    # Create model with same architecture as V2
    model = PINN(hidden_dims=(256, 256, 256), norm_params=norm_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    return model


def predict_interface_params(
    model: PINN,
    K_target: float,
    G_target: float,
    kappa_inc: float,
    mu_inc: float,
    kappa_mat: float,
    mu_mat: float,
    f: float,
    R: float
) -> Dict[str, float]:
    """
    Use the PINN to predict interface parameters.
    
    Returns:
        Dictionary with k_bar, lambda_bar, mu_bar, alpha
    """
    inputs = torch.tensor([[K_target, G_target, kappa_inc, mu_inc, kappa_mat, mu_mat, f, R]], 
                          dtype=torch.float32)
    
    with torch.no_grad():
        params = model(inputs)
        k_bar, lambda_bar, mu_bar, alpha = params[0].numpy()
    
    return {
        'k_bar': float(k_bar),
        'lambda_bar': float(lambda_bar),
        'mu_bar': float(mu_bar),
        'alpha': float(alpha)
    }


def compute_particle_radius(f: float, sz: float = 0.00278) -> float:
    """Compute particle radius from volume fraction and size parameter."""
    return ((3 * f) / (4 * np.pi)) ** (1/3) * sz


def run_single_test(
    model: PINN,
    kappa_inc: float,
    mu_inc: float,
    kappa_mat: float,
    mu_mat: float,
    f: float,
    phi_1: float = 0.2,
    phi_2: float = 0.3,
    phi_3: float = 0.4,
    interp_weights: Tuple[float, float, float] = None,
    sz: float = 0.00278,
    test_name: str = "Test"
) -> Dict:
    """
    Run a single test case.
    
    1. Compute K_eff, G_eff from 3-Layer model
    2. Predict interface params using PINN
    3. Verify with Extended Interface model
    """
    print(f"\n{'='*60}")
    print(f"  {test_name}")
    print(f"{'='*60}")
    
    # Step 1: Compute target properties from 3-Layer model
    print("\n1. Computing target K_eff, G_eff from 3-Layer Interphase model...")
    K_3layer, G_3layer = compute_3layer_properties(
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        phi_1, phi_2, phi_3, f, interp_weights
    )
    print(f"   Target K_eff = {K_3layer:.6f}")
    print(f"   Target G_eff = {G_3layer:.6f}")
    
    # Compute particle radius
    R = compute_particle_radius(f, sz)
    
    # Step 2: Use PINN to predict interface parameters
    print("\n2. Predicting interface parameters using PINN...")
    params = predict_interface_params(
        model, K_3layer, G_3layer,
        kappa_inc, mu_inc, kappa_mat, mu_mat, f, R
    )
    print(f"   Predicted k̄     = {params['k_bar']:.4f}")
    print(f"   Predicted λ̄     = {params['lambda_bar']:.6f}")
    print(f"   Predicted μ̄     = {params['mu_bar']:.6f}")
    print(f"   Predicted α     = {params['alpha']:.6f}")
    
    # Step 3: Verify with Extended Interface model
    print("\n3. Verifying with Extended Interface model...")
    K_extended, G_extended = compute_effective_properties(
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        params['k_bar'], params['lambda_bar'], params['mu_bar'], params['alpha'],
        f, R
    )
    print(f"   Reconstructed K_eff = {K_extended:.6f}")
    print(f"   Reconstructed G_eff = {G_extended:.6f}")
    
    # Compute errors
    K_error = abs(K_extended - K_3layer) / K_3layer * 100
    G_error = abs(G_extended - G_3layer) / G_3layer * 100
    
    print(f"\n4. Error Analysis:")
    print(f"   K_eff error = {K_error:.2f}%")
    print(f"   G_eff error = {G_error:.2f}%")
    
    # Pass/Fail threshold is 5%
    passed = K_error < 5 and G_error < 5
    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"\n   Status: {status}")
    
    return {
        'test_name': test_name,
        'K_3layer': K_3layer,
        'G_3layer': G_3layer,
        'K_extended': K_extended,
        'G_extended': G_extended,
        'K_error': K_error,
        'G_error': G_error,
        'params': params,
        'passed': passed
    }


def run_all_tests(model: PINN) -> List[Dict]:
    """Run a comprehensive test suite."""
    results = []
    
    # Reference material properties
    kappa_mat = 12 + (2/3) * 8   # = 17.3333
    mu_mat = 8.0
    
    # Test 1: Reference configuration (SR=0.1)
    kappa_inc = 1.2 + (2/3) * 0.8  # = 1.7333
    mu_inc = 0.8
    results.append(run_single_test(
        model, kappa_inc, mu_inc, kappa_mat, mu_mat,
        f=0.3, phi_1=0.2, phi_2=0.3, phi_3=0.4,
        test_name="Reference Config (SR=0.1, f=0.3)"
    ))
    
    # Test 2: Different volume fraction
    results.append(run_single_test(
        model, kappa_inc, mu_inc, kappa_mat, mu_mat,
        f=0.2, phi_1=0.2, phi_2=0.3, phi_3=0.4,
        test_name="Low Volume Fraction (f=0.2)"
    ))
    
    # Test 3: High volume fraction
    results.append(run_single_test(
        model, kappa_inc, mu_inc, kappa_mat, mu_mat,
        f=0.4, phi_1=0.2, phi_2=0.3, phi_3=0.4,
        test_name="High Volume Fraction (f=0.4)"
    ))
    
    # Test 4: Different phi values (thin interphase)
    results.append(run_single_test(
        model, kappa_inc, mu_inc, kappa_mat, mu_mat,
        f=0.3, phi_1=0.6, phi_2=0.7, phi_3=0.8,
        test_name="Thin Interphase (phi=0.6,0.7,0.8)"
    ))
    
    # Test 5: Thick interphase
    results.append(run_single_test(
        model, kappa_inc, mu_inc, kappa_mat, mu_mat,
        f=0.3, phi_1=0.1, phi_2=0.3, phi_3=0.5,
        test_name="Thick Interphase (phi=0.1,0.3,0.5)"
    ))
    
    # Test 6: Different stiffness ratio (SR=0.5)
    SR = 0.5
    kappa_inc_test = kappa_mat * SR
    mu_inc_test = mu_mat * SR
    results.append(run_single_test(
        model, kappa_inc_test, mu_inc_test, kappa_mat, mu_mat,
        f=0.3, phi_1=0.2, phi_2=0.3, phi_3=0.4,
        test_name="Medium Stiffness Ratio (SR=0.5)"
    ))
    
    # Test 7: Stiff particles (SR=2.0)
    SR = 2.0
    kappa_inc_test = kappa_mat * SR
    mu_inc_test = mu_mat * SR
    results.append(run_single_test(
        model, kappa_inc_test, mu_inc_test, kappa_mat, mu_mat,
        f=0.3, phi_1=0.2, phi_2=0.3, phi_3=0.4,
        test_name="Stiff Particles (SR=2.0)"
    ))
    
    return results


def print_summary(results: List[Dict]):
    """Print a summary table of all test results."""
    print("\n" + "=" * 80)
    print("  SUMMARY OF ALL TESTS")
    print("=" * 80)
    
    print(f"\n{'Test Name':<45} {'K_err%':>8} {'G_err%':>8} {'Status':>8}")
    print("-" * 75)
    
    passed_count = 0
    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        if r['passed']:
            passed_count += 1
        print(f"{r['test_name']:<45} {r['K_error']:>7.2f}% {r['G_error']:>7.2f}% {status:>8}")
    
    print("-" * 75)
    print(f"\nTotal: {passed_count}/{len(results)} tests passed ({100*passed_count/len(results):.1f}%)")
    
    # Statistics
    K_errors = [r['K_error'] for r in results]
    G_errors = [r['G_error'] for r in results]
    
    print(f"\nK_eff Error: mean={np.mean(K_errors):.2f}%, max={np.max(K_errors):.2f}%")
    print(f"G_eff Error: mean={np.mean(G_errors):.2f}%, max={np.max(G_errors):.2f}%")


def main():
    """Main function to run the test suite."""
    print("=" * 80)
    print("  NEURAL NETWORK PREDICTION TEST")
    print("  Testing PINN predictions against 3-Layer Interphase model")
    print("=" * 80)
    
    try:
        # Load the model
        model = load_pinn_model()
        
        # Run all tests
        results = run_all_tests(model)
        
        # Print summary
        print_summary(results)
        
        return results
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure the V2 model is trained and saved at:")
        print("  checkpoints/v2/pinn_best.pt")
        return None


if __name__ == "__main__":
    main()
