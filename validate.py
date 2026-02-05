"""
Validation Script: Test PINN Against 3-Layer Interphase Model

This script validates whether the PINN can find interface parameters
that produce the same K_eff, G_eff as the 3-Layer Interphase model.

Workflow:
1. Define 3-Layer model configurations
2. Compute K_eff, G_eff from 3-Layer model (or use reference values)
3. Feed to trained PINN to get interface parameters
4. Use interface params in Extended Interface model
5. Compare K_eff, G_eff

Author: Sumanth Reddy Settipalli
"""

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

from models import PINN, InverseMLPModel, get_device
from extended_interface import compute_effective_properties


def load_model(checkpoint_path: str, model_type: str = 'pinn'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    norm_params = checkpoint.get('norm_params', None)
    args = checkpoint.get('args', {})

    hidden_dims = args.get('hidden_dims', [128, 128, 128])

    if model_type == 'pinn':
        model = PINN(
            hidden_dims=tuple(hidden_dims),
            norm_params=norm_params
        )
    else:
        model = InverseMLPModel(
            hidden_dims=tuple(hidden_dims),
            norm_params=norm_params
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def validate_3layer_reference():
    """
    Validate against the reference 3-Layer model values from Model_Results.md

    Reference:
    - 3-Layer CSA: K_eff=14.91, G_eff=8.02
    - Material: Matrix (K=17.33, mu=8), Particle (K=1.73, mu=0.8), f=0.3
    """
    print("=" * 60)
    print("Validation Against 3-Layer Interphase Model")
    print("=" * 60)

    # Reference values from Model_Results.md
    reference = {
        'K_3layer': 14.91,
        'G_3layer': 8.02,
        'kappa_mat': 17.33,
        'mu_mat': 8.0,
        'kappa_inc': 1.73,
        'mu_inc': 0.8,
        'f': 0.3,
    }

    # Size parameter (use same as Extended Interface reference for consistency)
    sz = 0.0067
    R = ((3 * reference['f']) / (4 * np.pi)) ** (1/3) * sz

    print("\nTarget (from 3-Layer Interphase):")
    print(f"  K_eff = {reference['K_3layer']:.2f}")
    print(f"  G_eff = {reference['G_3layer']:.2f}")

    print("\nMaterial Properties:")
    print(f"  Matrix: kappa={reference['kappa_mat']:.2f}, mu={reference['mu_mat']:.2f}")
    print(f"  Inclusion: kappa={reference['kappa_inc']:.2f}, mu={reference['mu_inc']:.2f}")
    print(f"  Volume fraction: f={reference['f']:.2f}")
    print(f"  Particle radius: R={R:.6e}")

    # Load trained models
    results_dir = Path(__file__).parent

    models_to_test = []

    pinn_path = results_dir / 'results_pinn' / 'pinn_best.pt'
    if pinn_path.exists():
        models_to_test.append(('PINN', pinn_path, 'pinn'))

    mlp_path = results_dir / 'results' / 'mlp_best.pt'
    if mlp_path.exists():
        models_to_test.append(('MLP', mlp_path, 'mlp'))

    if not models_to_test:
        print("\nNo trained models found. Run train.py first.")
        print("Expected paths:")
        print(f"  - {pinn_path}")
        print(f"  - {mlp_path}")
        return

    # Test each model
    results = []

    for name, path, model_type in models_to_test:
        print(f"\n{'-' * 40}")
        print(f"Testing {name} Model")
        print(f"{'-' * 40}")

        model = load_model(str(path), model_type)

        # Prepare input
        inputs = torch.tensor([[
            reference['K_3layer'],
            reference['G_3layer'],
            reference['kappa_inc'],
            reference['mu_inc'],
            reference['kappa_mat'],
            reference['mu_mat'],
            reference['f'],
            R
        ]], dtype=torch.float32)

        # Predict interface parameters
        with torch.no_grad():
            params = model(inputs)[0].numpy()

        k_bar, lambda_bar, mu_bar, alpha = params

        print(f"\nPredicted Interface Parameters:")
        print(f"  k_bar = {k_bar:.4f}")
        print(f"  lambda_bar = {lambda_bar:.4f}")
        print(f"  mu_bar = {mu_bar:.4f}")
        print(f"  alpha = {alpha:.4f}")

        # Verify with Extended Interface model
        K_pred, G_pred = compute_effective_properties(
            reference['kappa_inc'], reference['mu_inc'],
            reference['kappa_mat'], reference['mu_mat'],
            k_bar, lambda_bar, mu_bar, alpha,
            reference['f'], R
        )

        K_error = abs(K_pred - reference['K_3layer']) / reference['K_3layer'] * 100
        G_error = abs(G_pred - reference['G_3layer']) / reference['G_3layer'] * 100

        print(f"\nReconstructed Effective Properties:")
        print(f"  K_eff = {K_pred:.4f} (target: {reference['K_3layer']:.2f}, error: {K_error:.2f}%)")
        print(f"  G_eff = {G_pred:.4f} (target: {reference['G_3layer']:.2f}, error: {G_error:.2f}%)")

        results.append({
            'model': name,
            'k_bar': float(k_bar),
            'lambda_bar': float(lambda_bar),
            'mu_bar': float(mu_bar),
            'alpha': float(alpha),
            'K_pred': float(K_pred),
            'G_pred': float(G_pred),
            'K_error': float(K_error),
            'G_error': float(G_error),
        })

    # Summary comparison
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    print(f"\n{'Model':<10} {'K_error':>10} {'G_error':>10}")
    print("-" * 32)
    for r in results:
        print(f"{r['model']:<10} {r['K_error']:>9.2f}% {r['G_error']:>9.2f}%")

    # Save results
    output_path = results_dir / 'validation_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'reference': reference,
            'results': results
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def parameter_sensitivity_analysis():
    """
    Analyze how changes in interface parameters affect K_eff, G_eff.
    Helps understand the inverse problem structure.
    """
    print("\n" + "=" * 60)
    print("Parameter Sensitivity Analysis")
    print("=" * 60)

    # Base parameters
    kappa_mat = 17.33
    mu_mat = 8.0
    kappa_inc = 1.73
    mu_inc = 0.8
    f = 0.3
    sz = 0.0067
    R = ((3 * f) / (4 * np.pi)) ** (1/3) * sz

    # Base interface params
    base_params = {
        'k_bar': 10.0,
        'lambda_bar': 0.1,
        'mu_bar': 0.1,
        'alpha': 0.0,
    }

    # Sweep each parameter
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    param_ranges = {
        'k_bar': np.linspace(1, 500, 50),
        'lambda_bar': np.linspace(0.01, 5, 50),
        'mu_bar': np.linspace(0.01, 5, 50),
        'alpha': np.linspace(0, 1, 50),
    }

    for idx, (param_name, values) in enumerate(param_ranges.items()):
        K_vals = []
        G_vals = []

        for val in values:
            params = base_params.copy()
            params[param_name] = val

            try:
                K, G = compute_effective_properties(
                    kappa_inc, mu_inc, kappa_mat, mu_mat,
                    params['k_bar'], params['lambda_bar'],
                    params['mu_bar'], params['alpha'], f, R
                )
                K_vals.append(K)
                G_vals.append(G)
            except:
                K_vals.append(np.nan)
                G_vals.append(np.nan)

        # Plot K_eff
        axes[0, idx].plot(values, K_vals)
        axes[0, idx].set_xlabel(param_name)
        axes[0, idx].set_ylabel('K_eff')
        axes[0, idx].set_title(f'K_eff vs {param_name}')
        axes[0, idx].axhline(y=14.91, color='r', linestyle='--', label='3-Layer target')
        axes[0, idx].legend()

        # Plot G_eff
        axes[1, idx].plot(values, G_vals)
        axes[1, idx].set_xlabel(param_name)
        axes[1, idx].set_ylabel('G_eff')
        axes[1, idx].set_title(f'G_eff vs {param_name}')
        axes[1, idx].axhline(y=8.02, color='r', linestyle='--', label='3-Layer target')
        axes[1, idx].legend()

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'sensitivity_analysis.png', dpi=150)
    plt.close()

    print("Sensitivity analysis plot saved to sensitivity_analysis.png")


def grid_search_validation():
    """
    Perform grid search to find interface params that best match 3-Layer results.
    This provides a baseline to compare against PINN performance.
    """
    print("\n" + "=" * 60)
    print("Grid Search for Optimal Interface Parameters")
    print("=" * 60)

    # Target values
    K_target = 14.91
    G_target = 8.02

    # Material properties
    kappa_mat = 17.33
    mu_mat = 8.0
    kappa_inc = 1.73
    mu_inc = 0.8
    f = 0.3
    sz = 0.0067
    R = ((3 * f) / (4 * np.pi)) ** (1/3) * sz

    # Grid search (coarse)
    k_bar_range = np.logspace(0, 3, 15)  # 1 to 1000
    lambda_bar_range = np.linspace(0.01, 2, 10)
    mu_bar_range = np.linspace(0.01, 2, 10)
    alpha_range = np.linspace(0, 1, 10)

    best_error = float('inf')
    best_params = None

    total = len(k_bar_range) * len(lambda_bar_range) * len(mu_bar_range) * len(alpha_range)
    print(f"Searching {total:,} parameter combinations...")

    count = 0
    for k_bar in k_bar_range:
        for lambda_bar in lambda_bar_range:
            for mu_bar in mu_bar_range:
                for alpha in alpha_range:
                    try:
                        K, G = compute_effective_properties(
                            kappa_inc, mu_inc, kappa_mat, mu_mat,
                            k_bar, lambda_bar, mu_bar, alpha, f, R
                        )
                        error = ((K - K_target)/K_target)**2 + ((G - G_target)/G_target)**2

                        if error < best_error:
                            best_error = error
                            best_params = {
                                'k_bar': k_bar,
                                'lambda_bar': lambda_bar,
                                'mu_bar': mu_bar,
                                'alpha': alpha,
                                'K_pred': K,
                                'G_pred': G,
                            }
                    except:
                        pass

                    count += 1
                    if count % 10000 == 0:
                        print(f"  Progress: {count}/{total}")

    if best_params:
        K_error = abs(best_params['K_pred'] - K_target) / K_target * 100
        G_error = abs(best_params['G_pred'] - G_target) / G_target * 100

        print(f"\nBest parameters found:")
        print(f"  k_bar = {best_params['k_bar']:.4f}")
        print(f"  lambda_bar = {best_params['lambda_bar']:.4f}")
        print(f"  mu_bar = {best_params['mu_bar']:.4f}")
        print(f"  alpha = {best_params['alpha']:.4f}")
        print(f"\nReconstructed:")
        print(f"  K_eff = {best_params['K_pred']:.4f} (error: {K_error:.2f}%)")
        print(f"  G_eff = {best_params['G_pred']:.4f} (error: {G_error:.2f}%)")

    return best_params


if __name__ == "__main__":
    # Run validation
    validate_3layer_reference()

    # Run sensitivity analysis
    parameter_sensitivity_analysis()

    # Grid search (optional - takes a few minutes)
    # grid_search_validation()
