"""
Test PINN Model on Multiple 3-Layer Configurations with Varying Phi Values

This script validates the trained PINN (Experiment 4) against various 3-Layer
interphase configurations by varying:
1. Phi ratios (geometry of interphase layers)
2. Material stiffness ratios

Author: Sumanth Reddy Settipalli
Date: 2026-02-03
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

from models import PINN
from extended_interface import compute_effective_properties
from three_layer_interphase import compute_3layer_properties


def load_best_model() -> PINN:
    """Load the best trained PINN model (Experiment 4)."""
    model_path = Path(__file__).parent / 'exp4_larger_balanced' / 'pinn_best.pt'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    checkpoint = torch.load(model_path, weights_only=False)
    norm_params = checkpoint.get('norm_params', None)
    args = checkpoint.get('args', {})
    hidden_dims = args.get('hidden_dims', [256, 256, 256])

    model = PINN(
        hidden_dims=tuple(hidden_dims),
        norm_params=norm_params
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def test_phi_configuration(
    model: PINN,
    phi_1: float,
    phi_2: float,
    phi_3: float,
    kappa_inc: float = 1.7333,
    mu_inc: float = 0.8,
    kappa_mat: float = 17.3333,
    mu_mat: float = 8.0,
    f_particle: float = 0.3,
    interp_weights: Tuple[float, float, float] = (0.75, 0.50, 0.25)
) -> Dict:
    """
    Test the PINN on a specific 3-Layer configuration.

    Returns:
        Dictionary with targets, predictions, and errors
    """
    # Compute 3-Layer target
    K_3layer, G_3layer = compute_3layer_properties(
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        phi_1, phi_2, phi_3, f_particle, interp_weights
    )

    # Compute R from geometry
    sz = 0.0067
    R = ((3 * f_particle) / (4 * np.pi)) ** (1/3) * sz

    # Prepare input for PINN
    inputs = torch.tensor([[
        K_3layer, G_3layer,
        kappa_inc, mu_inc,
        kappa_mat, mu_mat,
        f_particle, R
    ]], dtype=torch.float32)

    # Predict interface parameters
    with torch.no_grad():
        params = model(inputs)[0].numpy()

    k_bar, lambda_bar, mu_bar, alpha = params

    # Reconstruct with Extended Interface model
    K_pred, G_pred = compute_effective_properties(
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        k_bar, lambda_bar, mu_bar, alpha,
        f_particle, R
    )

    # Compute errors
    K_error = abs(K_pred - K_3layer) / K_3layer * 100
    G_error = abs(G_pred - G_3layer) / G_3layer * 100

    return {
        'phi': (phi_1, phi_2, phi_3),
        'K_3layer': float(K_3layer),
        'G_3layer': float(G_3layer),
        'K_pred': float(K_pred),
        'G_pred': float(G_pred),
        'K_error': float(K_error),
        'G_error': float(G_error),
        'params': {
            'k_bar': float(k_bar),
            'lambda_bar': float(lambda_bar),
            'mu_bar': float(mu_bar),
            'alpha': float(alpha)
        }
    }


def run_phi_variation_tests(model: PINN) -> List[Dict]:
    """
    Run tests with various phi configurations.

    Phi values represent volume ratios at each interface:
    - phi_1: particle/coat1 interface
    - phi_2: coat1/coat2 interface
    - phi_3: coat2/coat3 interface

    Constraints: 0 < phi_1 < phi_2 < phi_3 < 1
    """
    results = []

    # Test configurations
    phi_configs = [
        # Reference configuration
        {'name': 'Reference (baseline)', 'phi': (0.2, 0.3, 0.4)},

        # Thin interphase (phi values close together, closer to 1)
        {'name': 'Thin interphase', 'phi': (0.6, 0.7, 0.8)},

        # Thick interphase (phi values spread apart, closer to 0)
        {'name': 'Thick interphase', 'phi': (0.1, 0.3, 0.5)},

        # Dominant inner layer
        {'name': 'Dominant inner layer', 'phi': (0.1, 0.15, 0.4)},

        # Dominant outer layer
        {'name': 'Dominant outer layer', 'phi': (0.3, 0.35, 0.5)},

        # Uniform layers (equal spacing)
        {'name': 'Uniform layers', 'phi': (0.25, 0.50, 0.75)},

        # Very thin interphase
        {'name': 'Very thin interphase', 'phi': (0.7, 0.8, 0.9)},

        # Very thick interphase
        {'name': 'Very thick interphase', 'phi': (0.05, 0.2, 0.35)},

        # Asymmetric - thick inner
        {'name': 'Asymmetric thick inner', 'phi': (0.1, 0.4, 0.5)},

        # Asymmetric - thick outer
        {'name': 'Asymmetric thick outer', 'phi': (0.3, 0.4, 0.7)},
    ]

    print("=" * 80)
    print("PHI VARIATION TESTS")
    print("=" * 80)
    print(f"\n{'Configuration':<25} {'Phi Values':<20} {'K_err%':>10} {'G_err%':>10}")
    print("-" * 80)

    for config in phi_configs:
        phi = config['phi']
        result = test_phi_configuration(model, phi[0], phi[1], phi[2])
        result['name'] = config['name']
        results.append(result)

        print(f"{config['name']:<25} {str(phi):<20} {result['K_error']:>10.2f} {result['G_error']:>10.2f}")

    return results


def run_material_variation_tests(model: PINN) -> List[Dict]:
    """
    Run tests with various material stiffness ratios.
    """
    results = []

    # Reference values
    kappa_mat = 17.3333
    mu_mat = 8.0

    # Material configurations (stiffness ratios: inclusion/matrix)
    material_configs = [
        # Soft particles (reference - SR=0.1)
        {'name': 'Soft particle (SR=0.1)', 'SR': 0.1},

        # Very soft particles
        {'name': 'Very soft (SR=0.05)', 'SR': 0.05},

        # Medium soft
        {'name': 'Medium soft (SR=0.3)', 'SR': 0.3},

        # Nearly equal
        {'name': 'Nearly equal (SR=0.8)', 'SR': 0.8},

        # Equal
        {'name': 'Equal (SR=1.0)', 'SR': 1.0},

        # Stiff particles
        {'name': 'Stiff (SR=2.0)', 'SR': 2.0},

        # Very stiff
        {'name': 'Very stiff (SR=5.0)', 'SR': 5.0},
    ]

    print("\n" + "=" * 80)
    print("MATERIAL STIFFNESS RATIO TESTS (with reference phi=[0.2, 0.3, 0.4])")
    print("=" * 80)
    print(f"\n{'Configuration':<25} {'Stiffness Ratio':<15} {'K_err%':>10} {'G_err%':>10}")
    print("-" * 80)

    for config in material_configs:
        SR = config['SR']
        kappa_inc = kappa_mat * SR
        mu_inc = mu_mat * SR

        result = test_phi_configuration(
            model,
            phi_1=0.2, phi_2=0.3, phi_3=0.4,
            kappa_inc=kappa_inc, mu_inc=mu_inc,
            kappa_mat=kappa_mat, mu_mat=mu_mat
        )
        result['name'] = config['name']
        result['SR'] = SR
        results.append(result)

        print(f"{config['name']:<25} {SR:<15.2f} {result['K_error']:>10.2f} {result['G_error']:>10.2f}")

    return results


def run_volume_fraction_tests(model: PINN) -> List[Dict]:
    """
    Run tests with various particle volume fractions.
    """
    results = []

    # Volume fraction configurations
    f_configs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    print("\n" + "=" * 80)
    print("VOLUME FRACTION TESTS (with reference phi=[0.2, 0.3, 0.4], SR=0.1)")
    print("=" * 80)
    print(f"\n{'Volume Fraction':<20} {'K_3layer':>12} {'K_pred':>12} {'K_err%':>10} {'G_err%':>10}")
    print("-" * 80)

    for f_val in f_configs:
        result = test_phi_configuration(
            model,
            phi_1=0.2, phi_2=0.3, phi_3=0.4,
            f_particle=f_val
        )
        result['f'] = f_val
        results.append(result)

        print(f"{f_val:<20.2f} {result['K_3layer']:>12.4f} {result['K_pred']:>12.4f} {result['K_error']:>10.2f} {result['G_error']:>10.2f}")

    return results


def run_interpolation_weight_tests(model: PINN) -> List[Dict]:
    """
    Run tests with various interpolation weight schemes for interphase layers.
    """
    results = []

    # Interpolation weight configurations (w1, w2, w3)
    # wi = weight of inclusion properties in layer i
    weight_configs = [
        {'name': 'Linear (default)', 'weights': (0.75, 0.50, 0.25)},
        {'name': 'Strong gradient', 'weights': (0.90, 0.50, 0.10)},
        {'name': 'Weak gradient', 'weights': (0.60, 0.50, 0.40)},
        {'name': 'Inclusion-like', 'weights': (0.80, 0.70, 0.60)},
        {'name': 'Matrix-like', 'weights': (0.40, 0.30, 0.20)},
        {'name': 'Step-wise', 'weights': (0.90, 0.50, 0.10)},
    ]

    print("\n" + "=" * 80)
    print("INTERPOLATION WEIGHT TESTS (phi=[0.2, 0.3, 0.4], SR=0.1)")
    print("=" * 80)
    print(f"\n{'Configuration':<20} {'Weights':<25} {'K_err%':>10} {'G_err%':>10}")
    print("-" * 80)

    for config in weight_configs:
        result = test_phi_configuration(
            model,
            phi_1=0.2, phi_2=0.3, phi_3=0.4,
            interp_weights=config['weights']
        )
        result['name'] = config['name']
        result['weights'] = config['weights']
        results.append(result)

        weights_str = f"({config['weights'][0]:.2f}, {config['weights'][1]:.2f}, {config['weights'][2]:.2f})"
        print(f"{config['name']:<20} {weights_str:<25} {result['K_error']:>10.2f} {result['G_error']:>10.2f}")

    return results


def generate_summary_statistics(all_results: Dict) -> Dict:
    """Generate summary statistics from all test results."""

    def compute_stats(results: List[Dict]) -> Dict:
        K_errors = [r['K_error'] for r in results]
        G_errors = [r['G_error'] for r in results]

        return {
            'K_error': {
                'mean': np.mean(K_errors),
                'std': np.std(K_errors),
                'min': np.min(K_errors),
                'max': np.max(K_errors),
                'median': np.median(K_errors)
            },
            'G_error': {
                'mean': np.mean(G_errors),
                'std': np.std(G_errors),
                'min': np.min(G_errors),
                'max': np.max(G_errors),
                'median': np.median(G_errors)
            },
            'n_samples': len(results)
        }

    summary = {}
    for test_name, results in all_results.items():
        summary[test_name] = compute_stats(results)

    # Overall statistics
    all_tests = []
    for results in all_results.values():
        all_tests.extend(results)
    summary['overall'] = compute_stats(all_tests)

    return summary


def plot_results(all_results: Dict, output_dir: Path):
    """Create visualization plots for the test results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Phi variation errors
    ax1 = axes[0, 0]
    phi_results = all_results['phi_variation']
    names = [r['name'] for r in phi_results]
    K_errors = [r['K_error'] for r in phi_results]
    G_errors = [r['G_error'] for r in phi_results]

    x = np.arange(len(names))
    width = 0.35
    ax1.bar(x - width/2, K_errors, width, label='K_eff error', color='steelblue')
    ax1.bar(x + width/2, G_errors, width, label='G_eff error', color='coral')
    ax1.set_ylabel('Error (%)')
    ax1.set_title('Phi Configuration Variation')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='2% threshold')
    ax1.set_ylim(0, max(max(K_errors), max(G_errors)) * 1.2)

    # Plot 2: Material stiffness ratio errors
    ax2 = axes[0, 1]
    mat_results = all_results['material_variation']
    SRs = [r['SR'] for r in mat_results]
    K_errors = [r['K_error'] for r in mat_results]
    G_errors = [r['G_error'] for r in mat_results]

    ax2.plot(SRs, K_errors, 'o-', label='K_eff error', color='steelblue', markersize=8)
    ax2.plot(SRs, G_errors, 's-', label='G_eff error', color='coral', markersize=8)
    ax2.set_xlabel('Stiffness Ratio (SR)')
    ax2.set_ylabel('Error (%)')
    ax2.set_title('Material Stiffness Ratio Variation')
    ax2.legend()
    ax2.axhline(y=2, color='green', linestyle='--', alpha=0.5)
    ax2.set_xscale('log')

    # Plot 3: Volume fraction errors
    ax3 = axes[1, 0]
    vf_results = all_results['volume_fraction']
    f_vals = [r['f'] for r in vf_results]
    K_errors = [r['K_error'] for r in vf_results]
    G_errors = [r['G_error'] for r in vf_results]

    ax3.plot(f_vals, K_errors, 'o-', label='K_eff error', color='steelblue', markersize=8)
    ax3.plot(f_vals, G_errors, 's-', label='G_eff error', color='coral', markersize=8)
    ax3.set_xlabel('Volume Fraction (f)')
    ax3.set_ylabel('Error (%)')
    ax3.set_title('Volume Fraction Variation')
    ax3.legend()
    ax3.axhline(y=2, color='green', linestyle='--', alpha=0.5)

    # Plot 4: Summary box plot
    ax4 = axes[1, 1]
    test_names = ['Phi\nVariation', 'Material\nVariation', 'Volume\nFraction', 'Interp.\nWeights']
    K_data = [
        [r['K_error'] for r in all_results['phi_variation']],
        [r['K_error'] for r in all_results['material_variation']],
        [r['K_error'] for r in all_results['volume_fraction']],
        [r['K_error'] for r in all_results['interpolation_weights']]
    ]
    G_data = [
        [r['G_error'] for r in all_results['phi_variation']],
        [r['G_error'] for r in all_results['material_variation']],
        [r['G_error'] for r in all_results['volume_fraction']],
        [r['G_error'] for r in all_results['interpolation_weights']]
    ]

    positions = np.arange(len(test_names)) * 2
    bp1 = ax4.boxplot(K_data, positions=positions - 0.3, widths=0.5, patch_artist=True)
    bp2 = ax4.boxplot(G_data, positions=positions + 0.3, widths=0.5, patch_artist=True)

    for patch in bp1['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor('coral')
        patch.set_alpha(0.7)

    ax4.set_xticks(positions)
    ax4.set_xticklabels(test_names)
    ax4.set_ylabel('Error (%)')
    ax4.set_title('Error Distribution by Test Category')
    ax4.legend([bp1['boxes'][0], bp2['boxes'][0]], ['K_eff', 'G_eff'], loc='upper right')
    ax4.axhline(y=2, color='green', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'phi_variation_results.png', dpi=150)
    plt.close()

    print(f"\nPlot saved to {output_dir / 'phi_variation_results.png'}")


def main():
    """Main function to run all tests."""
    print("\n" + "=" * 80)
    print("PINN MODEL VALIDATION: MULTIPLE 3-LAYER CONFIGURATIONS")
    print("Testing Model: exp4_larger_balanced/pinn_best.pt")
    print("=" * 80)

    # Load model
    print("\nLoading best PINN model...")
    model = load_best_model()
    print("Model loaded successfully!")

    # Run all tests
    all_results = {}

    # 1. Phi variation tests
    all_results['phi_variation'] = run_phi_variation_tests(model)

    # 2. Material stiffness ratio tests
    all_results['material_variation'] = run_material_variation_tests(model)

    # 3. Volume fraction tests
    all_results['volume_fraction'] = run_volume_fraction_tests(model)

    # 4. Interpolation weight tests
    all_results['interpolation_weights'] = run_interpolation_weight_tests(model)

    # Generate summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    summary = generate_summary_statistics(all_results)

    for test_name, stats in summary.items():
        print(f"\n{test_name.upper().replace('_', ' ')}:")
        print(f"  K_error: mean={stats['K_error']['mean']:.2f}%, std={stats['K_error']['std']:.2f}%, "
              f"range=[{stats['K_error']['min']:.2f}%, {stats['K_error']['max']:.2f}%]")
        print(f"  G_error: mean={stats['G_error']['mean']:.2f}%, std={stats['G_error']['std']:.2f}%, "
              f"range=[{stats['G_error']['min']:.2f}%, {stats['G_error']['max']:.2f}%]")

    # Save results
    output_dir = Path(__file__).parent
    results_file = output_dir / 'phi_variation_results.json'

    with open(results_file, 'w') as f:
        json.dump({
            'results': all_results,
            'summary': summary
        }, f, indent=2, default=str)

    print(f"\nDetailed results saved to {results_file}")

    # Generate plots
    plot_results(all_results, output_dir)

    # Print conclusions
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    overall = summary['overall']
    print(f"\nOverall Performance (across all {overall['n_samples']} test cases):")
    print(f"  - K_eff error: {overall['K_error']['mean']:.2f}% +/- {overall['K_error']['std']:.2f}%")
    print(f"  - G_eff error: {overall['G_error']['mean']:.2f}% +/- {overall['G_error']['std']:.2f}%")

    # Find challenging cases
    all_tests = []
    for results in all_results.values():
        all_tests.extend(results)

    worst_K = max(all_tests, key=lambda x: x['K_error'])
    worst_G = max(all_tests, key=lambda x: x['G_error'])

    print(f"\nMost Challenging Cases:")
    print(f"  - Worst K_error: {worst_K['K_error']:.2f}% ({worst_K.get('name', 'unnamed')})")
    print(f"  - Worst G_error: {worst_G['G_error']:.2f}% ({worst_G.get('name', 'unnamed')})")

    # Check pass rate
    pass_threshold = 5.0  # 5% error threshold
    passed = sum(1 for t in all_tests if t['K_error'] < pass_threshold and t['G_error'] < pass_threshold)
    print(f"\nPass Rate (<{pass_threshold}% error for both K and G): {passed}/{len(all_tests)} ({100*passed/len(all_tests):.1f}%)")

    return all_results, summary


if __name__ == "__main__":
    all_results, summary = main()
