"""
Generate Training Data from 3-Layer Interphase Model

This script generates (input, target) pairs where:
- Input: (K_3layer, G_3layer, material_props)
- Target: Extended Interface params (k_bar, lambda_bar, mu_bar, alpha)
         that best match the 3-Layer effective properties

The target parameters are found via optimization.

Author: Sumanth Reddy Settipalli
"""

import sys
from pathlib import Path

# Add project root to path for standalone execution
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from scipy.optimize import minimize, differential_evolution
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from src.physics.three_layer_interphase import compute_3layer_properties
from src.physics.extended_interface import compute_effective_properties


def objective_function(params, K_target, G_target, kappa_inc, mu_inc, kappa_mat, mu_mat, f, R):
    """
    Objective function for optimization: minimize relative error in K and G.

    Args:
        params: [k_bar, lambda_bar, mu_bar, alpha]
        K_target, G_target: Target effective properties from 3-Layer model
        Other args: Material and geometric properties

    Returns:
        Weighted sum of squared relative errors
    """
    k_bar, lambda_bar, mu_bar, alpha = params

    # Enforce bounds
    if k_bar <= 0 or lambda_bar <= 0 or mu_bar <= 0 or alpha < 0 or alpha > 1:
        return 1e10

    try:
        K_pred, G_pred = compute_effective_properties(
            kappa_inc, mu_inc, kappa_mat, mu_mat,
            k_bar, lambda_bar, mu_bar, alpha, f, R
        )

        if not (np.isfinite(K_pred) and np.isfinite(G_pred) and K_pred > 0 and G_pred > 0):
            return 1e10

        # Relative error (weighted - K gets higher weight)
        K_error = ((K_pred - K_target) / K_target) ** 2
        G_error = ((G_pred - G_target) / G_target) ** 2

        return 2.0 * K_error + 1.0 * G_error  # Weight K more heavily

    except Exception:
        return 1e10


def find_optimal_interface_params(
    K_target: float,
    G_target: float,
    kappa_inc: float,
    mu_inc: float,
    kappa_mat: float,
    mu_mat: float,
    f: float,
    R: float,
    method: str = 'hybrid'
) -> dict:
    """
    Find Extended Interface parameters that produce the given K_target, G_target.

    Uses a hybrid approach: differential evolution for global search,
    followed by local optimization for refinement.

    Returns:
        Dict with optimal parameters and achieved error
    """
    # Parameter bounds
    bounds = [
        (1.0, 500.0),      # k_bar
        (0.01, 5.0),       # lambda_bar
        (0.01, 5.0),       # mu_bar
        (0.0, 1.0)         # alpha
    ]

    args = (K_target, G_target, kappa_inc, mu_inc, kappa_mat, mu_mat, f, R)

    if method == 'de' or method == 'hybrid':
        # Differential Evolution (global search)
        result_de = differential_evolution(
            objective_function,
            bounds,
            args=args,
            maxiter=100,
            seed=42,
            polish=False,
            workers=1
        )
        x0 = result_de.x
    else:
        # Random initial guess
        x0 = [50.0, 1.0, 1.0, 0.5]

    if method == 'hybrid' or method == 'local':
        # Local optimization (refinement)
        result = minimize(
            objective_function,
            x0,
            args=args,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500}
        )
        optimal_params = result.x
        final_error = result.fun
    else:
        optimal_params = x0
        final_error = objective_function(x0, *args)

    # Compute achieved properties
    K_achieved, G_achieved = compute_effective_properties(
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        optimal_params[0], optimal_params[1], optimal_params[2], optimal_params[3],
        f, R
    )

    return {
        'k_bar': optimal_params[0],
        'lambda_bar': optimal_params[1],
        'mu_bar': optimal_params[2],
        'alpha': optimal_params[3],
        'K_target': K_target,
        'G_target': G_target,
        'K_achieved': K_achieved,
        'G_achieved': G_achieved,
        'K_error': abs(K_achieved - K_target) / K_target,
        'G_error': abs(G_achieved - G_target) / G_target,
        'total_error': final_error
    }


def generate_3layer_config():
    """
    Generate a random 3-Layer configuration.

    Returns:
        Dict with material properties and phi values
    """
    # Sample material properties (vary around reference values)
    # Reference: kappa_inc=1.73, mu_inc=0.8, kappa_mat=17.33, mu_mat=8.0

    # Stiffness ratio (soft to stiff particles)
    SR = np.random.uniform(0.05, 2.0)

    # Matrix properties (moderate variation)
    kappa_mat = np.random.uniform(12.0, 25.0)
    mu_mat = np.random.uniform(6.0, 12.0)

    # Particle properties based on stiffness ratio
    kappa_inc = kappa_mat * SR
    mu_inc = mu_mat * SR

    # Volume fraction
    f = np.random.uniform(0.1, 0.5)

    # Phi values (increasing sequence)
    phi_1 = np.random.uniform(0.1, 0.3)
    phi_2 = np.random.uniform(phi_1 + 0.05, 0.5)
    phi_3 = np.random.uniform(phi_2 + 0.05, 0.7)

    # Size parameter
    sz = np.random.uniform(0.001, 0.01)
    R = ((3 * f) / (4 * np.pi)) ** (1/3) * sz

    return {
        'kappa_inc': kappa_inc,
        'mu_inc': mu_inc,
        'kappa_mat': kappa_mat,
        'mu_mat': mu_mat,
        'phi_1': phi_1,
        'phi_2': phi_2,
        'phi_3': phi_3,
        'f': f,
        'R': R,
        'sz': sz,
        'SR': SR
    }


def process_single_config(config, method='hybrid'):
    """Process a single 3-Layer configuration (for parallel processing)."""
    try:
        # Compute 3-Layer properties
        K_3layer, G_3layer = compute_3layer_properties(
            config['kappa_inc'], config['mu_inc'],
            config['kappa_mat'], config['mu_mat'],
            config['phi_1'], config['phi_2'], config['phi_3'],
            config['f']
        )

        # Check validity
        if not (np.isfinite(K_3layer) and np.isfinite(G_3layer) and K_3layer > 0 and G_3layer > 0):
            return None

        # Find optimal Extended Interface parameters
        result = find_optimal_interface_params(
            K_3layer, G_3layer,
            config['kappa_inc'], config['mu_inc'],
            config['kappa_mat'], config['mu_mat'],
            config['f'], config['R'],
            method=method
        )

        # Only keep good matches (< 5% error on both)
        if result['K_error'] < 0.05 and result['G_error'] < 0.05:
            return {
                **config,
                'K_3layer': K_3layer,
                'G_3layer': G_3layer,
                **result
            }
        return None

    except Exception as e:
        return None


def generate_3layer_dataset(
    n_samples: int = 1000,
    output_path: str = None,
    method: str = 'hybrid',
    n_workers: int = None
):
    """
    Generate dataset of 3-Layer configurations with matching Extended Interface params.

    Args:
        n_samples: Number of configurations to generate
        output_path: Path to save dataset
        method: Optimization method ('hybrid', 'de', 'local')
        n_workers: Number of parallel workers (default: CPU count - 1)

    Returns:
        List of successful configurations
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"Generating {n_samples} 3-Layer configurations...")
    print(f"Using {n_workers} parallel workers")

    # Generate configurations
    configs = [generate_3layer_config() for _ in range(n_samples * 2)]  # Generate extra to account for failures

    results = []
    failed = 0

    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_config, config, method): config
                   for config in configs}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result is not None:
                results.append(result)
                if len(results) >= n_samples:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
            else:
                failed += 1

    print(f"\nGenerated {len(results)} valid samples ({failed} failed)")

    # Statistics
    if results:
        K_errors = [r['K_error'] for r in results]
        G_errors = [r['G_error'] for r in results]
        print(f"K_error: mean={np.mean(K_errors):.4%}, max={np.max(K_errors):.4%}")
        print(f"G_error: mean={np.mean(G_errors):.4%}, max={np.max(G_errors):.4%}")

    # Save to file
    if output_path and results:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {output_path}")

    return results


def generate_reference_sample():
    """
    Generate a single sample using the reference values from Model_Results.md.
    This is useful for testing and validation.
    """
    config = {
        'kappa_inc': 1.7333,
        'mu_inc': 0.8,
        'kappa_mat': 17.3333,
        'mu_mat': 8.0,
        'phi_1': 0.2,
        'phi_2': 0.3,
        'phi_3': 0.4,
        'f': 0.3,
        'sz': 0.0067,
        'R': ((3 * 0.3) / (4 * np.pi)) ** (1/3) * 0.0067,
        'SR': 0.1
    }

    result = process_single_config(config, method='hybrid')

    if result:
        print("\nReference Sample Result:")
        print("=" * 50)
        print(f"3-Layer: K={result['K_3layer']:.4f}, G={result['G_3layer']:.4f}")
        print(f"Optimal params:")
        print(f"  k_bar = {result['k_bar']:.4f}")
        print(f"  lambda_bar = {result['lambda_bar']:.4f}")
        print(f"  mu_bar = {result['mu_bar']:.4f}")
        print(f"  alpha = {result['alpha']:.4f}")
        print(f"Achieved: K={result['K_achieved']:.4f}, G={result['G_achieved']:.4f}")
        print(f"Errors: K={result['K_error']:.4%}, G={result['G_error']:.4%}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate 3-Layer training data')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='./3layer_data.json',
                        help='Output file path')
    parser.add_argument('--method', type=str, default='hybrid',
                        choices=['hybrid', 'de', 'local'],
                        help='Optimization method')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers')
    parser.add_argument('--reference_only', action='store_true',
                        help='Only generate the reference sample')

    args = parser.parse_args()

    if args.reference_only:
        generate_reference_sample()
    else:
        generate_3layer_dataset(
            n_samples=args.n_samples,
            output_path=args.output,
            method=args.method,
            n_workers=args.workers
        )
