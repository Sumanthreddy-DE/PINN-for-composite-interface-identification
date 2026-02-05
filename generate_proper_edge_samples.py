"""
Generate PROPER Edge Region Samples with ACTUAL Interface Parameters

CRITICAL FIX: The previous edge samples used PLACEHOLDER/FAKE interface parameters.
This script generates REAL samples where we:
1. Sample interface parameters from full range
2. Compute K_eff, G_eff using Extended Interface model
3. Store both the inputs AND the ground truth outputs

For edge regions (high-f, high-SR), we:
- Sample more aggressively in these regions
- Ensure interface params span the FULL range (not restricted)

Author: Sumanth Reddy Settipalli
Date: 2026-02-03 (FIXED VERSION)
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse

from extended_interface import compute_effective_properties


def generate_edge_samples_proper(
    n_samples: int,
    f_range: Tuple[float, float] = (0.35, 0.50),
    SR_range: Tuple[float, float] = (0.05, 3.0),
    seed: int = 42
) -> List[Dict]:
    """
    Generate PROPER edge region samples with REAL interface parameters.

    The key difference from the buggy version:
    - We sample interface parameters FIRST (k_bar, lambda_bar, etc.)
    - Then compute K_eff, G_eff from Extended Interface model
    - This gives us CORRECT input-output pairs

    Args:
        n_samples: Number of samples
        f_range: Volume fraction range (edge region)
        SR_range: Stiffness ratio range
        seed: Random seed

    Returns:
        List of sample dicts with CORRECT interface parameters
    """
    np.random.seed(seed)
    samples = []

    # Interface parameter ranges (FULL RANGE like original dataset)
    k_bar_range = (1.0, 500.0)  # log-uniform
    lambda_bar_range = (0.01, 5.0)
    mu_bar_range = (0.01, 5.0)
    alpha_range = (0.0, 1.0)

    # Material property ranges
    kappa_mat_range = (10.0, 30.0)
    mu_mat_range = (5.0, 15.0)
    sz_range = (0.001, 0.01)

    print(f"Generating {n_samples} PROPER edge samples...")
    print(f"  Volume fraction: f in [{f_range[0]}, {f_range[1]}]")
    print(f"  Stiffness ratio: SR in [{SR_range[0]}, {SR_range[1]}]")
    print(f"  Interface k_bar: FULL range [{k_bar_range[0]}, {k_bar_range[1]}] (log-uniform)")

    attempts = 0
    max_attempts = n_samples * 10

    pbar = tqdm(total=n_samples, desc="Generating samples")

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1

        # Sample interface parameters (GROUND TRUTH OUTPUTS)
        k_bar = np.exp(np.random.uniform(np.log(k_bar_range[0]), np.log(k_bar_range[1])))
        lambda_bar = np.random.uniform(*lambda_bar_range)
        mu_bar = np.random.uniform(*mu_bar_range)
        alpha = np.random.uniform(*alpha_range)

        # Sample material properties (INPUTS)
        kappa_mat = np.random.uniform(*kappa_mat_range)
        mu_mat = np.random.uniform(*mu_mat_range)

        # Sample from edge regions with higher probability
        # Mix of high-f and varied SR
        if np.random.random() < 0.6:
            # High volume fraction (the main failure region)
            f = np.random.uniform(f_range[0], f_range[1])
            SR = np.random.uniform(SR_range[0], SR_range[1])
        else:
            # Higher SR with moderate f
            f = np.random.uniform(0.2, 0.4)
            SR = np.random.uniform(1.5, 5.0)

        kappa_inc = kappa_mat * SR
        mu_inc = mu_mat * SR

        # Compute R from geometry
        sz = np.random.uniform(*sz_range)
        R = ((3 * f) / (4 * np.pi)) ** (1/3) * sz

        try:
            # Compute effective properties using Extended Interface model
            K_eff, G_eff = compute_effective_properties(
                kappa_inc, mu_inc, kappa_mat, mu_mat,
                k_bar, lambda_bar, mu_bar, alpha, f, R
            )

            # Validate
            if K_eff > 0 and G_eff > 0 and np.isfinite(K_eff) and np.isfinite(G_eff):
                # Store sample with CORRECT interface parameters
                samples.append({
                    # Inputs (what PINN receives)
                    'K_eff': float(K_eff),
                    'G_eff': float(G_eff),
                    'kappa_inc': float(kappa_inc),
                    'mu_inc': float(mu_inc),
                    'kappa_mat': float(kappa_mat),
                    'mu_mat': float(mu_mat),
                    'f': float(f),
                    'R': float(R),
                    # Outputs (what PINN should predict) - GROUND TRUTH
                    'k_bar': float(k_bar),
                    'lambda_bar': float(lambda_bar),
                    'mu_bar': float(mu_bar),
                    'alpha': float(alpha),
                    # Metadata
                    'SR': float(SR),
                    'region': 'edge_proper'
                })
                pbar.update(1)

        except Exception as e:
            continue

    pbar.close()
    print(f"Generated {len(samples)} valid samples from {attempts} attempts")

    return samples


def generate_3layer_like_samples(
    n_samples: int,
    seed: int = 43
) -> List[Dict]:
    """
    Generate samples that approximate 3-layer configurations.

    These use the Extended Interface model but with interface parameters
    that approximate what the 3-layer model produces.

    Based on the successful grid search result: k_bar ~ 300-400 for reference config.
    """
    np.random.seed(seed)
    samples = []

    # Interface parameters biased toward values that work for 3-layer matching
    # From grid search: k_bar ~ 300-400, lambda_bar ~ 0.5-2, mu_bar ~ 2-4, alpha ~ 0.6-0.8
    k_bar_range = (150.0, 500.0)  # Higher range like grid search optimal
    lambda_bar_range = (0.5, 2.5)
    mu_bar_range = (1.5, 4.0)
    alpha_range = (0.5, 0.9)

    # Material ranges
    kappa_mat_range = (10.0, 30.0)
    mu_mat_range = (5.0, 15.0)
    f_range = (0.1, 0.5)
    SR_range = (0.05, 2.0)  # Focus on soft particles (like reference)
    sz_range = (0.001, 0.01)

    print(f"Generating {n_samples} 3-layer-like samples...")

    pbar = tqdm(total=n_samples, desc="3-layer-like samples")
    attempts = 0
    max_attempts = n_samples * 10

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1

        # Sample interface parameters from "good" range
        k_bar = np.random.uniform(*k_bar_range)
        lambda_bar = np.random.uniform(*lambda_bar_range)
        mu_bar = np.random.uniform(*mu_bar_range)
        alpha = np.random.uniform(*alpha_range)

        # Sample material properties
        kappa_mat = np.random.uniform(*kappa_mat_range)
        mu_mat = np.random.uniform(*mu_mat_range)
        f = np.random.uniform(*f_range)
        SR = np.random.uniform(*SR_range)

        kappa_inc = kappa_mat * SR
        mu_inc = mu_mat * SR

        sz = np.random.uniform(*sz_range)
        R = ((3 * f) / (4 * np.pi)) ** (1/3) * sz

        try:
            K_eff, G_eff = compute_effective_properties(
                kappa_inc, mu_inc, kappa_mat, mu_mat,
                k_bar, lambda_bar, mu_bar, alpha, f, R
            )

            if K_eff > 0 and G_eff > 0 and np.isfinite(K_eff) and np.isfinite(G_eff):
                samples.append({
                    'K_eff': float(K_eff),
                    'G_eff': float(G_eff),
                    'kappa_inc': float(kappa_inc),
                    'mu_inc': float(mu_inc),
                    'kappa_mat': float(kappa_mat),
                    'mu_mat': float(mu_mat),
                    'f': float(f),
                    'R': float(R),
                    'k_bar': float(k_bar),
                    'lambda_bar': float(lambda_bar),
                    'mu_bar': float(mu_bar),
                    'alpha': float(alpha),
                    'SR': float(SR),
                    'region': '3layer_like'
                })
                pbar.update(1)

        except Exception as e:
            continue

    pbar.close()
    return samples


def main():
    parser = argparse.ArgumentParser(description='Generate PROPER edge samples')
    parser.add_argument('--n_edge', type=int, default=10000,
                        help='Number of edge region samples')
    parser.add_argument('--n_3layer_like', type=int, default=5000,
                        help='Number of 3-layer-like samples')
    parser.add_argument('--output', type=str, default='edge_samples_proper.json',
                        help='Output file')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("GENERATING PROPER EDGE SAMPLES (FIXED VERSION)")
    print("=" * 70)
    print("\nCRITICAL FIX: Previous edge_samples.json had FAKE interface parameters!")
    print("This version computes REAL K_eff, G_eff from Extended Interface model.")
    print()

    # Generate edge region samples
    edge_samples = generate_edge_samples_proper(args.n_edge, seed=args.seed)

    # Generate 3-layer-like samples
    three_layer_like = generate_3layer_like_samples(args.n_3layer_like, seed=args.seed + 1)

    # Combine
    all_samples = edge_samples + three_layer_like

    # Statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    k_bars = [s['k_bar'] for s in all_samples]
    f_vals = [s['f'] for s in all_samples]
    SR_vals = [s['SR'] for s in all_samples]
    K_vals = [s['K_eff'] for s in all_samples]

    print(f"Total samples: {len(all_samples)}")
    print(f"  - Edge region: {len(edge_samples)}")
    print(f"  - 3-layer-like: {len(three_layer_like)}")
    print(f"\nParameter ranges:")
    print(f"  k_bar: [{min(k_bars):.2f}, {max(k_bars):.2f}], mean={np.mean(k_bars):.2f}")
    print(f"  f: [{min(f_vals):.2f}, {max(f_vals):.2f}], mean={np.mean(f_vals):.2f}")
    print(f"  SR: [{min(SR_vals):.2f}, {max(SR_vals):.2f}], mean={np.mean(SR_vals):.2f}")
    print(f"  K_eff: [{min(K_vals):.2f}, {max(K_vals):.2f}], mean={np.mean(K_vals):.2f}")

    # High-f and high-SR coverage
    high_f_count = sum(1 for s in all_samples if s['f'] > 0.35)
    high_SR_count = sum(1 for s in all_samples if s['SR'] > 2.0)
    print(f"\nEdge region coverage:")
    print(f"  High-f (f > 0.35): {high_f_count} samples ({100*high_f_count/len(all_samples):.1f}%)")
    print(f"  High-SR (SR > 2): {high_SR_count} samples ({100*high_SR_count/len(all_samples):.1f}%)")

    # Save
    output_path = Path(__file__).parent / args.output
    with open(output_path, 'w') as f:
        json.dump(all_samples, f, indent=2)

    print(f"\nSaved to: {output_path}")

    return all_samples


if __name__ == "__main__":
    main()
