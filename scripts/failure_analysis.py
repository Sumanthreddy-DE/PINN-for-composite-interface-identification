"""
Failure Analysis for PINN Model

Analyzes the 6 failing configurations (>5% error) and identifies patterns.
Creates visualizations showing failure regions in parameter space.

Author: Sumanth Reddy Settipalli
Date: 2026-02-03
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results():
    """Load test results from JSON file."""
    results_path = Path(__file__).parent / 'phi_variation_results.json'
    with open(results_path, 'r') as f:
        return json.load(f)


def extract_failing_configs(data, threshold=5.0):
    """Extract configurations that fail the threshold test."""
    failures = []
    passes = []

    for category, results in data['results'].items():
        for result in results:
            max_error = max(result['K_error'], result['G_error'])
            entry = {
                'category': category,
                'name': result.get('name', f"f={result.get('f', 'N/A')}"),
                'K_error': result['K_error'],
                'G_error': result['G_error'],
                'max_error': max_error,
                'phi': result.get('phi', [0.2, 0.3, 0.4]),
                'f': result.get('f', 0.3),
                'SR': result.get('SR', 0.1),
                'weights': result.get('weights', [0.75, 0.5, 0.25]),
                'params': result['params']
            }

            if max_error >= threshold:
                failures.append(entry)
            else:
                passes.append(entry)

    return failures, passes


def print_failure_report(failures, passes):
    """Print detailed failure analysis report."""
    print("=" * 80)
    print("PINN MODEL FAILURE ANALYSIS REPORT")
    print("=" * 80)

    total = len(failures) + len(passes)
    print(f"\nOverall: {len(passes)}/{total} passed ({100*len(passes)/total:.1f}%)")
    print(f"         {len(failures)}/{total} failed ({100*len(failures)/total:.1f}%)")

    print("\n" + "-" * 80)
    print("FAILING CONFIGURATIONS (>5% error on K or G)")
    print("-" * 80)
    print(f"\n{'#':<3} {'Category':<20} {'Name':<25} {'K_err%':>10} {'G_err%':>10}")
    print("-" * 80)

    for i, f in enumerate(sorted(failures, key=lambda x: x['max_error'], reverse=True), 1):
        print(f"{i:<3} {f['category']:<20} {f['name']:<25} {f['K_error']:>10.2f} {f['G_error']:>10.2f}")

    # Pattern analysis
    print("\n" + "=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)

    # Volume fraction analysis
    vf_failures = [f for f in failures if f['category'] == 'volume_fraction']
    print(f"\n1. VOLUME FRACTION FAILURES: {len(vf_failures)}")
    if vf_failures:
        f_values = [f['f'] for f in vf_failures]
        print(f"   Failing f values: {f_values}")
        print(f"   Range: f >= {min(f_values):.2f}")
        print("   Cause: Training data concentrated in f=0.1-0.4 range")

    # Stiffness ratio analysis
    sr_failures = [f for f in failures if f['category'] == 'material_variation' and f['SR'] >= 2]
    print(f"\n2. STIFFNESS RATIO FAILURES: {len(sr_failures)}")
    if sr_failures:
        sr_values = [f['SR'] for f in sr_failures]
        print(f"   Failing SR values: {sr_values}")
        print("   Cause: Training range limited to SR < 2 for dense coverage")

    # Phi configuration analysis
    phi_failures = [f for f in failures if f['category'] == 'phi_variation']
    print(f"\n3. PHI CONFIGURATION FAILURES: {len(phi_failures)}")
    if phi_failures:
        for f in phi_failures:
            phi = f['phi']
            spread = phi[2] - phi[0]
            print(f"   {f['name']}: phi={phi}, spread={spread:.2f}")
        print("   Cause: Wide phi spread or uniform layers less common in training")

    # Interpolation weight analysis
    iw_failures = [f for f in failures if f['category'] == 'interpolation_weights']
    print(f"\n4. INTERPOLATION WEIGHT FAILURES: {len(iw_failures)}")
    if iw_failures:
        for f in iw_failures:
            print(f"   {f['name']}: weights={f['weights']}")
        print("   Cause: Non-linear gradients less common in training")

    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print("""
1. EXTRAPOLATION is the primary cause of failures
   - 5/6 failures are outside the dense training region
   - Model works well within training distribution

2. SAFE OPERATING RANGES (based on passing configurations):
   - Volume fraction: f in [0.1, 0.35] (100% pass rate)
   - Stiffness ratio: SR in [0.05, 2.0] (100% pass rate)
   - Phi spread: (phi_3 - phi_1) <= 0.4 recommended

3. RECOMMENDATIONS:
   a) Add more training samples in high-f region (f > 0.35)
   b) Add more high-SR samples (SR > 2)
   c) Include more uniform/wide-spread phi configurations
   d) Consider clamping predictions to valid ranges
""")

    return failures, passes


def create_visualizations(failures, passes, output_dir):
    """Create visualization plots for failure analysis."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Error vs Volume Fraction
    ax1 = axes[0, 0]

    # Get all volume fraction results
    f_values_pass = [p['f'] for p in passes if p['category'] == 'volume_fraction']
    f_values_fail = [f['f'] for f in failures if f['category'] == 'volume_fraction']
    k_err_pass = [p['K_error'] for p in passes if p['category'] == 'volume_fraction']
    k_err_fail = [f['K_error'] for f in failures if f['category'] == 'volume_fraction']

    # Also add reference config (f=0.3) from other categories
    all_f = f_values_pass + f_values_fail
    all_k = k_err_pass + k_err_fail

    # Sort by f value for line plot
    sorted_data = sorted(zip(all_f, all_k))
    f_sorted, k_sorted = zip(*sorted_data) if sorted_data else ([], [])

    ax1.plot(f_sorted, k_sorted, 'o-', color='steelblue', markersize=10, linewidth=2)
    ax1.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5% threshold')
    ax1.axvspan(0.35, 0.55, alpha=0.2, color='red', label='High error region')
    ax1.set_xlabel('Volume Fraction (f)', fontsize=12)
    ax1.set_ylabel('K_eff Error (%)', fontsize=12)
    ax1.set_title('K Error vs Volume Fraction', fontsize=14)
    ax1.legend()
    ax1.set_xlim(0.05, 0.55)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error vs Stiffness Ratio
    ax2 = axes[0, 1]

    sr_values_pass = [p['SR'] for p in passes if p['category'] == 'material_variation']
    sr_values_fail = [f['SR'] for f in failures if f['category'] == 'material_variation']
    k_err_sr_pass = [p['K_error'] for p in passes if p['category'] == 'material_variation']
    k_err_sr_fail = [f['K_error'] for f in failures if f['category'] == 'material_variation']

    ax2.scatter(sr_values_pass, k_err_sr_pass, c='green', s=100, marker='o', label='Pass', zorder=3)
    ax2.scatter(sr_values_fail, k_err_sr_fail, c='red', s=100, marker='X', label='Fail', zorder=3)
    ax2.axhline(y=5, color='red', linestyle='--', alpha=0.7)
    ax2.axvspan(3, 6, alpha=0.2, color='red', label='High error region')
    ax2.set_xlabel('Stiffness Ratio (SR)', fontsize=12)
    ax2.set_ylabel('K_eff Error (%)', fontsize=12)
    ax2.set_title('K Error vs Stiffness Ratio', fontsize=14)
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: K vs G Error scatter (all configs)
    ax3 = axes[1, 0]

    k_all_pass = [p['K_error'] for p in passes]
    g_all_pass = [p['G_error'] for p in passes]
    k_all_fail = [f['K_error'] for f in failures]
    g_all_fail = [f['G_error'] for f in failures]

    ax3.scatter(k_all_pass, g_all_pass, c='green', s=80, marker='o', alpha=0.7, label=f'Pass ({len(passes)})')
    ax3.scatter(k_all_fail, g_all_fail, c='red', s=100, marker='X', alpha=0.9, label=f'Fail ({len(failures)})')

    # Add 5% threshold box
    ax3.axvline(x=5, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(y=5, color='red', linestyle='--', alpha=0.5)
    ax3.fill_between([0, 5], [0, 0], [5, 5], alpha=0.1, color='green', label='Safe region')

    ax3.set_xlabel('K_eff Error (%)', fontsize=12)
    ax3.set_ylabel('G_eff Error (%)', fontsize=12)
    ax3.set_title('K vs G Error (All Configurations)', fontsize=14)
    ax3.legend()
    ax3.set_xlim(-0.5, 15)
    ax3.set_ylim(-0.5, 10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Failure categories pie chart
    ax4 = axes[1, 1]

    categories = {}
    for f in failures:
        cat = f['category']
        categories[cat] = categories.get(cat, 0) + 1

    if categories:
        labels = [c.replace('_', '\n') for c in categories.keys()]
        sizes = list(categories.values())
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']

        wedges, texts, autotexts = ax4.pie(
            sizes, labels=labels, autopct='%1.0f%%',
            colors=colors[:len(sizes)], startangle=90,
            explode=[0.05]*len(sizes)
        )
        ax4.set_title('Failure Distribution by Category', fontsize=14)
    else:
        ax4.text(0.5, 0.5, 'No failures!', ha='center', va='center', fontsize=16)
        ax4.set_title('Failure Distribution', fontsize=14)

    plt.tight_layout()
    output_path = output_dir / 'failure_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to: {output_path}")

    return output_path


def create_training_coverage_plot(failures, passes, output_dir):
    """Create a plot showing training data coverage vs failure regions."""

    fig, ax = plt.subplots(figsize=(10, 8))

    # Training data ranges (from CSA_Results.md)
    training_ranges = {
        'f': (0.1, 0.5),
        'SR': (0.05, 2.0),  # Approximate from κ_inc/κ_mat ranges
    }

    # Dense training region (where most samples are)
    dense_region = {
        'f': (0.1, 0.4),
        'SR': (0.05, 2.0),
    }

    # Plot training coverage
    ax.add_patch(plt.Rectangle(
        (dense_region['f'][0], dense_region['SR'][0]),
        dense_region['f'][1] - dense_region['f'][0],
        dense_region['SR'][1] - dense_region['SR'][0],
        facecolor='green', alpha=0.2, edgecolor='green',
        linewidth=2, label='Dense training region'
    ))

    # Plot sparse training region
    ax.add_patch(plt.Rectangle(
        (0.4, 0.05),
        0.1, 2.0 - 0.05,
        facecolor='yellow', alpha=0.2, edgecolor='orange',
        linewidth=2, label='Sparse training region'
    ))

    # Plot extrapolation region (SR > 2)
    ax.add_patch(plt.Rectangle(
        (0.1, 2.0),
        0.4, 3.0,
        facecolor='red', alpha=0.1, edgecolor='red',
        linewidth=2, label='Extrapolation region'
    ))

    # Plot passing configs as green dots
    for p in passes:
        if p['category'] == 'material_variation':
            ax.scatter(0.3, p['SR'], c='green', s=100, marker='o', zorder=5)
        elif p['category'] == 'volume_fraction':
            ax.scatter(p['f'], 0.1, c='green', s=100, marker='o', zorder=5)

    # Plot failing configs as red X
    for f in failures:
        if f['category'] == 'material_variation':
            ax.scatter(0.3, f['SR'], c='red', s=150, marker='X', zorder=5)
        elif f['category'] == 'volume_fraction':
            ax.scatter(f['f'], 0.1, c='red', s=150, marker='X', zorder=5)

    # Add annotations for key failures
    ax.annotate('SR=5.0\n(10.1% error)', xy=(0.3, 5.0), xytext=(0.35, 4.5),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
    ax.annotate('f=0.5\n(13.1% error)', xy=(0.5, 0.1), xytext=(0.45, 0.5),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlabel('Volume Fraction (f)', fontsize=12)
    ax.set_ylabel('Stiffness Ratio (SR)', fontsize=12)
    ax.set_title('Training Coverage vs Failure Regions', fontsize=14)
    ax.set_xlim(0.05, 0.55)
    ax.set_ylim(0, 6)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    output_path = output_dir / 'training_coverage.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training coverage plot saved to: {output_path}")

    return output_path


def main():
    """Main function to run failure analysis."""
    print("\n" + "=" * 80)
    print("STARTING FAILURE ANALYSIS")
    print("=" * 80)

    # Load results
    data = load_results()

    # Extract failures
    failures, passes = extract_failing_configs(data, threshold=5.0)

    # Print report
    print_failure_report(failures, passes)

    # Create visualizations
    output_dir = Path(__file__).parent
    create_visualizations(failures, passes, output_dir)
    create_training_coverage_plot(failures, passes, output_dir)

    # Save failure details to JSON
    failure_report = {
        'total_configs': len(failures) + len(passes),
        'pass_count': len(passes),
        'fail_count': len(failures),
        'pass_rate': len(passes) / (len(failures) + len(passes)),
        'failures': failures,
        'summary': {
            'volume_fraction_failures': len([f for f in failures if f['category'] == 'volume_fraction']),
            'material_variation_failures': len([f for f in failures if f['category'] == 'material_variation']),
            'phi_variation_failures': len([f for f in failures if f['category'] == 'phi_variation']),
            'interpolation_failures': len([f for f in failures if f['category'] == 'interpolation_weights']),
        },
        'recommendations': [
            'Add more training samples with f > 0.35',
            'Add more training samples with SR > 2',
            'Include uniform phi configurations in training',
            'Consider clamping volume fraction to safe range',
        ]
    }

    report_path = output_dir / 'failure_report.json'
    with open(report_path, 'w') as f:
        json.dump(failure_report, f, indent=2)

    print(f"\nFailure report saved to: {report_path}")

    return failures, passes


if __name__ == "__main__":
    failures, passes = main()
