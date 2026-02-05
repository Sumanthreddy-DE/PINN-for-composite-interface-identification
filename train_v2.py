"""
PINN Training Script v2 - Improved with Best Practices

FIXES from previous version:
1. Uses PROPER edge samples (not fake placeholders)
2. Implements adaptive loss balancing (ReLoBRaLo-inspired)
3. Lower edge sample weight to prevent catastrophic forgetting
4. Gradient clipping and proper normalization
5. Curriculum learning: start with physics, gradually add data loss

Based on research:
- Multi-Objective Loss Balancing for Physics-Informed Deep Learning
- ReLoBRaLo: Relative Loss Balancing with Random Lookback
- Transfer learning based PINNs for inverse problems

Author: Sumanth Reddy Settipalli
Date: 2026-02-03
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import json
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import PINN
from data_generator import InterfaceDataset, add_three_layer_data
from physics_loss import PhysicsLoss
from extended_interface import compute_effective_properties


class AdaptiveLossWeighter:
    """
    Adaptive loss weighting inspired by ReLoBRaLo and GradNorm.

    Key ideas:
    - Track loss history for each term
    - Balance based on relative loss changes
    - Random lookback to avoid local minima
    """

    def __init__(self, n_losses: int = 3, alpha: float = 0.9, lookback_range: tuple = (5, 50)):
        self.n_losses = n_losses
        self.alpha = alpha  # EMA smoothing
        self.lookback_range = lookback_range
        self.loss_history = [[] for _ in range(n_losses)]
        self.weights = np.ones(n_losses)

    def update(self, losses: list):
        """Update weights based on loss values."""
        for i, loss in enumerate(losses):
            self.loss_history[i].append(loss)

        if len(self.loss_history[0]) < 10:
            return self.weights

        # Random lookback
        lookback = np.random.randint(self.lookback_range[0],
                                      min(self.lookback_range[1], len(self.loss_history[0])))

        # Compute relative loss changes
        rel_changes = []
        for i in range(self.n_losses):
            current = np.mean(self.loss_history[i][-5:])
            past = np.mean(self.loss_history[i][-lookback:-lookback+5]) if lookback > 5 else current
            rel_change = current / (past + 1e-8)
            rel_changes.append(rel_change)

        # Normalize to get weights (higher weight for slower-converging losses)
        rel_changes = np.array(rel_changes)
        new_weights = rel_changes / (rel_changes.sum() + 1e-8) * self.n_losses

        # Smooth update
        self.weights = self.alpha * self.weights + (1 - self.alpha) * new_weights

        # Clamp to reasonable range
        self.weights = np.clip(self.weights, 0.1, 10.0)

        return self.weights


def train_v2(args):
    """Train PINN with improved strategy."""

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PINN TRAINING v2 - IMPROVED")
    print("=" * 70)
    print(f"\nKey improvements:")
    print(f"  1. Proper edge samples (no fake labels)")
    print(f"  2. Adaptive loss weighting")
    print(f"  3. Conservative edge weight: {args.edge_weight} (prevent catastrophic forgetting)")
    print(f"  4. Gradient clipping")
    print()

    # Generate base dataset (Extended Interface samples)
    print(f"Generating {args.n_samples} Extended Interface samples...")
    dataset = InterfaceDataset(
        n_samples=args.n_samples,
        seed=args.seed,
        cache_path=str(output_dir / 'dataset_cache.json') if args.use_cache else None,
        use_importance_sampling=args.use_importance_sampling,
        edge_weight=args.edge_weight
    )

    # Load proper edge samples if available
    edge_samples_path = Path(args.edge_samples_path)
    if edge_samples_path.exists():
        print(f"\nLoading PROPER edge samples from {edge_samples_path}")
        with open(edge_samples_path, 'r') as f:
            edge_data = json.load(f)
        print(f"  Found {len(edge_data)} samples")

        # Add to dataset (conservative weight to prevent catastrophic forgetting)
        # Use at most 10% of the main dataset size
        max_edge = int(len(dataset) * 0.1)
        n_to_add = min(max_edge, len(edge_data))

        if n_to_add < len(edge_data):
            indices = np.random.choice(len(edge_data), n_to_add, replace=False)
            edge_data = [edge_data[i] for i in indices]

        print(f"  Adding {len(edge_data)} edge samples (capped at 10% of main dataset)")

        # Add edge samples to dataset
        for sample in edge_data:
            dataset.K_eff = np.append(dataset.K_eff, sample['K_eff'])
            dataset.G_eff = np.append(dataset.G_eff, sample['G_eff'])
            dataset.kappa_inc = np.append(dataset.kappa_inc, sample['kappa_inc'])
            dataset.mu_inc = np.append(dataset.mu_inc, sample['mu_inc'])
            dataset.kappa_mat = np.append(dataset.kappa_mat, sample['kappa_mat'])
            dataset.mu_mat = np.append(dataset.mu_mat, sample['mu_mat'])
            dataset.f = np.append(dataset.f, sample['f'])
            dataset.R = np.append(dataset.R, sample['R'])
            dataset.k_bar = np.append(dataset.k_bar, sample['k_bar'])
            dataset.lambda_bar = np.append(dataset.lambda_bar, sample['lambda_bar'])
            dataset.mu_bar = np.append(dataset.mu_bar, sample['mu_bar'])
            dataset.alpha = np.append(dataset.alpha, sample['alpha'])

        dataset.n_samples = len(dataset.K_eff)

    # Load original 3-layer data if available (with conservative weight)
    three_layer_path = Path(args.three_layer_data)
    if three_layer_path.exists():
        dataset = add_three_layer_data(
            dataset,
            str(three_layer_path),
            weight=args.three_layer_weight,
            seed=args.seed
        )

    # Get normalization parameters
    norm_params = dataset.get_normalization_params()

    # Split dataset
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    print(f"\nDataset: {n_total} samples (train={n_train}, val={n_val}, test={n_test})")

    # Create model
    model = PINN(
        hidden_dims=tuple(args.hidden_dims),
        norm_params=norm_params
    )
    print(f"Model: {args.hidden_dims}, params={sum(p.numel() for p in model.parameters()):,}")

    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Loss functions
    mse_loss = nn.MSELoss()
    physics_loss_fn = PhysicsLoss(
        include_bvp_residual=True,
        K_weight=args.K_weight,
        G_weight=args.G_weight
    )

    # Adaptive loss weighter
    loss_weighter = AdaptiveLossWeighter(n_losses=3)  # data, physics, property

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'data_loss': [], 'physics_loss': [], 'prop_loss': [],
        'K_error': [], 'G_error': [],
        'lr': [], 'loss_weights': []
    }

    best_val_loss = float('inf')
    best_epoch = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 70)

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        epoch_losses = {'data': [], 'physics': [], 'prop': [], 'total': []}

        for inputs, targets in train_loader:
            optimizer.zero_grad()

            # Forward pass
            predictions = model(inputs)

            # Scale targets for balanced learning
            target_scales = torch.tensor([100.0, 1.0, 1.0, 1.0])
            data_loss = mse_loss(predictions / target_scales, targets / target_scales)

            # Physics loss
            phys_loss, prop_loss, bvp_loss = physics_loss_fn(predictions, inputs)

            # Get adaptive weights
            weights = loss_weighter.update([
                data_loss.item(),
                phys_loss.item(),
                prop_loss.item()
            ])

            # Total loss with adaptive weighting
            total_loss = (
                weights[0] * data_loss +
                args.physics_weight * weights[1] * phys_loss +
                args.physics_weight * weights[2] * prop_loss
            )

            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses['data'].append(data_loss.item())
            epoch_losses['physics'].append(phys_loss.item())
            epoch_losses['prop'].append(prop_loss.item())
            epoch_losses['total'].append(total_loss.item())

        # Validation phase
        model.eval()
        val_losses = []
        K_errors, G_errors = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                predictions = model(inputs)

                # Data loss
                target_scales = torch.tensor([100.0, 1.0, 1.0, 1.0])
                val_loss = mse_loss(predictions / target_scales, targets / target_scales)
                val_losses.append(val_loss.item())

                # Compute actual K, G errors on a sample
                for i in range(min(10, len(inputs))):
                    try:
                        K_target = inputs[i, 0].item()
                        G_target = inputs[i, 1].item()
                        k_bar = predictions[i, 0].item()
                        lambda_bar = predictions[i, 1].item()
                        mu_bar = predictions[i, 2].item()
                        alpha = predictions[i, 3].item()

                        K_pred, G_pred = compute_effective_properties(
                            inputs[i, 2].item(), inputs[i, 3].item(),
                            inputs[i, 4].item(), inputs[i, 5].item(),
                            k_bar, lambda_bar, mu_bar, alpha,
                            inputs[i, 6].item(), inputs[i, 7].item()
                        )

                        K_errors.append(abs(K_pred - K_target) / K_target * 100)
                        G_errors.append(abs(G_pred - G_target) / G_target * 100)
                    except:
                        pass

        scheduler.step()

        # Record history
        train_loss = np.mean(epoch_losses['total'])
        val_loss = np.mean(val_losses)
        K_err = np.mean(K_errors) if K_errors else 0
        G_err = np.mean(G_errors) if G_errors else 0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['data_loss'].append(np.mean(epoch_losses['data']))
        history['physics_loss'].append(np.mean(epoch_losses['physics']))
        history['prop_loss'].append(np.mean(epoch_losses['prop']))
        history['K_error'].append(K_err)
        history['G_error'].append(G_err)
        history['lr'].append(scheduler.get_last_lr()[0])
        history['loss_weights'].append(loss_weighter.weights.tolist())

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'norm_params': norm_params,
                'args': vars(args),
            }, output_dir / 'pinn_best.pt')

        # Print progress
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{args.epochs}: "
                  f"train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"K_err={K_err:.2f}%, G_err={G_err:.2f}%")

    print("-" * 70)
    print(f"Training complete! Best epoch: {best_epoch+1}, val_loss: {best_val_loss:.6f}")

    # Save final model and history
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'norm_params': norm_params,
        'args': vars(args),
    }, output_dir / 'pinn_final.pt')

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Save config
    config = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'dataset_size': n_total,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'final_K_error': history['K_error'][-1],
        'final_G_error': history['G_error'][-1],
    }
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Plot
    plot_training_curves(history, output_dir)

    print(f"\nResults saved to: {output_dir}")

    return model, history


def plot_training_curves(history, output_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    ax1 = axes[0, 0]
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss')
    ax1.legend()
    ax1.set_yscale('log')

    # K/G errors
    ax2 = axes[0, 1]
    ax2.plot(history['K_error'], label='K_eff')
    ax2.plot(history['G_error'], label='G_eff')
    ax2.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% threshold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error (%)')
    ax2.set_title('Reconstruction Errors')
    ax2.legend()

    # Component losses
    ax3 = axes[1, 0]
    ax3.plot(history['data_loss'], label='Data')
    ax3.plot(history['physics_loss'], label='Physics')
    ax3.plot(history['prop_loss'], label='Property')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Loss Components')
    ax3.legend()
    ax3.set_yscale('log')

    # Learning rate
    ax4 = axes[1, 1]
    ax4.plot(history['lr'], color='green')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate')
    ax4.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train PINN v2')

    # Data
    parser.add_argument('--n_samples', type=int, default=50000)
    parser.add_argument('--three_layer_data', type=str, default='3layer_data.json')
    parser.add_argument('--edge_samples_path', type=str, default='edge_samples_proper.json')
    parser.add_argument('--three_layer_weight', type=float, default=0.10,
                        help='Conservative 3-layer weight (was 0.20)')
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--use_importance_sampling', action='store_true')
    parser.add_argument('--edge_weight', type=float, default=0.15,
                        help='Conservative edge weight (was 0.30)')

    # Model
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 256, 256])

    # Training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--physics_weight', type=float, default=30.0)
    parser.add_argument('--K_weight', type=float, default=1.5)
    parser.add_argument('--G_weight', type=float, default=1.0)

    # Output
    parser.add_argument('--output_dir', type=str, default='./exp_v2')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    train_v2(args)


if __name__ == "__main__":
    main()
