"""
Training Script for PINN and Baseline MLP

Trains neural networks for interface parameter identification.
Supports both data-driven MLP and physics-informed PINN.

Usage:
    python train.py --model mlp --epochs 200
    python train.py --model pinn --epochs 200 --physics_weight 1.0

Author: Sumanth Reddy Settipalli
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_generator import create_dataloaders, InterfaceDataset
from models import InverseMLPModel, PINN, EarlyStopping, get_device
from physics_loss import PhysicsLoss
from extended_interface import compute_effective_properties


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    physics_loss_fn: PhysicsLoss = None,
    physics_weight: float = 1.0,
    model_type: str = 'mlp'
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_data_loss = 0
    total_physics_loss = 0
    n_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(inputs)

        # Data loss (MSE on parameters - normalized)
        # Normalize targets for better loss scaling
        target_scales = torch.tensor([100.0, 1.0, 1.0, 1.0], device=device)
        data_loss = nn.functional.mse_loss(
            predictions / target_scales,
            targets / target_scales
        )

        # Physics loss (for PINN)
        if model_type == 'pinn' and physics_loss_fn is not None:
            phys_loss, prop_loss, bvp_loss = physics_loss_fn(predictions, inputs)
            loss = data_loss + physics_weight * phys_loss
            total_physics_loss += phys_loss.item()
        else:
            loss = data_loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_data_loss += data_loss.item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'data_loss': total_data_loss / n_batches,
        'physics_loss': total_physics_loss / n_batches if model_type == 'pinn' else 0,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    physics_loss_fn: PhysicsLoss = None,
    model_type: str = 'mlp'
) -> dict:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    total_physics_loss = 0
    all_predictions = []
    all_targets = []
    all_inputs = []
    n_batches = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)

            # Data loss
            target_scales = torch.tensor([100.0, 1.0, 1.0, 1.0], device=device)
            data_loss = nn.functional.mse_loss(
                predictions / target_scales,
                targets / target_scales
            )

            # Physics loss
            if model_type == 'pinn' and physics_loss_fn is not None:
                phys_loss, _, _ = physics_loss_fn(predictions, inputs)
                total_physics_loss += phys_loss.item()

            total_loss += data_loss.item()
            n_batches += 1

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_inputs.append(inputs.cpu())

    predictions = torch.cat(all_predictions, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    inputs = torch.cat(all_inputs, dim=0).numpy()

    # Compute per-parameter metrics
    param_names = ['k_bar', 'lambda_bar', 'mu_bar', 'alpha']
    metrics = {}

    for i, name in enumerate(param_names):
        pred = predictions[:, i]
        tgt = targets[:, i]
        mae = np.abs(pred - tgt).mean()
        rmse = np.sqrt(((pred - tgt) ** 2).mean())
        rel_error = (np.abs(pred - tgt) / (np.abs(tgt) + 1e-8)).mean()

        metrics[f'{name}_mae'] = mae
        metrics[f'{name}_rmse'] = rmse
        metrics[f'{name}_rel_error'] = rel_error

    # Compute reconstruction error (K_eff, G_eff)
    K_errors = []
    G_errors = []

    for i in range(min(len(predictions), 100)):  # Sample for speed
        K_target = inputs[i, 0]
        G_target = inputs[i, 1]
        kappa_inc = inputs[i, 2]
        mu_inc = inputs[i, 3]
        kappa_mat = inputs[i, 4]
        mu_mat = inputs[i, 5]
        f = inputs[i, 6]
        R = inputs[i, 7]

        k_bar = predictions[i, 0]
        lambda_bar = predictions[i, 1]
        mu_bar = predictions[i, 2]
        alpha = predictions[i, 3]

        try:
            K_pred, G_pred = compute_effective_properties(
                kappa_inc, mu_inc, kappa_mat, mu_mat,
                k_bar, lambda_bar, mu_bar, alpha, f, R
            )
            K_errors.append(abs(K_pred - K_target) / K_target)
            G_errors.append(abs(G_pred - G_target) / G_target)
        except:
            pass

    metrics['K_reconstruction_error'] = np.mean(K_errors) if K_errors else float('nan')
    metrics['G_reconstruction_error'] = np.mean(G_errors) if G_errors else float('nan')

    return {
        'loss': total_loss / n_batches,
        'physics_loss': total_physics_loss / n_batches if model_type == 'pinn' else 0,
        **metrics
    }


def train(args):
    """Main training function."""
    print("=" * 60)
    print(f"Training {args.model.upper()} Model")
    print("=" * 60)

    # Setup
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    cache_path = output_dir / 'dataset_cache.json'
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        cache_path=str(cache_path),
        three_layer_path=args.three_layer_data,
        three_layer_weight=args.three_layer_weight
    )

    # Create model
    if args.model == 'pinn':
        model = PINN(
            hidden_dims=tuple(args.hidden_dims),
            activation=args.activation,
            dropout=args.dropout,
            norm_params=norm_params,
            physics_weight=args.physics_weight
        ).to(device)
        physics_loss_fn = PhysicsLoss(
            K_weight=args.K_weight,
            G_weight=args.G_weight
        ).to(device)
    else:
        model = InverseMLPModel(
            hidden_dims=tuple(args.hidden_dims),
            activation=args.activation,
            dropout=args.dropout,
            norm_params=norm_params
        ).to(device)
        physics_loss_fn = None

    print(f"\nModel architecture:")
    print(f"  Hidden layers: {args.hidden_dims}")
    print(f"  Activation: {args.activation}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler - cosine annealing with warm restarts for better convergence
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
        use_step_scheduler = True
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        use_step_scheduler = False

    early_stopping = EarlyStopping(patience=args.patience)

    # Training loop
    history = {
        'train_loss': [], 'val_loss': [],
        'train_data_loss': [], 'train_physics_loss': [],
        'K_reconstruction_error': [], 'G_reconstruction_error': [],
    }
    best_val_loss = float('inf')

    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            physics_loss_fn, args.physics_weight, args.model
        )

        # Validate
        val_metrics = evaluate(model, val_loader, device, physics_loss_fn, args.model)

        # Update scheduler
        if use_step_scheduler:
            scheduler.step(epoch)
        else:
            scheduler.step(val_metrics['loss'])

        # Log history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_data_loss'].append(train_metrics['data_loss'])
        history['train_physics_loss'].append(train_metrics['physics_loss'])
        history['K_reconstruction_error'].append(val_metrics['K_reconstruction_error'])
        history['G_reconstruction_error'].append(val_metrics['G_reconstruction_error'])

        # Print progress
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train: {train_metrics['loss']:.4f} | "
                  f"Val: {val_metrics['loss']:.4f} | "
                  f"K_err: {val_metrics['K_reconstruction_error']:.4f} | "
                  f"G_err: {val_metrics['G_reconstruction_error']:.4f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'norm_params': norm_params,
                'args': vars(args),
            }, output_dir / f'{args.model}_best.pt')

        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(output_dir / f'{args.model}_best.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device, physics_loss_fn, args.model)

    print(f"\nTest Results:")
    print(f"  Data Loss: {test_metrics['loss']:.6f}")
    print(f"  K_eff Reconstruction Error: {test_metrics['K_reconstruction_error']:.4%}")
    print(f"  G_eff Reconstruction Error: {test_metrics['G_reconstruction_error']:.4%}")
    print(f"\nPer-Parameter Metrics:")
    for param in ['k_bar', 'lambda_bar', 'mu_bar', 'alpha']:
        print(f"  {param}: MAE={test_metrics[f'{param}_mae']:.4f}, "
              f"RelErr={test_metrics[f'{param}_rel_error']:.4%}")

    # Save results
    results = {
        'model': args.model,
        'test_metrics': test_metrics,
        'history': history,
        'args': vars(args),
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / f'{args.model}_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    # Plot training curves
    plot_training_history(history, output_dir, args.model)

    print(f"\nResults saved to {output_dir}")
    return model, test_metrics


def plot_training_history(history: dict, output_dir: Path, model_name: str):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curves
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].set_yscale('log')

    # Reconstruction error
    axes[1].plot(history['K_reconstruction_error'], label='K_eff')
    axes[1].plot(history['G_reconstruction_error'], label='G_eff')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Relative Error')
    axes[1].set_title('Reconstruction Error')
    axes[1].legend()

    # Physics loss (if PINN)
    if max(history['train_physics_loss']) > 0:
        axes[2].plot(history['train_physics_loss'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Physics Loss')
        axes[2].set_title('Physics Loss')
        axes[2].set_yscale('log')
    else:
        axes[2].text(0.5, 0.5, 'N/A for MLP', ha='center', va='center',
                     transform=axes[2].transAxes, fontsize=14)
        axes[2].set_title('Physics Loss')

    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_training.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train PINN for interface parameter identification')

    # Model
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'pinn'])
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--dropout', type=float, default=0.0)

    # Data
    parser.add_argument('--n_samples', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--three_layer_data', type=str, default=None,
                        help='Path to 3-Layer data JSON for mixed training')
    parser.add_argument('--three_layer_weight', type=float, default=0.2,
                        help='Fraction of 3-Layer samples in training data')

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--physics_weight', type=float, default=1.0)
    parser.add_argument('--K_weight', type=float, default=1.0, help='Weight for K_eff in physics loss')
    parser.add_argument('--G_weight', type=float, default=1.0, help='Weight for G_eff in physics loss')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine'],
                        help='Learning rate scheduler type')

    # Output
    parser.add_argument('--output_dir', type=str, default='./results')

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
