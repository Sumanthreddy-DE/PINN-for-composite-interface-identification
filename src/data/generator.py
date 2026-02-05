"""
Data Generator for PINN Training

Generates training data by sampling interface parameters and computing
effective properties using the Extended Interface model.

Dataset structure:
- Inputs: K_target, G_target, kappa_inc, mu_inc, kappa_mat, mu_mat, f
- Outputs: k_bar, lambda_bar, mu_bar, alpha

IMPROVEMENTS (2026-02-03):
- Added importance sampling for edge regions (high-f, high-SR)
- Added stratified sampling option
- Better coverage of failure regions identified in analysis

Author: Sumanth Reddy Settipalli
"""

import sys
from pathlib import Path

# Add project root to path for standalone execution
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict
import json
from tqdm import tqdm

from src.physics.extended_interface import compute_effective_properties


def importance_sample(n_samples: int, low: float, high: float,
                      edge_start: float, edge_weight: float = 0.3) -> np.ndarray:
    """
    Sample with higher density in edge regions.

    Args:
        n_samples: Total number of samples
        low: Lower bound
        high: Upper bound
        edge_start: Where the edge region starts
        edge_weight: Fraction of samples in edge region (0-1)

    Returns:
        Array of samples with importance weighting

    Example:
        # For volume fraction f in [0.1, 0.5] with edge at 0.35:
        f_samples = importance_sample(10000, 0.1, 0.5, edge_start=0.35, edge_weight=0.3)
        # This gives 30% of samples in [0.35, 0.5] (edge region)
        # and 70% in [0.1, 0.35] (normal region)
    """
    n_edge = int(n_samples * edge_weight)
    n_normal = n_samples - n_edge

    # Normal region samples
    normal = np.random.uniform(low, edge_start, n_normal)

    # Edge region samples (denser coverage)
    edge = np.random.uniform(edge_start, high, n_edge)

    # Combine and shuffle
    samples = np.concatenate([normal, edge])
    np.random.shuffle(samples)

    return samples


def stratified_sample(n_samples: int, ranges: Dict[str, Tuple[float, float, float]]) -> Dict[str, np.ndarray]:
    """
    Stratified sampling across multiple parameter ranges.

    Args:
        n_samples: Total number of samples
        ranges: Dict mapping param name to (low, mid, high) tuple
                where mid is the boundary between normal and edge regions

    Returns:
        Dict of parameter arrays

    Example:
        ranges = {
            'f': (0.1, 0.35, 0.5),      # 30% in [0.35, 0.5]
            'SR': (0.05, 2.0, 10.0),    # 30% in [2.0, 10.0]
        }
        samples = stratified_sample(10000, ranges)
    """
    result = {}
    for param, (low, mid, high) in ranges.items():
        result[param] = importance_sample(n_samples, low, high, edge_start=mid, edge_weight=0.3)
    return result


class InterfaceDataset(Dataset):
    """
    Dataset for interface parameter identification.

    Each sample contains:
    - Effective properties (K_eff, G_eff) as targets to match
    - Material properties (kappa_inc, mu_inc, kappa_mat, mu_mat, f)
    - Ground truth interface parameters (k_bar, lambda_bar, mu_bar, alpha)
    """

    def __init__(
        self,
        n_samples: int = 20000,
        param_ranges: Optional[dict] = None,
        material_ranges: Optional[dict] = None,
        seed: int = 42,
        cache_path: Optional[str] = None,
        use_importance_sampling: bool = False,
        edge_weight: float = 0.3
    ):
        """
        Args:
            n_samples: Number of samples to generate
            param_ranges: Dict with ranges for interface parameters
            material_ranges: Dict with ranges for material properties
            seed: Random seed for reproducibility
            cache_path: Path to load/save cached dataset
            use_importance_sampling: If True, use importance sampling for edge regions
            edge_weight: Fraction of samples in edge regions (only if importance_sampling=True)
        """
        np.random.seed(seed)
        self.use_importance_sampling = use_importance_sampling
        self.edge_weight = edge_weight

        # Default parameter ranges (from Firooz paper analysis)
        self.param_ranges = param_ranges or {
            'k_bar': (1.0, 500.0),           # Normal stiffness [N/m^3]
            'lambda_bar': (0.01, 5.0),       # Tangential Lame [N/m]
            'mu_bar': (0.01, 5.0),           # Tangential shear [N/m]
            'alpha': (0.0, 1.0),             # Position parameter
        }

        # Default material ranges (soft to stiff inclusions)
        self.material_ranges = material_ranges or {
            'kappa_inc': (0.5, 50.0),        # Inclusion bulk modulus
            'mu_inc': (0.2, 25.0),           # Inclusion shear modulus
            'kappa_mat': (10.0, 30.0),       # Matrix bulk modulus
            'mu_mat': (5.0, 15.0),           # Matrix shear modulus
            'f': (0.1, 0.5),                 # Volume fraction
            'sz': (0.001, 0.01),             # Size parameter
        }

        self.n_samples = n_samples

        # Try to load from cache
        if cache_path and Path(cache_path).exists():
            print(f"Loading cached dataset from {cache_path}")
            self._load_cache(cache_path)
        else:
            print(f"Generating {n_samples} samples...")
            self._generate_data()
            if cache_path:
                self._save_cache(cache_path)

    def _generate_data(self):
        """Generate dataset by sampling parameters and computing effective properties."""
        # Sample interface parameters (log-uniform for k_bar, uniform for others)
        self.k_bar = np.exp(np.random.uniform(
            np.log(self.param_ranges['k_bar'][0]),
            np.log(self.param_ranges['k_bar'][1]),
            self.n_samples
        ))
        self.lambda_bar = np.random.uniform(
            self.param_ranges['lambda_bar'][0],
            self.param_ranges['lambda_bar'][1],
            self.n_samples
        )
        self.mu_bar = np.random.uniform(
            self.param_ranges['mu_bar'][0],
            self.param_ranges['mu_bar'][1],
            self.n_samples
        )
        self.alpha = np.random.uniform(
            self.param_ranges['alpha'][0],
            self.param_ranges['alpha'][1],
            self.n_samples
        )

        # Sample material properties
        # Use importance sampling for f and SR (kappa_inc/kappa_mat ratio) if enabled
        if self.use_importance_sampling:
            print(f"Using importance sampling with edge_weight={self.edge_weight}")
            print("  - Volume fraction f: 70% in [0.1, 0.35], 30% in [0.35, 0.5]")
            print("  - Stiffness: More samples for higher kappa_inc values")

            # Volume fraction with importance sampling
            # Edge region is f >= 0.35 (where failures occurred)
            self.f = importance_sample(
                self.n_samples,
                low=self.material_ranges['f'][0],
                high=self.material_ranges['f'][1],
                edge_start=0.35,
                edge_weight=self.edge_weight
            )

            # Inclusion properties - sample higher values more to get high SR
            # Edge region for kappa_inc: values that give SR > 2 (kappa_inc > 2*kappa_mat_min)
            # With kappa_mat in [10, 30], SR > 2 means kappa_inc > 20
            self.kappa_inc = importance_sample(
                self.n_samples,
                low=self.material_ranges['kappa_inc'][0],
                high=self.material_ranges['kappa_inc'][1],
                edge_start=20.0,  # SR > 2 boundary (approx)
                edge_weight=self.edge_weight
            )
            self.mu_inc = importance_sample(
                self.n_samples,
                low=self.material_ranges['mu_inc'][0],
                high=self.material_ranges['mu_inc'][1],
                edge_start=10.0,  # Corresponding to high SR
                edge_weight=self.edge_weight
            )
        else:
            # Standard uniform sampling (original behavior)
            self.kappa_inc = np.random.uniform(
                self.material_ranges['kappa_inc'][0],
                self.material_ranges['kappa_inc'][1],
                self.n_samples
            )
            self.mu_inc = np.random.uniform(
                self.material_ranges['mu_inc'][0],
                self.material_ranges['mu_inc'][1],
                self.n_samples
            )
            self.f = np.random.uniform(
                self.material_ranges['f'][0],
                self.material_ranges['f'][1],
                self.n_samples
            )

        # Matrix properties (always uniform - these don't cause failures)
        self.kappa_mat = np.random.uniform(
            self.material_ranges['kappa_mat'][0],
            self.material_ranges['kappa_mat'][1],
            self.n_samples
        )
        self.mu_mat = np.random.uniform(
            self.material_ranges['mu_mat'][0],
            self.material_ranges['mu_mat'][1],
            self.n_samples
        )
        sz = np.random.uniform(
            self.material_ranges['sz'][0],
            self.material_ranges['sz'][1],
            self.n_samples
        )
        # Compute radius from volume fraction and size
        self.R = ((3 * self.f) / (4 * np.pi)) ** (1/3) * sz

        # Compute effective properties for each sample
        self.K_eff = np.zeros(self.n_samples)
        self.G_eff = np.zeros(self.n_samples)

        valid_mask = np.ones(self.n_samples, dtype=bool)

        for i in tqdm(range(self.n_samples), desc="Computing effective properties"):
            try:
                K, G = compute_effective_properties(
                    self.kappa_inc[i], self.mu_inc[i],
                    self.kappa_mat[i], self.mu_mat[i],
                    self.k_bar[i], self.lambda_bar[i], self.mu_bar[i],
                    self.alpha[i], self.f[i], self.R[i]
                )
                # Check for valid (positive) moduli
                if K > 0 and G > 0 and np.isfinite(K) and np.isfinite(G):
                    self.K_eff[i] = K
                    self.G_eff[i] = G
                else:
                    valid_mask[i] = False
            except Exception as e:
                valid_mask[i] = False

        # Filter invalid samples
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            print(f"Removing {n_invalid} invalid samples ({n_invalid/self.n_samples*100:.1f}%)")
            self._filter_samples(valid_mask)

        print(f"Dataset size: {len(self.K_eff)} samples")

    def _filter_samples(self, mask):
        """Remove samples that don't satisfy the mask."""
        self.k_bar = self.k_bar[mask]
        self.lambda_bar = self.lambda_bar[mask]
        self.mu_bar = self.mu_bar[mask]
        self.alpha = self.alpha[mask]
        self.kappa_inc = self.kappa_inc[mask]
        self.mu_inc = self.mu_inc[mask]
        self.kappa_mat = self.kappa_mat[mask]
        self.mu_mat = self.mu_mat[mask]
        self.f = self.f[mask]
        self.R = self.R[mask]
        self.K_eff = self.K_eff[mask]
        self.G_eff = self.G_eff[mask]
        self.n_samples = len(self.K_eff)

    def _save_cache(self, path):
        """Save dataset to disk."""
        data = {
            'k_bar': self.k_bar.tolist(),
            'lambda_bar': self.lambda_bar.tolist(),
            'mu_bar': self.mu_bar.tolist(),
            'alpha': self.alpha.tolist(),
            'kappa_inc': self.kappa_inc.tolist(),
            'mu_inc': self.mu_inc.tolist(),
            'kappa_mat': self.kappa_mat.tolist(),
            'mu_mat': self.mu_mat.tolist(),
            'f': self.f.tolist(),
            'R': self.R.tolist(),
            'K_eff': self.K_eff.tolist(),
            'G_eff': self.G_eff.tolist(),
            'param_ranges': self.param_ranges,
            'material_ranges': self.material_ranges,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Saved dataset to {path}")

    def _load_cache(self, path):
        """Load dataset from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.k_bar = np.array(data['k_bar'])
        self.lambda_bar = np.array(data['lambda_bar'])
        self.mu_bar = np.array(data['mu_bar'])
        self.alpha = np.array(data['alpha'])
        self.kappa_inc = np.array(data['kappa_inc'])
        self.mu_inc = np.array(data['mu_inc'])
        self.kappa_mat = np.array(data['kappa_mat'])
        self.mu_mat = np.array(data['mu_mat'])
        self.f = np.array(data['f'])
        self.R = np.array(data['R'])
        self.K_eff = np.array(data['K_eff'])
        self.G_eff = np.array(data['G_eff'])
        self.param_ranges = data['param_ranges']
        self.material_ranges = data['material_ranges']
        self.n_samples = len(self.K_eff)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns:
            inputs: (K_eff, G_eff, kappa_inc, mu_inc, kappa_mat, mu_mat, f, R) - 8 features
            targets: (k_bar, lambda_bar, mu_bar, alpha) - 4 parameters
        """
        inputs = torch.tensor([
            self.K_eff[idx],
            self.G_eff[idx],
            self.kappa_inc[idx],
            self.mu_inc[idx],
            self.kappa_mat[idx],
            self.mu_mat[idx],
            self.f[idx],
            self.R[idx],
        ], dtype=torch.float32)

        targets = torch.tensor([
            self.k_bar[idx],
            self.lambda_bar[idx],
            self.mu_bar[idx],
            self.alpha[idx],
        ], dtype=torch.float32)

        return inputs, targets

    def get_normalization_params(self):
        """Compute normalization parameters for inputs and targets."""
        inputs_all = np.stack([
            self.K_eff, self.G_eff,
            self.kappa_inc, self.mu_inc,
            self.kappa_mat, self.mu_mat,
            self.f, self.R
        ], axis=1)

        targets_all = np.stack([
            self.k_bar, self.lambda_bar, self.mu_bar, self.alpha
        ], axis=1)

        return {
            'input_mean': inputs_all.mean(axis=0),
            'input_std': inputs_all.std(axis=0),
            'target_mean': targets_all.mean(axis=0),
            'target_std': targets_all.std(axis=0),
        }


def create_dataloaders(
    n_samples: int = 20000,
    batch_size: int = 256,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    cache_path: Optional[str] = None,
    three_layer_path: Optional[str] = None,
    three_layer_weight: float = 0.2
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train, validation, and test dataloaders.

    Args:
        n_samples: Total number of samples
        batch_size: Batch size for training
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (rest is test)
        seed: Random seed
        cache_path: Path to cache dataset
        three_layer_path: Path to 3-Layer data JSON (optional)
        three_layer_weight: Fraction of training data to be 3-Layer samples (0-1)

    Returns:
        train_loader, val_loader, test_loader, norm_params
    """
    # Generate or load Extended Interface dataset
    dataset = InterfaceDataset(n_samples=n_samples, seed=seed, cache_path=cache_path)

    # Optionally add 3-Layer data
    if three_layer_path and Path(three_layer_path).exists():
        dataset = add_three_layer_data(dataset, three_layer_path, three_layer_weight, seed)

    norm_params = dataset.get_normalization_params()

    # Split dataset
    n_train = int(len(dataset) * train_ratio)
    n_val = int(len(dataset) * val_ratio)
    n_test = len(dataset) - n_train - n_val

    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test]
    )

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset splits: train={n_train}, val={n_val}, test={n_test}")

    return train_loader, val_loader, test_loader, norm_params


def add_three_layer_data(
    dataset: InterfaceDataset,
    three_layer_path: str,
    weight: float = 0.2,
    seed: int = 42
) -> InterfaceDataset:
    """
    Add 3-Layer data samples to the dataset.

    Args:
        dataset: Existing InterfaceDataset
        three_layer_path: Path to 3-Layer data JSON
        weight: Target fraction of 3-Layer samples in final dataset
        seed: Random seed for sampling

    Returns:
        Modified dataset with 3-Layer samples added
    """
    import json
    np.random.seed(seed)

    with open(three_layer_path, 'r') as f:
        three_layer_data = json.load(f)

    print(f"Loading {len(three_layer_data)} 3-Layer samples from {three_layer_path}")

    # Calculate how many 3-Layer samples to add
    current_size = len(dataset)
    target_3layer = int(current_size * weight / (1 - weight))
    n_to_add = min(target_3layer, len(three_layer_data))

    # Randomly sample if we have more than needed
    if n_to_add < len(three_layer_data):
        indices = np.random.choice(len(three_layer_data), n_to_add, replace=False)
        samples = [three_layer_data[i] for i in indices]
    else:
        samples = three_layer_data

    # Add to dataset arrays
    for sample in samples:
        dataset.K_eff = np.append(dataset.K_eff, sample['K_3layer'])
        dataset.G_eff = np.append(dataset.G_eff, sample['G_3layer'])
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
    print(f"Added {len(samples)} 3-Layer samples. Total dataset size: {dataset.n_samples}")

    return dataset


if __name__ == "__main__":
    # Test data generation
    print("Testing data generator...")

    # Generate small dataset for testing
    dataset = InterfaceDataset(n_samples=1000, seed=42)

    print(f"\nDataset statistics:")
    print(f"  K_eff: [{dataset.K_eff.min():.2f}, {dataset.K_eff.max():.2f}]")
    print(f"  G_eff: [{dataset.G_eff.min():.2f}, {dataset.G_eff.max():.2f}]")
    print(f"  k_bar: [{dataset.k_bar.min():.2f}, {dataset.k_bar.max():.2f}]")
    print(f"  lambda_bar: [{dataset.lambda_bar.min():.4f}, {dataset.lambda_bar.max():.4f}]")
    print(f"  mu_bar: [{dataset.mu_bar.min():.4f}, {dataset.mu_bar.max():.4f}]")
    print(f"  alpha: [{dataset.alpha.min():.2f}, {dataset.alpha.max():.2f}]")

    # Test dataloader
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        n_samples=1000, batch_size=32
    )

    inputs, targets = next(iter(train_loader))
    print(f"\nBatch shapes: inputs={inputs.shape}, targets={targets.shape}")
    print(f"Normalization params computed: {list(norm_params.keys())}")
