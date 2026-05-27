# PINN for Composite Interface Parameter Identification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Physics-Informed Neural Network for bridging 3-Layer Interphase and Extended Interface models in composite material homogenization.**

## Overview

This repository contains the neural network implementation for a Master's thesis on micromechanics-based homogenization of composite materials with interphase/interface effects.

### The Problem

When modeling composite materials (particles embedded in a matrix), the **interphase region** between inclusion and matrix significantly affects overall properties. Two modeling approaches exist:

1. **3-Layer Interphase Model**: Models interphase as finite-thickness coating layers with distinct material properties
2. **Extended Interface Model**: Models interphase as zero-thickness interface with parameters (k&#772;, &#955;&#772;, &#956;&#772;, &#945;)

### Solution: PINN Bridge

This PINN finds Extended Interface parameters that produce the **same effective properties** (K_eff, G_eff) as a given 3-Layer configuration:

```
+------------------------+         +----------+         +------------------------+
|  3-Layer Interphase    |         |          |         |  Extended Interface    |
|  Model                 |-------->|   PINN   |-------->|  Model                 |
|  K_eff, G_eff          |         |          |         |  k_bar, lambda_bar,    |
+------------------------+         +----------+         |  mu_bar, alpha         |
                                                        +------------------------+
```

## Results

Evaluated on **28 held-out validation configurations** within the parameter ranges below:

| Metric | Value |
|--------|-------|
| Pass rate (error ≤ 5%) | 28 / 28 |
| K_eff error mean | 1.77% |
| K_eff error max | 4.87% |
| G_eff error mean | 1.87% |
| G_eff error max | 4.94% |

> **Scope:** These results hold inside the validated ranges listed below. Accuracy drops on wider parameter sweeps (larger configuration sets), so the model is calibrated for these ranges rather than generalized beyond them.

### Validated Parameter Ranges

| Parameter | Range |
|-----------|-------|
| Volume fraction (f) | 0.1 - 0.5 |
| Stiffness ratio (SR) | 0.05 - 5.0 |
| Phi values | 0.05 - 0.9 |

## Installation

```bash
# Clone repository
git clone https://github.com/Sumanthreddy-DE/PINN-for-composite-interface-identification.git
cd PINN-for-composite-interface-identification

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Using the Pre-trained Model

```python
import torch
from src.models.pinn import PINN
from src.physics.extended_interface import compute_effective_properties
from src.physics.three_layer_interphase import compute_3layer_properties

# Load V2 model (best-performing checkpoint)
checkpoint = torch.load('checkpoints/v2/pinn_best.pt', weights_only=False)
model = PINN(hidden_dims=(256, 256, 256), norm_params=checkpoint.get('norm_params'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define material properties
kappa_inc = 1.73      # Particle bulk modulus
mu_inc = 0.8          # Particle shear modulus
kappa_mat = 17.33     # Matrix bulk modulus
mu_mat = 8.0          # Matrix shear modulus
f = 0.3               # Volume fraction
phi_1, phi_2, phi_3 = 0.2, 0.3, 0.4  # Interphase layer ratios

# Step 1: Compute target K_eff, G_eff from 3-Layer model
K_target, G_target = compute_3layer_properties(
    kappa_inc, mu_inc, kappa_mat, mu_mat,
    phi_1, phi_2, phi_3, f
)
print(f"3-Layer: K_eff={K_target:.4f}, G_eff={G_target:.4f}")

# Step 2: Use PINN to predict interface parameters
R = 0.00278  # Particle radius
inputs = torch.tensor([[K_target, G_target, kappa_inc, mu_inc, kappa_mat, mu_mat, f, R]])

with torch.no_grad():
    params = model(inputs)
    k_bar, lambda_bar, mu_bar, alpha = params[0].numpy()

print(f"Predicted: k_bar={k_bar:.2f}, lambda_bar={lambda_bar:.4f}, mu_bar={mu_bar:.4f}, alpha={alpha:.4f}")

# Step 3: Verify with Extended Interface model
K_recon, G_recon = compute_effective_properties(
    kappa_inc, mu_inc, kappa_mat, mu_mat,
    k_bar, lambda_bar, mu_bar, alpha, f, R
)
print(f"Extended Interface: K_eff={K_recon:.4f}, G_eff={G_recon:.4f}")
print(f"Errors: K={abs(K_recon-K_target)/K_target*100:.2f}%, G={abs(G_recon-G_target)/G_target*100:.2f}%")
```

### Run the Demo

```bash
python scripts/demo.py
```

## Project Structure

```
PINN-for-composite-interface-identification/
|
+-- README.md                    # This file
+-- LICENSE                      # MIT License
+-- requirements.txt             # Python dependencies
+-- config.yaml                  # Default training configuration
|
+-- src/                         # SOURCE CODE
|   +-- __init__.py
|   |
|   +-- physics/                 # Physics Models
|   |   +-- extended_interface.py    # Extended Interface (CSA) - Mori-Tanaka
|   |   +-- three_layer_interphase.py # 3-Layer Interphase (CSA)
|   |   +-- physics_loss.py          # BVP constraints for PINN
|   |
|   +-- models/                  # Neural Networks
|   |   +-- pinn.py                  # PINN and MLP architectures
|   |
|   +-- data/                    # Data Generation
|   |   +-- generator.py             # Training dataset creation
|   |   +-- generate_3layer.py       # 3-Layer reference samples
|   |   +-- generate_edge_samples.py # Edge sample generation
|   |
|   +-- training/                # Training Scripts
|       +-- train.py                 # Original training script
|       +-- train_v2.py              # V2 with adaptive loss (RECOMMENDED)
|
+-- scripts/                     # Utility Scripts
|   +-- demo.py                      # Quick demonstration
|   +-- predict.py                   # Prediction testing
|   +-- validate.py                  # Validation suite
|   +-- test_phi_variations.py       # Phi sensitivity tests
|   +-- failure_analysis.py          # Failure analysis
|
+-- examples/                    # Example Notebooks
|   +-- quickstart.ipynb             # Getting started guide
|
+-- data/                        # Data Files
|   +-- 3layer_data.json             # 3-Layer reference samples
|   +-- edge_samples_proper.json     # Edge region samples
|
+-- checkpoints/                 # Trained Models
|   +-- v2/                          # Best model (28/28 on validation set)
|       +-- pinn_best.pt
|       +-- pinn_final.pt
|       +-- training_config.json
|       +-- training_history.json
|
+-- experiments/                 # Archived Experiments
|   +-- archive/
|       +-- exp4_larger_balanced/    # Previous best (78.6%)
|
+-- cca/                         # Future: Fiber Composites
    +-- README.md                    # Placeholder
```

## How to Navigate

| Task | Files to Look At |
|------|------------------|
| **Use the trained model** | `scripts/demo.py`, `checkpoints/v2/pinn_best.pt` |
| **Understand the physics** | `src/physics/extended_interface.py`, `src/physics/three_layer_interphase.py` |
| **Train from scratch** | `src/training/train_v2.py`, `src/data/generator.py` |
| **Validate results** | `scripts/validate.py`, `scripts/predict.py` |
| **Generate training data** | `src/data/generate_3layer.py`, `src/data/generate_edge_samples.py` |

## Training

### Train V2 Model (Recommended)

```bash
# Step 1: Generate edge samples (optional - already included)
python -m src.data.generate_edge_samples --n_edge 10000 --n_3layer_like 5000

# Step 2: Train
python -m src.training.train_v2 \
    --n_samples 50000 \
    --epochs 500 \
    --output_dir ./checkpoints/v2 \
    --edge_samples_path data/edge_samples_proper.json \
    --three_layer_data data/3layer_data.json
```

### Key V2 Improvements

| Aspect | Original | V2 |
|--------|----------|-----|
| Edge samples | Fake labels | Real computed values |
| Edge weight | 30% | 10% (capped) |
| Loss weighting | Fixed | Adaptive (ReLoBRaLo-inspired) |
| Gradient clipping | No | Yes |
| **Pass rate** (validation set) | 78.6% | 28 / 28 |

## Key Contributions

1. **First PINN application** for interphase-interface parameter bridging
2. **Strong accuracy** across the validated configuration range
3. **Adaptive loss balancing** (ReLoBRaLo-inspired) for stable training
4. **Physics-constrained learning** embedding Mori-Tanaka BVP equations

## References

- Firooz, S. et al. (2022). Extended general interfaces: Mori-Tanaka homogenization. *Int. J. Solids Structures*
- Raissi, M. et al. (2019). Physics-informed neural networks. *J. Computational Physics*

## Citation

If you use this code, please cite:

```bibtex
@thesis{settipalli2026pinn,
  title={Physics-Informed Neural Networks for Composite Interface Parameter Identification},
  author={Settipalli, Sumanth Reddy},
  year={2026},
  school={University}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.

---

*Part of Master's Thesis: Micromechanics-based Homogenization of Composite Materials*
