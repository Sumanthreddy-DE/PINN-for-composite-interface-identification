# PINN for Composite Interface Parameter Identification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

**Physics-informed neural network that identifies the four parameters of the Extended General Interface Model (EGIM) directly from target effective properties computed via a three-layer interphase model.**

Code accompanying the Master's thesis *"Bridging Interphase and Interface Models for
Composites"* (Chair of Applied Mechanics, FAU Erlangen-Nürnberg). All numbers below are
taken from Chapter 4 of the thesis. Thesis source:
[Sumanthreddy-DE/Master-Thesis](https://github.com/Sumanthreddy-DE/Master-Thesis).

## Overview

The mechanical behaviour of a composite depends strongly on the thin **interphase** region
between inclusion and matrix. Modelling that region explicitly is expensive, so it is
usually replaced by a **zero-thickness interface** described by a few phenomenological
parameters. Recovering those parameters from known interphase properties is a non-trivial
inverse problem — that is what this network solves.

### The two models

1. **Three-layer interphase model** — the interphase is a stack of finite-thickness coating
   layers, each with its own moduli. Accurate, expensive.
2. **Extended General Interface Model (EGIM)** — the interphase collapses to a zero-thickness
   interface carrying four parameters. Cheap, but the parameters are not directly measurable.

| EGIM parameter | Meaning |
|----------------|---------|
| `k_bar` (k̄) | Interface normal stiffness |
| `lambda_bar` (λ̄) | Surface Lamé parameter |
| `mu_bar` (μ̄) | Surface shear modulus |
| `alpha_bar` (ᾱ) | Distance (interface position) parameter, ∈ [0, 1] |

### The bridge

The inverse PINN takes the effective moduli produced by the three-layer model and predicts
the EGIM parameters that reproduce them. A physics-informed loss passes the prediction back
through the EGIM forward model at every training step, so the network is scored on whether
it actually reconstructs the target properties — not merely on parameter regression.

```
+------------------------+         +----------+         +------------------------+
|  3-Layer Interphase    |         |          |         |  Extended General      |
|  Model                 |-------->|   PINN   |-------->|  Interface Model       |
|  K_eff, G_eff          |         |          |         |  k_bar, lambda_bar,    |
+------------------------+         +----------+         |  mu_bar, alpha_bar     |
                                                        +------------------------+
```

This repository covers the **CSA geometry** (particle-reinforced / spherical composites),
which is the case where the bridge works. The CCA geometry (fibre-reinforced) is covered in
the thesis and fails structurally — see [Scope and limitations](#scope-and-limitations).

## Model

Shared architecture for both geometries (thesis §3, *Inverse Parameter Identification
Network*):

| Aspect | Value |
|--------|-------|
| Hidden layers | 3 x 256, tanh activation |
| Initialisation | Xavier normal |
| Parameters | ~135,000 |
| Inputs (CSA) | Target `K_eff`, `G_eff` + constituent moduli (κ⁽¹⁾, μ⁽¹⁾, κ⁽²⁾, μ⁽²⁾) + volume fraction `f` + inclusion radius |
| Outputs | `k_bar`, `lambda_bar`, `mu_bar`, `alpha_bar` |

Physical constraints are enforced on the output layer rather than through penalties:

- **softplus** on `lambda_bar` and `mu_bar` — non-negativity of surface elastic moduli
- **exponential**, clamped to [-2, 8.5] → `k_bar` ∈ [1, 5000] — spans several orders of magnitude
- **sigmoid** on `alpha_bar` — bounds it to [0, 1]

Tanh is used because the targets are bounded and continuous; it is smooth everywhere and
zero-centred, which suits both gradient flow and the bounded output space.

### Loss

```
L = w_data * L_data + w_phys * L_phys
```

`L_data` is an MSE on the interface parameters (with `k_bar` scaled down by 1000 to balance
its magnitude). `L_phys` is the relative reconstruction error after passing the prediction
through the EGIM forward model:

```
L_phys = (1/N) * sum_i [ w_K * ((K_pred - K_tgt) / K_tgt)^2
                       + w_G * ((G_pred - G_tgt) / G_tgt)^2 ]
```

with `w_K = 1.5`, `w_G = 1.0`. The two terms are balanced adaptively using a
ReLoBRaLo-inspired scheme (Bischof & Kraus, 2021), with the adaptive weights capped at 150
for `w_phys` and 5 for `w_data`.

## Results

Chapter 4 of the thesis evaluates the CSA inverse PINN four ways: two sets of hand-picked
worked examples, a broad random test, and a structured parameter sweep. All four are given
below.

### 1. Worked examples — soft particle (SR = 0.25, f = 0.3)

κ⁽¹⁾ = 5, μ⁽¹⁾ = 3, κ⁽²⁾ = 20, μ⁽²⁾ = 12, layer thickness ratios φ = (0.2, 0.3, 0.4).

| Case | κ: particle → coatings → matrix | K_tgt | K_pred | K err | G_tgt | G_pred | G err |
|------|--------------------------------|-------|--------|-------|-------|--------|-------|
| 1 | 5 → 18.5, 19.3, 19.7 → 20 | 19.651 | 19.335 | 1.61% | 11.957 | 11.915 | 0.35% |
| 2 | 5 → 8.75, 12.5, 16.3 → 20 | 17.895 | 17.881 | 0.08% | 11.919 | 11.354 | 4.74% |
| 3 | 5 → 5.75, 7.25, 8.75 → 20 | 15.234 | 15.735 | 3.29% | 11.286 | 10.080 | **10.68%** |
| 4 | 5 → 100, 10, 1 → 20 | 11.687 | 12.559 | **7.47%** | 12.286 | 8.177 | **33.44%** |
| 5 | 5 → 1, 10, 100 → 20 | 25.085 | 31.028 | **23.69%** | 21.751 | 19.398 | **10.81%** |

Cases 1–3 are interpolated coatings (properties between particle and matrix); **two of the
three pass the 5% threshold**. Case 3 fails because the coating sits very close to the
particle, creating a sharp stiffness jump at the coating–matrix boundary that a
zero-thickness interface struggles to represent. Cases 4–5 are out-of-distribution coatings
(κ = 100, beyond both constituents) and fail badly.

### 2. Worked examples — stiff particle (SR = 2.0, f = 0.3)

κ⁽¹⁾ = 40, μ⁽¹⁾ = 24, κ⁽²⁾ = 20, μ⁽²⁾ = 12, same f and layer thicknesses.

| Case | κ: particle → coatings → matrix | K_tgt | K_pred | K err | G_tgt | G_pred | G err |
|------|--------------------------------|-------|--------|-------|-------|--------|-------|
| 1 | 40 → 22.0, 21.0, 20.4 → 20 | 20.304 | 20.287 | 0.08% | 12.085 | 12.087 | 0.01% |
| 2 | 40 → 35.0, 30.0, 25.0 → 20 | 21.951 | 21.913 | 0.17% | 12.686 | 12.514 | 1.35% |
| 3 | 40 → 39.0, 37.0, 35.0 → 20 | 23.667 | 23.629 | 0.16% | 14.083 | 14.087 | 0.03% |
| 4 | 40 → 100, 10, 1 → 20 | 11.690 | 10.188 | **12.85%** | 1.807 | 6.528 | **261.31%** |
| 5 | 40 → 1, 10, 100 → 20 | 25.096 | 26.537 | **5.74%** | 16.712 | 16.226 | 2.91% |

All three interpolated cases pass comfortably (largest error 1.35%). The out-of-distribution
cases fail in the same pattern as above — Case 4 produces a G_eff error above **260%**.

**Takeaway from both tables:** the failure mode is *data coverage*, not model structure. For
in-distribution configurations the EGIM is structurally adequate for spherical composites and
the PINN learns an accurate inverse mapping. It does not extrapolate to coatings stiffer or
softer than both constituents.

### 3. Broad statistical validation — 500 random configurations

500 three-layer configurations drawn at random from the training parameter space, pushed
through the PINN, then forward-evaluated through the EGIM:

| Metric | K_eff | G_eff |
|--------|-------|-------|
| Median error | 1.4% | 4.0% |
| Mean error | 4.2% | 10.0% |
| Pass rate (error ≤ 5%) | **82%** | **56%** |

> Bulk modulus reconstruction is reliable across the parameter space. The shear modulus is
> the harder target: its *median* error sits below the 5% threshold, so most randomly sampled
> configurations do predict well — but **44% of draws still miss it**. The outliers correspond
> to extreme stiffness ratios and unusual interphase profiles near the boundaries of the
> training domain.
>
> This is the honest generalization number for the model. Treat it as calibrated for a region
> of the parameter space, not general across it.

### 4. Structured parameter sweep — 36 configurations

A 6x6 grid over stiffness ratio SR ∈ {0.1, 0.25, 0.5, 1.0, 1.5, 2.0} and volume fraction
f ∈ {0.05, 0.10, 0.20, 0.30, 0.40, 0.50}, with interphase geometry fixed at
φ = (0.2, 0.3, 0.4) and grading weights (0.75, 0.50, 0.25):

- **33 / 36** configurations pass the 5% threshold for *both* K_eff and G_eff.
- The 3 failures all occur at **SR = 0.1 with f ≥ 0.30** — inclusion much softer than matrix,
  where the interphase grading dominates the effective response.
- Outside that corner, errors stay well below 2% for both moduli.

### Training parameter ranges

The network is only expected to hold inside these ranges. Constituent properties are sampled
uniformly; the ranges are chosen so the stiffness ratio SR = κ⁽¹⁾/κ⁽²⁾ spans well below 1
(soft particle in stiff matrix) through well above 1 (stiff particle in soft matrix).

| Parameter | Range |
|-----------|-------|
| κ⁽¹⁾ (particle bulk modulus) | 0.5 – 100 |
| μ⁽¹⁾ (particle shear modulus) | 0.5 – 50 |
| κ⁽²⁾ (matrix bulk modulus) | 5 – 100 |
| μ⁽²⁾ (matrix shear modulus) | 2 – 50 |
| Volume fraction (f) | 0.1 – 0.5 |
| Layer thickness ratios (φ) | 0.05 – 0.9 |

## Scope and limitations

- **Out-of-distribution coatings are not recovered.** Interphase stiffness profiles lying
  outside both constituents fail, sometimes catastrophically (261% G_eff error). This is a
  data-coverage limit, not a structural one.
- **G_eff is the weak axis.** 56% pass rate on random draws vs 82% for K_eff.
- **Fibre composites (CCA) fail structurally — not included here.** In the thesis, the same
  approach applied to the CCA geometry cannot reproduce the transverse shear modulus G_tr at
  all: only 2% of 500 random configurations pass the 5% threshold, with a mean error of 23%.
  A per-configuration differential-evolution optimiser — free to fit each case independently
  — leaves the same ~23% gap, and a 36-point SR-vs-f sweep passes in only 3 cases, all at
  SR = 0.1 with f ≤ 0.10. The limitation is in the zero-thickness interface model itself,
  not in the network. See thesis Chapter 4. The `cca/` directory here is a placeholder only.

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
+-- LICENSE                      # Apache 2.0 License
+-- requirements.txt             # Python dependencies
+-- config.yaml                  # Default training configuration
|
+-- src/                         # SOURCE CODE
|   +-- __init__.py
|   |
|   +-- physics/                 # Physics Models
|   |   +-- extended_interface.py    # Extended General Interface Model (CSA) - Mori-Tanaka
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
|   +-- v2/                          # Best model (used for all reported results)
|       +-- pinn_best.pt
|       +-- pinn_final.pt
|       +-- training_config.json
|       +-- training_history.json
|
+-- experiments/                 # Archived Experiments
|   +-- archive/
|       +-- exp4_larger_balanced/    # Earlier training run, superseded by v2
|
+-- cca/                         # Fiber Composites (see Scope and limitations)
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

Because only two effective moduli (`K_eff`, `G_eff`) are available to recover four interface
parameters, the CSA training set deliberately combines three sources (thesis §3, *CSA
training considerations*):

| Source | Count | Purpose |
|--------|-------|---------|
| EGIM-generated samples | 100,000 | Cover the full parameter space |
| Edge samples | 5,000 | Concentrated near extreme stiffness ratios and volume fractions |
| 3-layer interphase samples | 500 | Anchor the network to the actual deployment domain |

The third source matters most: it exposes the network to the real interphase-to-interface
mapping during training instead of relying purely on EGIM-generated data.

### Hyperparameters

| Setting | Value |
|---------|-------|
| Optimiser | AdamW, lr = 1e-3, weight decay = 1e-4 |
| Schedule | Cosine annealing |
| Epochs | 500 (best validation loss at epoch 322) |
| Batch size | 256 |
| Gradient clipping | Norm 1.0 |
| Loss weights | `w_K` = 1.5, `w_G` = 1.0 |
| Adaptive weight caps | `w_phys` ≤ 150, `w_data` ≤ 5 |

Full run completes in under one hour on a single CPU core.

> **Note on sample count:** the thesis states 100,000 EGIM samples, whereas the shipped
> checkpoint's `checkpoints/v2/training_config.json` records `n_samples: 50000` and
> `dataset_size: 55500`. Both report `best_epoch: 322`, so this is the same run and the
> discrepancy is unresolved. The command below reproduces the shipped checkpoint.

### Train V2 Model

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

V2 is the checkpoint behind every number in the [Results](#results) section. The original run
is kept in `experiments/archive/` for reference only; it was evaluated under an earlier
protocol and its numbers are not comparable to the ones above.

## Key Contributions

1. **Inverse PINN bridging the two models.** A network that recovers EGIM interface
   parameters from three-layer interphase effective properties, with the forward model
   embedded in the loss so predictions are physically self-consistent.
2. **Reliable reconstruction inside the calibrated region** — 33/36 on the structured
   SR-vs-f sweep, with the failure corner (SR = 0.1, f ≥ 0.30) identified and characterised
   rather than hidden.
3. **Adaptive loss balancing** (ReLoBRaLo-inspired) that keeps the data and physics terms
   from dominating one another during training.
4. **Quantification of the interface–interphase gap.** Establishing that for fibre composites
   the zero-thickness approximation cannot represent transverse shear at all — a negative
   result confirmed independently by a per-configuration optimiser.

## References

- Firooz, S., Chatzigeorgiou, G., Steinmann, P., Javili, A. (2022). Extended general interfaces: Mori-Tanaka homogenization and average fields. *International Journal of Solids and Structures*, 254–255, 111933.
- Firooz, S., Steinmann, P., Javili, A. (2021). Homogenization of composites with extended general interfaces: Comprehensive review and unified modeling. *Applied Mechanics Reviews*, 73(4), 040802.
- Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.
- Bischof, R., Kraus, M. (2021). Multi-objective loss balancing for physics-informed deep learning (ReLoBRaLo). *arXiv:2110.09813*.
- Krishnapriyan, A. S. et al. (2021). Characterizing possible failure modes in physics-informed neural networks. *NeurIPS*, 34, 26548–26560.
- Loshchilov, I., Hutter, F. (2019). Decoupled weight decay regularization (AdamW). *ICLR*.
- Loshchilov, I., Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. *ICLR*.
- Glorot, X., Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*, 249–256.

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{settipalli2026bridging,
  title  = {Bridging Interphase and Interface Models for Composites},
  author = {Settipalli, Sumanth Reddy},
  year   = {2026},
  school = {Friedrich-Alexander-Universit\"{a}t Erlangen-N\"{u}rnberg},
  type   = {Master's thesis},
  note   = {Chair of Applied Mechanics (LTM)}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---

*Master's Thesis: "Bridging Interphase and Interface Models for Composites" — Parameter
Identification in Mechanical Problems. Chair of Applied Mechanics, FAU Erlangen-Nürnberg.*
*Thesis source (LaTeX): [Sumanthreddy-DE/Master-Thesis](https://github.com/Sumanthreddy-DE/Master-Thesis)*
