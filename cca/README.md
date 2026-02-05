# CCA (Fiber Composites) - Planned Implementation

This directory is reserved for the **CCA (Composite Cylinder Assemblage)** implementation which will extend the PINN approach to **fiber-reinforced composites**.

## Status: 🚧 Not Started

## Planned Work

### Phase 1: Port MATLAB Models
- [ ] Port `Three_Layered_Interphase_2D.m` to Python
- [ ] Port `Mori_Tanaka_Extended_2D.m` to Python
- [ ] Add transversely isotropic effective properties (K_tr, G_tr, G_ax)

### Phase 2: Train PINN
- [ ] Adapt data generator for 4 BVPs (vs 2 in CSA)
- [ ] Modify physics loss for 2D equations
- [ ] Train and validate model

### Key Differences from CSA

| Aspect | CSA (Particles) | CCA (Fibers) |
|--------|-----------------|--------------|
| Geometry | Spherical | Cylindrical |
| Symmetry | Isotropic (3D) | Transversely Isotropic (2D) |
| BVPs | 2 | 4 |
| Outputs | K_eff, G_eff | K_tr, G_tr, G_ax |

## Reference Files

MATLAB implementations can be found at:
- `../../Interphase/Three_Layered_Interphase_2D.m`
- `../../interface/fiber/Mori_Tanaka_Extended_2D.m`

---

*Target: Complete after CSA thesis chapter is finalized*
