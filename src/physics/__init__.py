"""
Physics models for composite material homogenization.

This module contains:
    - Extended Interface model (CSA) for particle composites
    - 3-Layer Interphase model (CSA) for particle composites
    - Physics loss functions for PINN training (BVP constraints)
"""

from .extended_interface import (
    compute_effective_properties,
    solve_volumetric_bvp,
    solve_deviatoric_bvp,
    compute_effective_properties_torch,
)

from .three_layer_interphase import (
    compute_3layer_properties,
    solve_volumetric_bvp_3layer,
    solve_deviatoric_bvp_3layer,
)

from .physics_loss import PhysicsLoss

__all__ = [
    # Extended Interface
    "compute_effective_properties",
    "compute_effective_properties_torch",
    "solve_volumetric_bvp",
    "solve_deviatoric_bvp",
    # 3-Layer Interphase
    "compute_3layer_properties",
    "solve_volumetric_bvp_3layer",
    "solve_deviatoric_bvp_3layer",
    # Physics Loss
    "PhysicsLoss",
]
