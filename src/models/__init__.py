"""
Neural network architectures for interface parameter identification.

This module contains:
    - PINN: Physics-Informed Neural Network for inverse problem
    - InverseMLPModel: Basic MLP for comparison
    - EarlyStopping: Training utility
    - get_device: Utility to detect available device (CPU/GPU)
"""

from .pinn import PINN, InverseMLPModel, EarlyStopping, get_device

__all__ = [
    "PINN",
    "InverseMLPModel",
    "EarlyStopping",
    "get_device",
]
