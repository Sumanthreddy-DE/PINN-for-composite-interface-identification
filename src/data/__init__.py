"""
Data generation and handling for PINN training.

This module contains:
    - InterfaceDataset: PyTorch dataset for training
    - create_dataloaders: Utility to create train/val dataloaders
    - Data generation scripts for 3-Layer and edge samples
"""

from .generator import (
    InterfaceDataset,
    create_dataloaders,
    importance_sample,
    stratified_sample,
)

__all__ = [
    "InterfaceDataset",
    "create_dataloaders",
    "importance_sample",
    "stratified_sample",
]
