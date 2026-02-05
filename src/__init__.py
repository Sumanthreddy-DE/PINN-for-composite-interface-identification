"""
PINN for Composite Interface Parameter Identification

This package contains the source code for a Physics-Informed Neural Network
that bridges 3-Layer Interphase and Extended Interface models for composite
material homogenization.

Modules:
    physics: Physics models (Extended Interface, 3-Layer Interphase, BVP constraints)
    models: Neural network architectures (PINN, MLP)
    data: Data generation and handling
    training: Training scripts
"""

from . import physics
from . import models
from . import data
from . import training

__version__ = "2.0.0"
__author__ = "Sumanth Reddy Settipalli"
