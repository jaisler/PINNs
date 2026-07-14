# src/pinn/__init.py

from .physics_informed_nn import PhysicsInformedNN
from .factory import build_pinn_model

__all__ = [
        "PhysicsInformedNN",
        "build_pinn_model",
]
