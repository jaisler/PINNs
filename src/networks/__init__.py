# src/networks/__init__.py

from .mlp import MLP
from .gnn import GNN
from .message_passing import MessagePassingLayer

__all__ = [
    "MLP",
    "GNN",
    "MessagePassingLayer"
]
