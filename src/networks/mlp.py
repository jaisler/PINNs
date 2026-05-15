import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNetwork

class MLP(BaseNetwork):
    """
    Fully connect neural network.
    """

    def __init__(
        self,
        layers,
        activation="tanh",
        dropout_p=0.0,
        dropout_indices=None,
    ):
        super().__init__(
            # Base class initialisation
            input_dim=layers[0],
            output_dim=layers[-1],
        )
        
        # Input layer, hidden layers and output layer
        self.layers = layers
        # Dropout probability
        self.dropout_p = dropout_p
        self.dropout_indices = dropout_indices or []

        self.activation = self._get_activation(activation)

    def _get_activation(self, activation):

        if activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "silu":
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")        

    def forward(self, X, use_dropout=False):

        return True