# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNetwork

class MLP(BaseNetwork):
    """
    Fully connect neural network.

    The input coordinates are assumed to be already normalized before
    being passed to this network.
    """

    def __init__(
        self,
        layers,
        activation="tanh",
        dropout_p=0.0,
        dropout_indices=None,
    ):
        """Build an MLP from layer widths and activation settings."""
        super().__init__(
            # Base class initialisation
            input_dim=layers[0],
            output_dim=layers[-1],
        )

        # Input layer, hidden layers and output layer
        self.layers = layers
        
        # Control data dropout
        self.enable_data_dropout = False
        # Check dropout probability
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must satisfy 0.0 <= dropout_p < 1.0")
        # Dropout probability
        self.dropout_p = dropout_p
        # Dropout indices
        self.dropout_indices = dropout_indices or []

        # Get activation function
        self.activation = self._get_activation(activation)

        # Initialise NN - weights and biases
        self.initialise_nn()

    def initialise_nn(self):
        """
        Build the fully connected layers.
        """

        # Fully connected layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(self.layers) - 1):
            self.hidden_layers.append(
                nn.Linear(self.layers[i], self.layers[i + 1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize linear-layer weights and biases.
        """
       
        if isinstance(m, nn.Linear):
            # Xavier initialization
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)    
    

    def forward(self, X, use_dropout=False):
        """
        Forward pass.

        The input X is assumed to be already normalized.
                    
        Dropout:
            If use_dropout=True, dropout is applied to hidden-layer activations.
            If use_dropout=False, dropout is disabled.

        Dropout is not applied to the output layer.

        Parameters
        ----------
        X : torch.Tensor
            Normalized input coordinates.

        use_dropout : bool
            If True, apply dropout to selected hidden layers.

        Returns
        -------
        torch.Tensor
            Raw network output.
        """

        # Scale inputs to [-1, 1]
        #H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        H = X

        # Hidden layers with activation
        for i, layer in enumerate(self.hidden_layers[:-1]):
                                  
            H = self.activation(layer(H))

            # Apply dropout only in selected hidden layers
            if i in self.dropout_indices and self.dropout_p > 0.0:
                H = F.dropout(H, p=self.dropout_p, training=use_dropout)

        # Last layer without activation: linear
        Y = self.hidden_layers[-1](H)

        return Y
