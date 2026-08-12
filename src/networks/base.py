# SPDX-License-Identifier: MIT
from abc import ABC, abstractmethod
import torch.nn as nn


class BaseNetwork(nn.Module, ABC):
    """Define the common interface for PIRFlow neural networks."""

    def __init__(self, input_dim, output_dim):
        """Store the network feature dimensions.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        output_dim : int
            Number of predicted features.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, X, use_dropout=False):
        """Evaluate the neural network.

        Parameters
        ----------
        X : torch.Tensor
            Input coordinates, usually with shape [N, 2].

        use_dropout : bool, optional
            Whether to activate dropout.

        Returns
        -------
        torch.Tensor
            Network output with shape ``(N, output_dim)``.
        """
        pass

    def _get_activation(self, activation):
        """Create an activation module from its configured name.

        Parameters
        ----------
        activation : str
            Activation name.

        Returns
        -------
        torch.nn.Module
            Instantiated activation module.
        """

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
