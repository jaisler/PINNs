# SPDX-License-Identifier: MIT
from abc import ABC, abstractmethod
import torch.nn as nn


class BaseNetwork(nn.Module, ABC):
    """
    Base class for neural network architectures used by the PINN.

    Every derived network must map input coordinates X to the predicted
    physical variables.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, X, use_dropout=False):
        """
        Evaluate the neural network.

        Parameters
        ----------
        X : torch.Tensor
            Input coordinates, usually with shape [N, 2].

        use_dropout : bool
            Whether to activate dropout during the forward pass.

        Returns
        -------
        torch.Tensor
            Network output with shape [N, output_dim].
        """
        pass

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
