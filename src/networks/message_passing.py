# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn

from .mlp import MLP

class MessagePassingLayer(nn.Module):
    """
    Message passing: processor stage of the graph neural network algorithm.
    """

    def __init__(
        self,
        latent_dim,
        activation="tanh",
        message_layers=4,
        aggregation="sum",
        residual=True
    ):
        super().__init__()

        message_mlp = MLP(
            layers=[],
            activation=activation
        )


def forward(self, h, g, edge_index, use_dropout=False):
    
    print("pass")
