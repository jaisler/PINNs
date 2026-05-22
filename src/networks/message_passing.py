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
    ):
        print("pass")
