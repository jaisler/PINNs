# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNetwork
from .mlp import MLP
from .message_passing import MessagePassingLayer

class GNN(BaseNetwork):
    """
    Graph neural network
    
    The input coordinates are assumed to be already normalized before
    being passed to this network.
    """

    def __init__(
        self,
        node_input_dim,
        edge_input_dim,
        output_dim,
        neighbors=8,
        latent_enc_dim=32,
        activation="tanh",
        message_layers=4,
        aggregation="sum",
        residual=True,
        latent_dec_dim=32,
    ):
        super().__init__(
            # Base class initialisation
            input_dim=node_input_dim,
            output_dim=output_dim,
        )

        self.neighbors = neighbors
        # Encoder
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.latent_dim = latent_enc_dim
        self.activation = activation    
        # h: class object
        self.node_encoder = MLP(
            layers=[node_input_dim, latent_enc_dim, latent_enc_dim],
            activation=activation,
        )
        # g: class object
        self.edge_encoder = MLP(
            layers=[edge_input_dim, latent_enc_dim, latent_enc_dim],
            activation=activation
        )

        # Processor
        self.message_layers = message_layers
        self.aggregation = aggregation
        self.residual = residual
        # Class object
        self.processor = MessagePassingLayer(
            latent_dim=latent_enc_dim,
            activation=activation,
            message_layers=message_layers, 
            aggregation=aggregation,
            residual=residual
        )

        # Decoder
        self.output_dim = output_dim
        self.latent_dec_dim = latent_dec_dim
        # Class object
        self.decoder = MLP(
            layers=[latent_dec_dim, latent_dec_dim, output_dim],
            activation=activation
        )    

def forward(self, X, edge_index, edge_attr):


    Y = []
    return Y


def build_knn_graph(x, k):
    """
    Build a k-nearest-neighbor graph from point coordinates.

    Parameters
    ----------
    x : torch.Tensor
        Node coordinates with shape (N, d).
    k : int
        Number of neighbors per node.

    Returns
    -------
    edge_index : torch.Tensor
        Edge list with shape (2, N*k).
        edge_index[0] contains receiver nodes i.
        edge_index[1] contains sender nodes j.
    """
    N = x.shape[0]

    # Pairwise distances: (N, N)
    dist = torch.cdist(x, x)

    # Avoid self-neighbor
    dist.fill_diagonal_(float("inf"))

    # knn_idx[i] gives the k nearest neighbors of node i
    knn_idx = torch.topk(dist, k=k, largest=False).indices

    # Receiver nodes
    receivers = torch.arange(N, device=x.device).repeat_interleave(k)

    # Sender nodes
    senders = knn_idx.reshape(-1) # flattens the matrix into one long vector

    # Each column represents one edege: receiver, sender.
    edge_index = torch.stack([receivers, senders], dim=0)

    return edge_index