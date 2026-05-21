# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNetwork

class GNN(BaseNetwork):
    """
    Graph neural network
    
    The input coordinates are assumed to be already normalized before
    being passed to this network.
    """

    def __init__(
        self,
        input_dim=2,
        edge_dim=3,
        hidden_dim=64,
        output_dim=4
    ):
        super().__init__(
            # Base class initialisation
            input_dim=input_dim,
            output_dim=output_dim,
        )

        print("Pass")

def forward(self, X, edge_index, edge_attr):

    # Encode and edge features
    #h = self.node_encoder(X)          # (N, hidden_dim)
    #g = self.edge_encoder(edge_attr)  # (N, hidden_dim)

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