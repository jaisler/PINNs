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
        latent_dim=32,
        activation="tanh",
        neighbors=8,
        message_layers=4,
        aggregation="sum",
        residual=True,
        use_boundary_marker=False, 
        use_edge_distance=False,
    ):
        super().__init__(
            # Base class initialisation
            input_dim=node_input_dim,
            output_dim=output_dim,
        )

        # Use boundary marker - node attributes
        self.use_boundary_marker = use_boundary_marker
        # Use edge distance -  edge attributes
        self.use_edge_distance = use_edge_distance

        # Number of neighbors of each node
        self.neighbors = neighbors
        # Encoder info
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.latent_dim = latent_dim
        self.activation = activation    
        # self.node_encoder: MLP object
        # h: tensor of shape (N, latent_dim), produced later in forward(...)
        self.node_encoder = MLP(
            layers=[node_input_dim, latent_dim, latent_dim],
            activation=activation,
        )

        # self.edge_encoder: MLP object
        # g: tensor of shape (E, latent_dim), produced later in forward(...)
        self.edge_encoder = MLP(
            layers=[edge_input_dim, latent_dim, latent_dim],
            activation=activation
        )

        # Processor info
        self.message_layers = message_layers
        self.aggregation = aggregation
        self.residual = residual
        # Class object
        self.processor = MessagePassingLayer(
            latent_dim=latent_dim,
            activation=activation,
            message_layers=message_layers, 
            aggregation=aggregation,
            residual=residual
        )

        # Decoder info
        self.output_dim = output_dim
        # Class object
        self.decoder = MLP(
            layers=[latent_dim, latent_dim, output_dim],
            activation=activation
        )    

    def forward(self, X, use_dropout=False, boundary_marker=None):

        # Build graph - rebuild every call
        edge_index = self.build_knn_graph(X)

        # Build edge attributes
        edge_attr = self.build_edge_attr(X, edge_index)

        Y = []

        return Y

    def build_edge_attr(self, X, edge_index):

        receiver = edge_index[0]
        sender = edge_index[1]

        x_i = X[receiver]
        x_j = X[sender]

        # Relative position
        relative_position = x_i - x_j

        edge_features = [relative_position]

        if self.use_edge_distance:
            distance = torch.norm(relative_position, dim=1, keepdim=True)
            edge_features.append(distance)

        print(edge_features[0][0])

        edge_attr = torch.cat(edge_features, dim=1)

        return edge_attr

    def build_knn_graph(self, X):
        """
        Build a k-nearest-neighbor graph from point coordinates.

        Parameters
        ----------
        X : torch.Tensor
            Node coordinates with shape (N, d).

        Returns
        -------
        edge_index : torch.Tensor
            Edge list with shape (2, N*k).
            edge_index[0] contains receiver nodes i.
            edge_index[1] contains sender nodes j.
        """

        N = X.shape[0]
        k = self.neighbors

        with torch.no_grad():
            X_graph = X.detach()

            # Pairwise distances: (N, N)
            dist = torch.cdist(X_graph, X_graph)

            # Avoid self-neighbor
            dist.fill_diagonal_(float("inf"))

            # knn_idx[i] gives the k nearest neighbors of node i
            knn_idx = torch.topk(dist, k=k, largest=False).indices

            # Receiver nodes
            receivers = torch.arange(N, device=X.device).repeat_interleave(k)

            # Sender nodes
            senders = knn_idx.reshape(-1) # flattens the matrix into one long vector

            # Each column represents one edege: receiver, sender.
            edge_index = torch.stack([receivers, senders], dim=0)

            print(edge_index)


        return edge_index