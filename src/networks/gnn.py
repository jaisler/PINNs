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
        boundary_marker=None, 
        use_edge_distance=False,
    ):
        super().__init__(
            # Base class initialisation
            input_dim=node_input_dim,
            output_dim=output_dim,
        )

        # Boundary marker - node attributes
        if boundary_marker is not None:
            self.boundary_marker = boundary_marker
        
        # Use edge distance -  edge attributes
        self.use_edge_distance = use_edge_distance

        # Number of neighbors of each node
        if neighbors < 3:
            raise ValueError("Each node should have at least 3 neighbors")
        self.neighbors = neighbors

        # Get activation function
        self.activation = self._get_activation(activation)

        # Encoder info
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.latent_dim = latent_dim
        # self.node_encoder: MLP object
        # h: tensor of shape (N, latent_dim), produced later in forward(...)
        # MLP node encoder
        self.node_encoder = MLP(
            layers=[node_input_dim, latent_dim, latent_dim],
            activation=activation,
        )

        # self.edge_encoder: MLP object
        # g: tensor of shape (E, latent_dim), produced later in forward(...)
        # MLP edge encoder
        self.edge_encoder = MLP(
            layers=[edge_input_dim, latent_dim, latent_dim],
            activation=activation
        )

        # Class object
        # Processor
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
        # MLP decoder
        self.decoder = MLP(
            layers=[latent_dim, latent_dim, output_dim],
            activation=activation
        )    

    def forward(self, X, use_dropout=False):
        """
        Forward pass of the GNN.

        The method builds a k-nearest-neighbor graph from the input nodes,
        computes edge attributes, encodes node and edge features, applies
        message passing, and decodes the final node features into the output
        variables.

        Parameters
        ----------
        X : torch.Tensor
            Input node features with shape (N, node_input_dim), where N is the
            number of nodes. Usually this contains the node coordinates, such as
            [x, y].

        use_dropout : bool, optional
            Whether to use dropout. Default is False.

        Returns
        -------
        Y : torch.Tensor
            Output predictions at the nodes, with shape (N, output_dim).
            For example, this may contain [rho, u, v, p].
        """

        if self.neighbors >= X.shape[0]:
            raise ValueError(
                f"neighbors must be smaller than the number of nodes. "
                f"Got neighbors={self.neighbors}, number_of_nodes={X.shape[0]}"
            )

        # Node attributes
        node_attr = X 

        # Build graph - rebuild every call - not efficient
        edge_index = self.build_knn_graph(X)

        # Build edge attributes
        edge_attr = self.build_edge_attr(X, edge_index)

        # Latent node features
        h = self.node_encoder(node_attr) # self.node_encoder.forward(...)
        # Latent edge features
        g = self.edge_encoder(edge_attr) # self.edge_encoder.forward(...)

        # Updated latent node and edge features
        h, g = self.processor(h, g, edge_index) # self.processor.forward(...)

        # Output features [rho, u, v, p, ...]
        # maps latent/hidden features to physical outputs
        Y = self.decoder(h) # self.processor.forward(...)

        return Y

    def build_edge_attr(self, X, edge_index):
        """
        Build edge features from node coordinates.

        For each edge j -> i, this method computes the relative position
        between the receiver node i and the sender node j. Optionally, it
        also appends the Euclidean distance between the two nodes.

        Parameters
        ----------
        X : torch.Tensor
            Node coordinates with shape (N, node_input_dim), where N is the
            number of nodes.

        edge_index : torch.Tensor
            Edge connectivity with shape (2, E), where E is the number of
            edges. The first row contains receiver node indices and the second
            row contains sender node indices.

        Returns
        -------
        edge_attr : torch.Tensor
            Edge feature tensor with shape (E, edge_input_dim). It contains
            the relative position for each edge, and optionally the edge
            distance if `self.use_edge_distance` is enabled.
        """

        receivers = edge_index[0]
        senders = edge_index[1]

        # Get receivers and senders nodes
        x_i = X[receivers]
        x_j = X[senders]

        # Relative position
        relative_position = x_i - x_j

        # Edge features 
        edge_features = [relative_position]

        # Distance between node i and j
        if self.use_edge_distance:
            distance = torch.norm(relative_position, dim=1, keepdim=True)
            edge_features.append(distance)

        # Concatenate relative position with the distance vector
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

        return edge_index