# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNetwork
from .mlp import MLP
from .message_passing import MessagePassingLayer

class GNN(BaseNetwork):
    """Predict flow variables with an encode-process-decode graph network."""

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
        boundary_marker=False, 
        use_edge_sdistance=False,
    ):
        """Build the GNN encoder, processor, and decoder.

        Parameters
        ----------
        node_input_dim : int
            Number of node input features.
        edge_input_dim : int
            Number of geometric edge features.
        output_dim : int
            Number of predicted flow variables.
        latent_dim : int, optional
            Width of latent features.
        activation : str, optional
            Activation used by the internal MLPs.
        neighbors : int, optional
            Number of nearest neighbors per node.
        message_layers : int, optional
            Number of processor iterations.
        aggregation : str, optional
            Incoming-message reduction.
        residual : bool, optional
            Whether processor updates are residual.
        boundary_marker : array_like or None, optional
            Optional node boundary markers.
        use_edge_sdistance : bool, optional
            Whether edges include squared distance.
        """
        super().__init__(
            # Base class initialisation
            input_dim=node_input_dim,
            output_dim=output_dim,
        )

        # Boundary marker - node attributes
        if boundary_marker is not None:
            self.boundary_marker = boundary_marker
        else:
            self.boundary_marker = None
        
        # Use edge squared distance - edge attributes
        self.use_edge_sdistance = use_edge_sdistance

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
            neighbors=neighbors,
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

    def forward(self, X, edge_index, edge_attr, use_dropout=False):
        """Evaluate node features on an existing graph.

        Parameters
        ----------
        X : torch.Tensor
            Node features with shape ``(N, node_input_dim)``.
        edge_index : torch.Tensor
            Receiver and sender indices with shape ``(2, E)``.
        edge_attr : torch.Tensor
            Edge features with shape ``(E, edge_input_dim)``.
        use_dropout : bool, optional
            Reserved dropout flag.

        Returns
        -------
        torch.Tensor
            Raw node predictions with shape ``(N, output_dim)``.
        """
 
        # Node attributes
        node_attr = X 

        # Latent node features
        h = self.node_encoder(node_attr) # self.node_encoder.forward(...)
        # Latent edge features
        g = self.edge_encoder(edge_attr) # self.edge_encoder.forward(...)

        # Updated latent node and edge features
        h = self.processor(h, g, edge_index) # self.processor.forward(...)

        # Output features [rho, u, v, p, ...]
        # maps latent/hidden features to physical outputs
        Y = self.decoder(h) # self.processor.forward(...)

        return Y

    def build_graph(self, X):
        """Build connectivity and geometric edge attributes.

        Parameters
        ----------
        X : torch.Tensor
            Normalized node coordinates.

        Returns
        -------
        tuple of torch.Tensor
            Edge indices and edge attributes. Neighbor selection uses detached
            coordinates.
        """

        edge_index = self.build_knn_graph(X.detach())
        edge_attr = self.build_edge_attr(X, edge_index)

        return edge_index, edge_attr

    def build_edge_attr(self, X, edge_index):
        """Build geometric features for each directed edge.

        Parameters
        ----------
        X : torch.Tensor
            Node coordinates with shape ``(N, node_input_dim)``.
        edge_index : torch.Tensor
            Receiver and sender indices with shape ``(2, E)``.

        Returns
        -------
        torch.Tensor
            Relative positions and optional squared distances.
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

        if self.use_edge_sdistance:
            squared_distance = torch.sum(
                relative_position.square(),
                dim=1,
                keepdim=True,
            )
            edge_features.append(squared_distance)

        # Distance between node i and j
        #if self.use_edge_distance:
        #    distance = torch.norm(relative_position, dim=1, keepdim=True)
        #    edge_features.append(distance)

        # Concatenate relative position with the distance vector
        edge_attr = torch.cat(edge_features, dim=1)

        return edge_attr

    def build_knn_graph(self, X):
        """Build a directed k-nearest-neighbor graph.

        Parameters
        ----------
        X : torch.Tensor
            Node coordinates with shape (N, d).

        Returns
        -------
        torch.Tensor
            Receiver and sender indices with shape ``(2, N * neighbors)``.
        """

        if self.neighbors >= X.shape[0]:
            raise ValueError(
                f"neighbors must be smaller than the number of nodes. "
                f"Got neighbors={self.neighbors}, number_of_nodes={X.shape[0]}"
            )

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

            # Receiver nodes. Creates [0, 0, 0, 1, 1, 1, 2, ...] if k = 3
            receivers = torch.arange(N, device=X.device).repeat_interleave(k)

            # Sender nodes. Flattens the matrix into one long vector
            senders = knn_idx.reshape(-1) 

            # Each column represents one edge: receiver, sender.
            edge_index = torch.stack([receivers, senders], dim=0)

        return edge_index
