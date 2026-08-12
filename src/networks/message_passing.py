# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn

from .mlp import MLP

class MessagePassingLayer(nn.Module):
    """Update latent edge and node states through message passing."""

    def __init__(
        self,
        latent_dim,
        neighbors,
        activation="tanh",
        message_layers=4,
        aggregation="sum",
        residual=True
    ):
        """Configure the message-passing processor.

        Parameters
        ----------
        latent_dim : int
            Width of latent node and edge features.
        neighbors : int
            Configured number of neighbors per node.
        activation : str, optional
            Activation used by update MLPs.
        message_layers : int, optional
            Number of message-passing iterations.
        aggregation : str, optional
            Incoming-message reduction: ``"sum"``, ``"max"``, or ``"mean"``.
        residual : bool, optional
            Whether to use residual state updates.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.neighbors = neighbors

        if message_layers < 1:
            raise ValueError(f"{message_layers} should be greater than zero")
        self.message_layers = message_layers
        
        # Aggregation function: sum, max, 
        self.aggregation = aggregation
        
        if not isinstance(residual, bool):
            raise TypeError(f"residual must be a boolean, got {type(residual).__name__}.")
        self.residual = residual
        
        # Edge update MLP
        self.edge_update = MLP(
            layers=[3 * latent_dim, 2 * latent_dim, latent_dim, latent_dim],
            activation=activation
        )

        # Node update MLP
        self.node_update = MLP(
            layers=[2 * latent_dim, 2* latent_dim, latent_dim, latent_dim],
            activation=activation
        )

    def forward(self, h, g, edge_index, use_dropout=False):
        """Apply the configured updates to latent graph features.

        Parameters
        ----------
        h : torch.Tensor
            Node features with shape ``(N, latent_dim)``.
        g : torch.Tensor
            Edge features with shape ``(E, latent_dim)``.
        edge_index : torch.Tensor
            Receiver and sender indices with shape ``(2, E)``.
        use_dropout : bool, optional
            Reserved dropout flag.

        Returns
        -------
        torch.Tensor
            Updated node features.
        """

        receivers = edge_index[0]
        senders = edge_index[1]
        n_nodes = h.shape[0]

        for _ in range(self.message_layers):
            
            # Edge information: h_i, h_j, g_ij
            hi = h[receivers] 
            hj = h[senders] 

            # Message passing - Edge update / message computation
            message_input = torch.cat([hi, hj, g], dim=1)
            delta_g = self.edge_update(message_input)

            # Update edge
            if self.residual:
                g = g + delta_g
            else:
                g = delta_g

            # Collect all messages coming from its neighbors j E N(i)
            aggr_mi = self.message_aggregation(g, receivers, n_nodes)

            # Update node 
            update_input = torch.cat([h, aggr_mi], dim=1)
            delta_h = self.node_update(update_input) 

            if self.residual:
                h = h + delta_h
            else:
               h = delta_h
            
        return h

    def message_aggregation(self, mij, receivers, n_nodes):
        """Aggregate incoming edge messages for each receiver node.

        Parameters
        ----------
        mij : torch.Tensor
            Edge messages.
        receivers : torch.Tensor
            Receiver index for each edge.
        n_nodes : int
            Number of graph nodes.

        Returns
        -------
        torch.Tensor
            Aggregated message for every node.
        """

        if self.aggregation == 'sum':
            # aggr_mi = (N, latent_dim)
            # using the same device and data type as mij
            aggr_mi = mij.new_zeros((n_nodes, self.latent_dim))

            #for e in range(len(receivers)):
            #    aggr_mi[receivers[e],:] += mij[e,:]
            aggr_mi.index_add_(0, receivers, mij)

        elif self.aggregation == 'max':
            # fill vector with -inf to evaluate max
            aggr_mi = mij.new_full((n_nodes, self.latent_dim), float('-inf'))

            #for e in range(len(receivers)):
            #    aggr_mi[receivers[e],:] = torch.maximum(aggr_mi[receivers[e],:], mij[e,:])

            index = receivers[:, None].expand(-1, self.latent_dim)

            aggr_mi.scatter_reduce_(
                dim=0,
                index=index,
                src=mij,
                reduce="amax",
                include_self=True,
            )

        elif self.aggregation == 'mean':
            # Note that this implementation assumes that then nodes always have
            # the same number of neighbors and the receiver are ordered.
            aggr_mi = mij.new_zeros((n_nodes, self.latent_dim))
            
            #n = 0
            #for e in range(len(receivers)):
            #    aggr_mi[receivers[e],:] += mij[e,:]
                # Count neighbors
            #    n += 1
                # Check if change node               
            #    if n == self.neighbors:
            #        aggr_mi[receivers[e],:] /= float(self.neighbors) 
            #        n = 0

            aggr_mi.index_add_(0, receivers, mij)

            counts = mij.new_zeros((n_nodes, 1))
            ones = mij.new_ones((receivers.numel(), 1))
            counts.index_add_(0, receivers, ones)

            aggr_mi = aggr_mi / counts.clamp_min(1.0)

        # TODO
        #elif self.aggretation == 'attention':

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}. " 
                             "Available aggregation functions are sum, max, mean, attention.")

        return aggr_mi
        
