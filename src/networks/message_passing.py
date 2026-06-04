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

        self.latent_dim = latent_dim

        if message_layers < 1:
            raise ValueError(f"{message_layers} should be greater than zero")
        self.message_layers = message_layers
        
        # Aggregation function: sum, max, 
        self.aggregation = aggregation
        
        if not isinstance(residual, bool):
            raise TypeError(f"residual must be a boolean, got {type(residual).__name__}.")
        self.residual = residual
        
        # Message MLP
        self.message = MLP(
            layers=[3 * latent_dim, latent_dim, latent_dim],
            activation=activation
        )

        # Update MLP
        self.update = MLP(
            layers=[2 * latent_dim, latent_dim, latent_dim],
            activation=activation
        )

    def forward(self, h, g, edge_index, use_dropout=False):

        receivers = edge_index[0]
        senders = edge_index[1]
        n_nodes = h.shape[0]

        # Get information on the edges
        hi = h[receivers] 
        hj = h[senders] 

        for i in range(self.message_layers):
            
            # Message passing
            message_input = torch.cat([hi, hj, g], dim=1)
            mij = self.message(message_input)

            # Collect all messages coming from its neighbors j E N(i)
            aggr_mi = self.message_aggregation(mij, receivers, n_nodes)

            # Update 
            #update_input = torch.cat([hi, aggr_mi], dim=1)
            #if self.residual:
            #    delta_hi = self.update(update_input) 
            #    hi = hi + delta_hi
            #else:
            #   hi = self.update(update_input) 
                 
        ######## Should I return hi? ###########
        return h, g

    def message_aggregation(self, mij, receivers, n_nodes):

        if self.aggregation == 'sum':
            # aggr_mi = (N, latent_dim)
            aggr_mi = torch.zeros(n_nodes, self.latent_dim)

            for e in range(len(receivers)):
                aggr_mi[receivers[e],:] += mij[e,:]

        elif self.aggregation == 'max':
            # fill vector with -inf to evaluate max
            aggr_mi = torch.new_full(n_nodes, self.latent_dim,float('-inf'))

            for e in range(len(receivers)):
                aggr_mi[receivers[e],:] = torch.maximum(aggr_mi[receivers[e],:], mij[e,:])

        elif self.aggregation == 'mean':
            # Note that this implementation assumes that then nodes always have
            # the same number of neighbors and the receiver are ordered.
            aggr_mi = torch.zeros(n_nodes, self.latent_dim)

            n = 0
            for e in range(len(receivers)):
                aggr_mi[receivers[e],:] += mij[e,:]
                # Count neighbors
                n += 1
                # Check if change node               
                if n == self.neighbors:
                    aggr_mi[receivers[e],:] /= float(self.neighbors) 
                    n = 0

        # TODO
        #elif self.aggretation == 'attention':

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}. " 
                             "Available aggregation functions are sum, max, mean, attention.")

        return aggr_mi
        

