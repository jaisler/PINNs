# SPDX-License-Identifier: MIT
"""
Factory functions for creating neural network architectures.
"""

from src.networks import MLP, GNN

def build_network(params):
    """
    Create the neural network from the configuration file.

    The selected architecture can be:
    - MLP
    - GNN
    """

    architecture = params['network']['architecture']

    # Multilayer Perceptron
    if architecture == 'mlp':

        mlp_cfg = params['network']['mlp']

        network = MLP(
            layers=mlp_cfg['layers'],
            activation=mlp_cfg['activation'],
            dropout_p=mlp_cfg['dropout'].get("probability", 0.0),
            dropout_indices=mlp_cfg['dropout'].get("hidden_layer_indices", []),
        )

    # Graph Neural Network
    elif architecture == 'gnn':

        gnn_cfg = params['network']['gnn']
        # Node input dimension
        node_input_dim = params['geometry']['dimension']
        # Edge input dimension
        edge_input_dim = params['geometry']['dimension']

        if gnn_cfg['attributes']['node']['boundary_marker']:
            node_input_dim += 1

        if gnn_cfg['attributes']['edge']['squared_distance']:
            edge_input_dim += 1

        # Get output dimension
        if params['run']['equation'] == 'euler':
            output_dim = 4

        elif params['run']['equation'] == 'rans':
            output_dim = 5

        else:
            raise ValueError(
                f"Unknown equation: {params['run']['equation']}."
            )

        if gnn_cfg['attributes']['node']['boundary_marker']:
            boundary_marker = None  # TODO: implement boundary marker
        else:
            boundary_marker = None

        network = GNN(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            output_dim=output_dim,
            latent_dim=gnn_cfg['latent_dim'],
            activation=gnn_cfg['activation'],
            neighbors=gnn_cfg['neighbors'],
            message_layers=gnn_cfg['processor']['message_layers'],
            aggregation=gnn_cfg['processor']['aggregation'],
            residual=gnn_cfg['processor']['residual'],
            boundary_marker=boundary_marker,
            use_edge_sdistance=gnn_cfg['attributes']['edge']['squared_distance'],
        )

    else:
        raise ValueError(
            f"Unknown network architecture: {architecture}."
        )

    return network