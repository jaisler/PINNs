# SPDX-License-Identifier: MIT
from .physics_informed_nn import PhysicsInformedNN
"""
Factory function for creating physics-informed neural network model.
"""

def build_pinn_model(network, data, params):
    """Build a configured physics-informed model wrapper.

    Parameters
    ----------
    network : torch.nn.Module
        MLP or GNN used for field prediction.
    data : dict
        Prepared training, validation, and collocation arrays.
    params : dict
        PIRFlow configuration.

    Returns
    -------
    PhysicsInformedNN
        Initialized model wrapper.
    """

    model = PhysicsInformedNN(
        network, # MLP or GNN 
        data["training"]["xtrain"], data["training"]["ytrain"], # training data
        data["training"]["rhotrain"], data["training"]["utrain"], data["training"]["vtrain"], 
        data["training"]["ptrain"], # training data
        data["collocation"]["xftrain"], data["collocation"]["yftrain"], # collocation data
        params, # general parameters
        data["training"]["muttrain"], # RANS eq.
        data["validation"]["xval"], data["validation"]["yval"], data["validation"]["rhoval"], 
        data["validation"]["uval"], data["validation"]["vval"], data["validation"]["pval"], 
        data["validation"]["mutval"] # validation data
    )

    return model
