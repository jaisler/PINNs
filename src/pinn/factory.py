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
        data["xtrain"], data["ytrain"], # training data
        data["rhotrain"], data["utrain"], data["vtrain"], 
        data["ptrain"], # training data
        data["xftrain"], data["yftrain"], # collocation data
        params, # general parameters
        data["muttrain"], # RANS eq.
        data["xval"], data["yval"], data["rhoval"], data["uval"], 
        data["vval"], data["pval"], data["mutval"] # validation data
    )

    return model
