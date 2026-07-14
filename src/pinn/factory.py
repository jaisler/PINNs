# SPDX-License-Identifier: MIT

from .physics_informed_nn import PhysicsInformedNN

def build_pinn_model(network, data, params):
    """
    Build the physics-informed neural network model.

    Parameters
    ----------
    network : torch.nn.Module
        Neural network architecture used by the model, such as an MLP
        or GNN. Object from a class.

    data : dict
        Dictionary containing the prepared training, validation, and
        collocation datasets. 

    params : dict
        Dictionary containing the model, physics, optimizer, and runtime
        configuration parameters.

    Returns
    -------
    model : PhysicsInformedNN
        Initialized physics-informed neural network model. Object of the
        PhysicsInformedNN class.
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