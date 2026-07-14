# SPDX-License-Identifier: MIT
from .sampling import SamplingData

def get_data_points(params):
    """
    Sample new data points or read existing data points from file.

    Parameters
    ----------
    params : dict
        Configuration parameters.

    Returns
    -------
    data_points : dict
        Data points and flow variables
    """

    sampling_enabled = params['run']['routines']['sampling']
       
    # False indicates data points
    data_handler = SamplingData(params, False)

    if sampling_enabled:
        # Sample
        data_handler.sample() 
        # Write data to file
        data_handler.write_data_to_npz()
        
        # Get sampling ponits and fields. 
        X = data_handler.get_x()
        U = data_handler.get_u() 
        rho = data_handler.get_rho() 
        p = data_handler.get_p() 
        # Note that if Euler equations are used 
        # it returns an array of zeros
        mut = data_handler.get_mut()

        # Get domain points
        pts_in = data_handler.get_pts_in()
        pts_bc = data_handler.get_pts_bc()
        pts_grad = data_handler.get_pts_grad()

    else:
        X, pts_in, pts_bc, pts_grad, U, rho, p, mut = \
            data_handler.read_data_from_npz()
    
    data_points = {
        "X": X,
        "U": U,
        "rho": rho,
        "p": p,
        "mut": mut,
        "pts_in": pts_in,
        "pts_bc": pts_bc,
        "pts_grad": pts_grad,
    }

    return data_points

def get_collocation_points(params):
    """
    Sample new collocation points or read existing points from file.

    Collocation points are only required for a physics-informed model.
   
    Parameters
    ----------
    params : dict
        Configuration parameters.

    Returns
    -------
    collocation_points : dict
        Collocation points    
    """

    if params["run"]["model"] != "pinn":
        return None

    sampling_enabled = params['run']['routines']['sampling']

    # True indicates collocation points
    collocation_handler = SamplingData(params, True)

    # Sampling points - Data points
    if sampling_enabled:
        # Sample
        collocation_handler.sample() 
        # Write data to file
        collocation_handler.write_data_to_npz()
        # Get collocation points
        Xf = collocation_handler.get_x()  

        # Get domain points
        pts_in = collocation_handler.get_pts_in()
        pts_bc = collocation_handler.get_pts_bc()
        pts_grad = collocation_handler.get_pts_grad()

    else:
        Xf, pts_in, pts_bc, pts_grad, _, _, _, _ = \
            collocation_handler.read_data_from_npz()

    collocation_points = {
        "Xf": Xf,
        "pts_in": pts_in,
        "pts_bc": pts_bc,
        "pts_grad": pts_grad,
    }

    return collocation_points