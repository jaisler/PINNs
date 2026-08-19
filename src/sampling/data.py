# SPDX-License-Identifier: MIT
from .sampling import SamplingData

def get_data_points(params):
    """Sample CFD observations or load a saved observation set.

    Parameters
    ----------
    params : dict
        Configuration parameters.

    Returns
    -------
    dict
        Coordinates, flow variables, and sampling groups.
    """

    # Validate the selected problem mode.
    problem = params["run"].get("problem", "forward").lower()

    if problem not in {"forward", "inverse"}:
        raise ValueError(
            "Unsupported run.problem value. "
            "Expected 'forward' or 'inverse'."
        )

    if problem == "inverse":
        
        data_points = {
            "X": None,
            "U": None,
            "rho": None,
            "p": None,
            "mut": None,
            "pts_in": None,
            "pts_bc": None,
            "pts_grad": None,
        }

        return data_points

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
    """Sample or load collocation points for physics-informed training.
   
    Parameters
    ----------
    params : dict
        Configuration parameters.

    Returns
    -------
    dict
        Collocation coordinates and sampling groups, or ``None`` values for
        a supervised model.
    """
    # Validate the selected model.
    model = params["run"].get("model", "supervised").lower()

    if model not in {"supervised", "pinn"}:
        raise ValueError(
            "Unsupported run.model value. "
            "Expected 'supervised' or 'pinn'."
        )

    if model == "supervised":
        
        collocation_points = {
            "Xf": None,
            "pts_in": None,
            "pts_bc": None,
            "pts_grad": None,
        }

        return collocation_points

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
