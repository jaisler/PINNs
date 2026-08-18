# SPDX-License-Identifier: MIT
from pathlib import Path

import numpy as np
import pyvista as pv

def generate_synthetic_schlieren(
        file_path,
        density_name="rho",
        grad_type="magnitude",
        dims=2,
    ):
    """
    Generate a synthetic schlieren field from a CFD density field.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the CFD VTK file.

    density_name : str, default="rho"
        Name of the density array in the VTK file.

    grad_type : str, default="magnitude"
        Selected density-gradient quantity. Available options are
        "grad_x", "grad_y", "grad_z", and "magnitude", depending 
        on the dimension.

    dims : int, default=2
        Spatial dimension of the problem.

    Returns
    -------
    dict
        Coordinates, density-gradient fields, and selected values.
    """

    file_path = Path(file_path)
    mesh = pv.read(file_path)

    # Determine whether density is stored at points or cells.
    if density_name in mesh.point_data:
        association = "point"
    elif density_name in mesh.cell_data:
        association = "cell"
    else:
        raise KeyError(
            f"Density variable '{density_name}' was not found. "
            f"Available arrays: {mesh.array_names}"
        )

    # Compute the density gradient.
    derivative = mesh.compute_derivative(
        scalars=density_name,
        gradient="density_gradient", # naming
        preference=association,
    )

    # Get coordinates and gradient
    if association == "point":
        coordinates = np.asarray(derivative.points)
        gradient = np.asarray(
            derivative.point_data["density_gradient"]
        )
    else:
        coordinates = np.asarray(derivative.cell_centers().points)
        gradient = np.asarray(
            derivative.cell_data["density_gradient"]
        )

    coordinate_names = ["x", "y", "z"][:dims]

    schlieren = {
        coordinate: coordinates[:, index]
        for index, coordinate in enumerate(coordinate_names)
    }

    if grad_type == "grad_x":
        schlieren["grad_x"] = gradient[:,0]
    elif grad_type == "grad_y":
        schlieren["grad_y"] = gradient[:,1]
    elif grad_type == "grad_z": 
        schlieren["grad_z"] = gradient[:,2]
    elif grad_type == "magnitude":
        if dims == 1:
            schlieren["magnitude"] = (
                np.sqrt(gradient[:,0] * gradient[:,0])
            )
        if dims == 2:
            schlieren["magnitude"] = (
                np.sqrt(gradient[:,0] * gradient[:,0] 
                        + gradient[:,1] * gradient[:,1])
                )
        if dims == 3:
            schlieren["magnitude"] = (
                np.sqrt(gradient[:,0] * gradient[:,0] 
                        + gradient[:,1] * gradient[:,1]
                        + gradient[:,2] * gradient[:,2])
                )
    else:
        raise KeyError(
            f"Invalid schlieren gradient type: '{grad_type}'."
        ) 

    return schlieren