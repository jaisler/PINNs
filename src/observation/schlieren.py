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
    Generate synthetic schlieren field from density field 
    from CFD data.
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
        gradient="density_gradient",
        preference=association,
    )

    # Get coordinates and gradient
    if association == "point":
        coordinates = derivative.points
        gradient = np.asarray(
            derivative.point_data["density_gradient"]
        )
    else:
        coordinates = derivative.cell_centers().points
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
        schlieren["grad_x"] = gradient[:,1]
    elif grad_type == "magnitude":
        schlieren["grad_x"] = gradient[:,2]
    else:
        raise KeyError(
            f"Gradient type '{grad_type}' is not available. "
        ) 

    return schlieren