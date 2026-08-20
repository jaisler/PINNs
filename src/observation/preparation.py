# SPDX-License-Identifier: MIT
import numpy as np
from ..datasets import create_split_indices

def prepare_observation_data(raw_observation, params):
    prepared = {}

    if "schlieren" in raw_observation:
        prepared["schlieren"] = (
            _prepare_schlieren(raw_observation["schlieren"], params)
        )

    #if "pressure_taps" in raw_observation:
    #    prepared["pressure_taps"] = (
    #        _prepare_schlieren(raw_observation["pressure_taps"], params)
    #    )

    return prepared


def _prepare_schlieren(schlieren, params):
    """Split Schlieren coordinates and measurements into data subsets.

    Parameters
    ----------
    schlieren : dict
        Coordinate arrays and the configured density-gradient observable.
    params : dict
        Project configuration dictionary.

    Returns
    -------
    dict
        Training, validation, and test dictionaries. Each contains an
        ``X`` coordinate array and a ``value`` observation array.
    """

    # Check dimension
    dims = params["geometry"]["dimension"]
    if dims not in (1, 2, 3):
        raise ValueError(f"Invalid problem dimension: {dims}")

    # Coordinates
    coordinate_names = ("x", "y", "z")[:dims]

    # Get density-gradient type
    gradient_type = (
        params["identification"]
        ["observations"]
        ["schlieren"]
        ["grad_type"]
    )

    required_fields = (*coordinate_names, gradient_type)

    missing_fields = [
        field
        for field in required_fields
        if field not in schlieren
    ]

    if missing_fields:
        raise KeyError(
            f"Missing Schlieren fields: {', '.join(missing_fields)}"
        )

    coordinates = np.column_stack([
        np.asarray(schlieren[name], dtype=float).reshape(-1)
        for name in coordinate_names
    ])

    values = np.asarray(
        schlieren[gradient_type],
        dtype=float,
    ).reshape(-1, 1)

    number_of_points = coordinates.shape[0]

    if number_of_points == 0:
        raise ValueError("Schlieren observations cannot be empty")

    if values.shape[0] != number_of_points:
        raise ValueError(
            "Schlieren coordinates and values must contain the same "
            "number of points"
        )

    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Schlieren coordinates contain non-finite values")

    if not np.all(np.isfinite(values)):
        raise ValueError("Schlieren values contain non-finite values")

    # Get split indices
    split_indices = create_split_indices(
        number_of_points,
        params,
    )

    return {
        subset: {
            "X": coordinates[indices],
            "value": values[indices],
        }
        for subset, indices in split_indices.items()
    }
