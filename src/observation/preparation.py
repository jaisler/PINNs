# SPDX-License-Identifier: MIT
import numpy as np
from ..datasets import create_split_indices

def prepare_observation_data(raw_observation, params):
    """Prepare all loaded observation modalities.

    Parameters
    ----------
    raw_observation : dict
        Raw observations organized by modality. Supported keys are
        ``"schlieren"``, ``"velocity_profiles"``, and
        ``"pressure_taps"``.
    params : dict
        Project configuration dictionary.

    Returns
    -------
    dict
        Enabled observation modalities containing prepared training,
        validation, and test subsets.
    """

    prepared = {}

    if "schlieren" in raw_observation:
        prepared["schlieren"] = (
            _prepare_schlieren(raw_observation["schlieren"], params)
        )

    if "velocity_profiles" in raw_observation:
        prepared["velocity_profiles"] = (
            _prepare_velocity_profiles(
                raw_observation["velocity_profiles"], 
                params,
            )
        )

    if "pressure_taps" in raw_observation:
        prepared["pressure_taps"] = (
            _prepare_pressure_taps(
                raw_observation["pressure_taps"],
                params,
            )
        )

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
        Training, validation, and test subsets. Each subset contains
        ``"X"`` with shape ``(N, dimension)`` and ``"value"`` with
        shape ``(N, 1)``.
    """

    dims = params["geometry"]["dimension"]
    
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

def _prepare_velocity_profiles(velocity_profiles, params):
    """Split velocity profiles coordinates and measurements 
    into data subsets.

    Parameters
    ----------
    velocity_profiles : list of dict
        Velocity profiles. Each dictionary contains coordinate arrays
        and one measured velocity-component array.
    params : dict
        Project configuration dictionary.

    Returns
    -------
    dict
        Training, validation, and test subsets. List of prepared 
        velocity profiles.
    """



    return {}


def _prepare_pressure_taps(pressure_taps, params):
    """Split pressure taps coordinates and measurements into data subsets.

    Parameters
    ----------
    pressure_taps : dict
        Coordinate arrays and the measured pressure array ``"p"``.
    params : dict
        Project configuration dictionary.

    Returns
    -------
    dict
        Training, validation, and test subsets. Each subset contains
        ``"X"`` with shape ``(N, dimension)`` and ``"p"`` with shape
        ``(N, 1)``.
    """

    dims = params["geometry"]["dimension"]
    
    # Coordinates
    coordinate_names = ("x", "y", "z")[:dims]

    coordinates = np.column_stack([
        np.asarray(pressure_taps[name], dtype=float).reshape(-1)
        for name in coordinate_names
    ])

    pressure = np.asarray(pressure_taps["p"], dtype=float).reshape(-1, 1)

    number_of_points = coordinates.shape[0]

    if number_of_points == 0:
        raise ValueError("Pressure taps observations cannot be empty")

    if pressure.shape[0] != number_of_points:
        raise ValueError(
            "Pressure taps coordinates and pressure must contain the same "
            "number of points"
        )

    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Pressure taps coordinates contain non-finite values")

    if not np.all(np.isfinite(pressure)):
        raise ValueError("Pressure taps values contain non-finite values")

    # Get split indices
    split_indices = create_split_indices(
        number_of_points,
        params,
    )

    return {
        subset: {
            "X": coordinates[indices],
            "p": pressure[indices],
        }
        for subset, indices in split_indices.items()
    }