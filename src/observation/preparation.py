# SPDX-License-Identifier: MIT
import numpy as np

from .schlieren_sampling import sample_schlieren_observations
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
        Coordinate arrays and the configured density-gradient observable
        on a camera grid.
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
        params["identification"]["observations"]["schlieren"]["grad_type"]
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

    values = np.asarray(schlieren[gradient_type], dtype=float).reshape(-1, 1)

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

    sampling_config = (
        params["identification"]["observations"]["schlieren"]["sampling"]
    )
    sampled_coords, sampled_values = sample_schlieren_observations(
        coordinates, 
        values, 
        sampling_config, 
        params["seed"]
    )

    number_of_samples = sampled_coords.shape[0]

    # Get split indices
    split_indices = create_split_indices(
        number_of_samples,
        params,
    )

    return {
        subset: {
            "X": sampled_coords[indices],
            "value": sampled_values[indices],
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

    dims = params["geometry"]["dimension"]
    
    # Coordinates
    coordinate_names = ("x", "y", "z")[:dims]
    # Velocity profiles
    velocity_names = ("u", "v", "w")[:dims]

    velocity_config = (
        params["identification"]["observations"]["velocity_profiles"]
    )
    # Number of velocity profile files
    n_files = velocity_config["n_files"]

    if n_files <= 0:
        raise ValueError(
            "Number of velocity-profile files must be positive"
        )

    expected_profiles = dims * n_files

    if len(velocity_profiles) != expected_profiles:
        raise ValueError(
            f"Expected {expected_profiles} velocity profiles, "
            f"received {len(velocity_profiles)}"
        )


    components = {}
    for component_index, component_name in enumerate(velocity_names):
        component_coordinates = []
        component_velocities = []

        for file_index in range(n_files):
            profile_index = (
                component_index * n_files + file_index
            )

            profile = velocity_profiles[profile_index]


            component_coordinates.append(
                np.column_stack([
                    np.asarray(profile[name], dtype=float).reshape(-1)
                    for name in coordinate_names
                ])
            )

            component_velocities.append(
                np.asarray(profile[component_name], dtype=float).reshape(-1, 1)
            )

            number_of_points = component_coordinates[file_index].shape[0]

            if number_of_points == 0:
                raise ValueError("Velocity observations cannot be empty")

            if component_velocities[file_index].shape[0] != number_of_points:
                raise ValueError(
                    "Velocity profile coordinates and values must contain the same "
                    "number of points"
                )

            if not np.all(np.isfinite(component_coordinates[file_index])):
                raise ValueError(
                    "Velocity profiles coordinates contain non-finite values"
                )

            if not np.all(np.isfinite(component_velocities[file_index])):
                raise ValueError(
                    "Velocity profiles values contain non-finite values"
                )

        coordinates = np.vstack(component_coordinates)
        velocities = np.vstack(component_velocities)

        # Get split indices
        split_indices = create_split_indices(
            coordinates.shape[0],
            params,
        )

        components[component_name] = {
            subset: {
                "X": coordinates[indices],
                "value": velocities[indices],
            }
            for subset, indices in split_indices.items()
        }

    return components

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
            "value": pressure[indices],
        }
        for subset, indices in split_indices.items()
    }