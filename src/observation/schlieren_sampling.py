# SPDX-License-Identifier: MIT
"""Select camera pixels from synthetic Schlieren observations."""

import numpy as np

def sample_schlieren_observations(
        coordinates,
        values,
        sampling_config,
        seed,
):
    """Select Schlieren observations using uniform and signal-based sampling.

    The configured fraction is selected uniformly without replacement. The
    remaining points are selected from unused pixels with probabilities
    proportional to ``abs(values) ** alpha``.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Valid camera-pixel coordinates with shape ``(N, dimension)``.
    values : numpy.ndarray
        Schlieren values with shape ``(N, 1)``.
    sampling_config : dict
        Sampling settings containing ``points``, ``uniform_fraction``, and
        ``alpha``.
    seed : int
        Random seed used for reproducible sampling.

    Returns
    -------
    sampled_coordinates : numpy.ndarray
        Selected coordinates with shape ``(points, dimension)``.
    sampled_values : numpy.ndarray
        Selected Schlieren values with shape ``(points, 1)``.
    """
    
    # Camera pixels
    number_of_available_points = coordinates.shape[0]
    number_of_points = int(sampling_config["points"])
    uniform_fraction = float(sampling_config["uniform_fraction"])
    alpha = float(sampling_config["alpha"])

    if number_of_points <= 0:
        raise ValueError("sampling.points must be positive")

    if number_of_points > number_of_available_points:
        raise ValueError(
            f"Requested {number_of_points} Schlieren points, but only "
            f"{number_of_available_points} are available from Schlieren image"
        )

    if not 0.0 <= uniform_fraction <= 1.0:
        raise ValueError(
            "sampling.uniform_fraction must be between 0 and 1"
        )

    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError(
            "sampling.alpha must be finite and nonnegative"
        )

    rng = np.random.default_rng(seed)

    # Number of uniformly generated points
    number_of_uniform_points = round(
        number_of_points * uniform_fraction
    )

    # Regarding the sampling points
    number_of_remaining_points = (
        number_of_points - number_of_uniform_points
    )

    # Note that, the number_of_available_points is related to the
    # number of points in the camera (pixels)
    available_indices = np.arange(
        number_of_available_points,
        dtype=int,
    )

    # Uniformly sampled observations.
    if number_of_uniform_points > 0:
        uniform_indices = rng.choice(
            available_indices,
            size=number_of_uniform_points,
            replace=False,
        )
    else:
        uniform_indices = np.empty(0, dtype=int)

    # Remove uniform samples from available points.
    remaining_mask = np.ones(
        number_of_available_points,
        dtype=bool,
    )
    remaining_mask[uniform_indices] = False
    remaining_indices = available_indices[remaining_mask]

    if number_of_remaining_points > 0:

        signal_strength = np.abs(
            values[remaining_indices, 0]
        )

        # The small epsilon also produces uniform probabilities when
        # every remaining signal value is zero.
        weights = (
            signal_strength**alpha
            + np.finfo(float).eps
        )

        probabilities = weights / weights.sum()

        importance_indices = rng.choice(
            remaining_indices,
            size=number_of_remaining_points,
            replace=False,
            p=probabilities,
        )
    else:
        importance_indices = np.empty(0, dtype=int)

    # Concatenate uniform and importance indices
    selected_indices = np.concatenate([
        uniform_indices,
        importance_indices,
    ])

    # Mix uniform and importance-selected observations.
    rng.shuffle(selected_indices)

    return (
        coordinates[selected_indices],
        values[selected_indices],
    )
