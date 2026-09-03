# SPDX-License-Identifier: MIT
import numpy as np

def sample_schlieren_observations(
        coordinates,
        values,
        sampling_config,
        seed,
):
    """Sample pixels from a generated Schlieren observation.

    A fraction of the requested pixels is selected uniformly. The
    remainder is selected with probabilities proportional to

        abs(signal) ** alpha

    Sampling is performed without replacement.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Observation coordinates with shape ``(N, dimension)``.

    values : numpy.ndarray
        Schlieren values with shape ``(N,)`` or ``(N, 1)``.

    sampling_config : dict
        Configuration containing ``points``, ``uniform_fraction``,
        and ``alpha``.

    seed : int
        Random seed used for reproducible sampling.

    Returns
    -------
    sampled_coordinates : numpy.ndarray
        Sampled coordinates with shape ``(number_of_points, dimension)``.

    sampled_values : numpy.ndarray
        Sampled Schlieren values with shape ``(number_of_points, 1)``.
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


    # TODO:
    # 1. Sample uniformly (randomlly LHS)
    # 2. Sample using gradient-based
    # 3. Put them together in arrays like: sampled_coordinates and
    # sampled_values.
    # 4. Return arrays
 
    return coordinates, values