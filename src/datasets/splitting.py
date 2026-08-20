# SPDX-License-Identifier: MIT
import numpy as np

def create_split_indices(number_of_points, params):
    """Create shuffled training, validation, and test indices.

    Parameters
    ----------
    number_of_points
        Number of points in the dataset.
    params : dict
        Project configuration dictionary.

    Returns
    -------
    dict
        Training, validation, and test indices. 
    """

    # Seed for permutation
    seed = params["seed"]

    split_config = params["dataset"]
    # Get percentages in an array
    percentages = np.asarray([
        split_config["p_training_data"],
        split_config["p_validation_data"],
        split_config["p_test_data"],
    ], dtype=float)

    if percentages.shape != (3,):
        raise ValueError(
            "Expected training, validation, and test percentages"
        )

    if np.any(percentages < 0.0):
        raise ValueError(
            "Split percentages cannot be negative"
        )

    if not np.isclose(percentages.sum(), 100.0):
        raise ValueError(
            "Split percentages must sum to 100%"
        )

    subset_sizes = np.floor(
        number_of_points * percentages / 100.0
    ).astype(int)

    # Assign rounding remainder to training.
    subset_sizes[0] += number_of_points - subset_sizes.sum()

    # Get shuffled indices
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(number_of_points)

    train_end = subset_sizes[0]
    validation_end = train_end + subset_sizes[1]

    return {
        "training": shuffled_indices[:train_end],
        "validation": shuffled_indices[train_end:validation_end],
        "test": shuffled_indices[validation_end:]
    }