# SPDX-License-Identifier: MIT
import numpy as np

def valid_indices(indices, expected_size, number_of_points):
    """
    Check that an index array has the expected size and contains
    valid indices.

    Empty arrays are valid when expected_size == 0.
    """
    indices = np.asarray(indices)

    if indices.ndim != 1:
        return False

    if indices.size != expected_size:
        return False

    if indices.size == 0:
        return True

    return (
        np.all(indices >= 0)
        and np.all(indices < number_of_points)
    )