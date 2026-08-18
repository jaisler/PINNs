# SPDX-License-Identifier: MIT
import numpy as np

def add_noise(values, noise_type, level, seed=None, nonnegative=False):
    """
    Add noise to an observation field.

    Parameters
    ----------
    values : numpy.ndarray
        Clean observation values.

    noise_type : str
        Noise type: "none" or "gaussian".

    level : float
        Noise standard deviation relative to the standard deviation
        of the clean values. For example, 0.05 represents 5% noise.

    seed : int, optional
        Random seed for reproducibility.

    nonnegative : bool, default=False
        Whether negative noisy values should be clipped to zero.

    Returns
    -------
    numpy.ndarray
        Noisy observation values.
    """

    if (level < 0.0 or level > 1.0):
        raise ValueError("Noise level must be between 0.0 and 1.0")

    if noise_type == "none" or level == 0.0:
        return values.copy()

    if noise_type == "gaussian":

        values = np.asarray(values, dtype=float)
        rng = np.random.default_rng(seed)

        reference_scale = np.std(values)
        sigma = level * reference_scale

        noisy_values = values + rng.normal(
            loc=0.0,
            scale=sigma,
            size=values.shape,
        )
    else:
        raise ValueError(f"Unknown noise type: {noise_type}.")

    if nonnegative:
        noisy_values = np.maximum(noisy_values, 0.0)

    return noisy_values