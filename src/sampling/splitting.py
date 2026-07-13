# SPDX-License-Identifier: MIT
import numpy as np
from pathlib import Path

def valid_indices(indices, expected_size, number_of_points):
    """
    Check that an index array has the expected size and contains
    valid indices.

    Check if an array contains valid indices.

    Empty arrays are valid when expected_size == 0.

    Parameters
    ----------
    indices : array-like
        Indices to check.
    expected_size : int
        Expected number of indices.
    number_of_points : int
        Total number of available points.

    Returns
    -------
    bool
        True if the indices have the correct size and are within the
        valid range.
    
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

def get_data_split_indeces(N, params):
    """
    Create or load train/validation/test indices for the data points.

    If a valid saved split exists, it is loaded. Otherwise, a new random
    split is generated and saved.

    Parameters
    ----------
    N : int
        Total number of data points.
    params : dict
        Configuration parameters.

    Returns
    -------
    idx_train : numpy.ndarray
        Training indices.
    idx_val : numpy.ndarray
        Validation indices.
    idx_test : numpy.ndarray
        Test indices.
    N_train_data : int
        Number of training points.
    N_val_data : int
        Number of validation points.
    N_test_data : int
        Number of test points.
    """    

    # Data points
    # Train / validation / test split
    perc_train = params["dataset"]["p_training_data"]
    perc_val = params["dataset"]["p_validation_data"]
    perc_test = params["dataset"]["p_test_data"]

    if perc_train < 0 or perc_val < 0 or perc_test < 0:
        raise ValueError("Split percentages cannot be negative")

    if (perc_train + perc_val + perc_test) > 100:
        raise ValueError("Split percentages cannot be greater than 100%")

    if perc_test <= 0:
        raise ValueError("Test data percentage must be greater than 0%")

    # Number of points in each subset
    N_train_data = int(N * perc_train / 100)
    N_val_data = int(N * perc_val / 100)
    N_test_data = int(N * perc_test / 100)

    # Give any rounding remainder to the training dataset
    N_train_data += (N - N_train_data - N_val_data - N_test_data)

    # Path for the dataset split
    idx_file = Path(params['paths']['samples']) / "idx_split_data.npz"

    load_existing_split = (
        idx_file.exists()
        and not params["run"]["routines"]["sampling"]
    )

    split_is_valid = False

    # Try to load an existing split
    if load_existing_split:
        try:
            with np.load(idx_file) as split:
                idx_train = split["idx_train"]
                idx_val = split["idx_val"]
                idx_test = split["idx_test"]

            split_is_valid = (
                valid_indices(idx_train, N_train_data, N)
                and valid_indices(idx_val, N_val_data, N)
                and valid_indices(idx_test, N_test_data, N)
            )

            if split_is_valid:
                # Check that train, validation, and test do not overlap
                idx_combined = np.concatenate(
                    [idx_train, idx_val, idx_test]
                )

                split_is_valid = (
                    np.unique(idx_combined).size
                    == idx_combined.size
                )

            if not split_is_valid:
                print("---------------------------------------")
                print(
                    "Existing dataset split is incompatible with the "
                    "current configuration."
                )

        except (OSError, ValueError, KeyError) as error:
            print(
                f"Could not load the existing dataset split: {error}"
            )
            split_is_valid = False

    # Generate a split when no valid existing split was loaded
    if not split_is_valid:
        print("---------------------------------------")
        print("Generating a new dataset split.")

        rng = np.random.default_rng(
            params.get("seed", 1234)
        )

        idx_all = rng.permutation(N)

        train_end = N_train_data
        val_end = train_end + N_val_data
        test_end = val_end + N_test_data

        idx_train = idx_all[:train_end]
        idx_val = idx_all[train_end:val_end]
        idx_test = idx_all[val_end:test_end]

        np.savez(
            idx_file,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
        )

        print("Dataset split prepared.")

        return (
            idx_train,
            idx_val,
            idx_test,
            N_train_data,
            N_val_data,
            N_test_data,
        )
    
def get_collocation_indeces(N_coll, params):
    """
    Create or load collocation indices for the training collocation points.

    Parameters
    ----------
    N_coll : int
        Total number of collocation points.
    params : dict
        Configuration parameters.

    Returns
    -------
    idxc : numpy.ndarray
        Collocation-point indices.
    """

    # File for loading collocation ponts
    idxc_file = Path(params['paths']['samples']) / "idx_train_coll.npy"

    if idxc_file.exists() and not params['run']['routines']['sampling']:
        idxc = np.load(idxc_file)

        if idxc.shape[0] != N_coll:
            raise ValueError(
                "Loaded collocation indices have a different size from Xf. "
                f"Expected {N_coll}, got {idxc.shape[0]}."
            )

        if idxc.size > 0 and np.max(idxc) >= N_coll:
            raise ValueError(
                "Loaded collocation indices are not compatible with Xf."
            )

        if idxc.size > 0 and np.any(idxc < 0):
            raise ValueError(
                "Loaded collocation indices contain negative values."
            )

    else:
        rng = np.random.default_rng(params.get("seed", 1234))
        idxc = rng.choice(N_coll, N_coll, replace=False)
        np.save(idxc_file, idxc)

    return idxc