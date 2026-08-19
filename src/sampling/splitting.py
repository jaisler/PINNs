# SPDX-License-Identifier: MIT
import numpy as np
from pathlib import Path

def valid_indeces(indeces, expected_size, number_of_points):
    """Check the size and bounds of a one-dimensional index array.

    Parameters
    ----------
    indeces : array_like
        Indices to validate.
    expected_size : int
        Required number of indices.
    number_of_points : int
        Total number of available points.

    Returns
    -------
    bool
        Whether the indices have the expected size and valid bounds.
    """

    indeces = np.asarray(indeces)

    if indeces.ndim != 1:
        return False

    if indeces.size != expected_size:
        return False

    if indeces.size == 0:
        return True

    return (
        np.all(indeces >= 0)
        and np.all(indeces < number_of_points)
    )

def get_data_split_indeces(N, params):
    """Load a compatible data split or generate and save a new one.

    Parameters
    ----------
    N : int
        Total number of data points.
    params : dict
        Configuration parameters.

    Returns
    -------
    tuple
        Training, validation, and test indices followed by their sizes.
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
                valid_indeces(idx_train, N_train_data, N)
                and valid_indeces(idx_val, N_val_data, N)
                and valid_indeces(idx_test, N_test_data, N)
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
    """Create or load indices for training collocation points.

    Parameters
    ----------
    N_coll : int
        Total number of collocation points.
    params : dict
        Configuration parameters.

    Returns
    -------
    numpy.ndarray
        Collocation-point indices.
    """

    # File for loading collocation ponts
    idxc_file = Path(params['paths']['samples']) / "idx_train_coll.npy"

    if idxc_file.exists() and not params['run']['routines']['sampling']:
        idxc = np.load(idxc_file)

        if idxc.shape[0] != N_coll:
            raise ValueError(
                "Loaded collocation indeces have a different size from Xf. "
                f"Expected {N_coll}, got {idxc.shape[0]}."
            )

        if idxc.size > 0 and np.max(idxc) >= N_coll:
            raise ValueError(
                "Loaded collocation indeces are not compatible with Xf."
            )

        if idxc.size > 0 and np.any(idxc < 0):
            raise ValueError(
                "Loaded collocation indeces contain negative values."
            )

    else:
        rng = np.random.default_rng(params.get("seed", 1234))
        idxc = rng.choice(N_coll, N_coll, replace=False)
        np.save(idxc_file, idxc)

    return idxc

def prepare_data(X, U, rho, p, mut, Xf, params):
    """Prepare field arrays for training, validation, test, and physics.

    Parameters
    ----------
    X : np.ndarray
        Coordinates of the data points.
    U : np.ndarray
        Velocity field.
    rho : np.ndarray
        Density field.
    p : np.ndarray
        Pressure field.
    mut : np.ndarray
        Eddy viscosity field.
    Xf : np.ndarray or None
        Collocation points.
    params : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Prepared data and collocation arrays.
    """
    
    # Note that, the number of points inside the geometry is not the same
    # number of the points provided in the configuration file.
    
    # Data points
    N = X.shape[0]

    # Rearrange Data 
    x = X[:,0]   # N 
    y = X[:,1]   # N
    rho = rho[:] # N
    u = U[:,0]   # N
    v = U[:,1]   # N
    p = p[:]     # N
    mut = mut[:] # N: eddy viscosity

    # Get data split indeces 
    (idx_train, idx_val, idx_test, N_train_data, N_val_data, 
        N_test_data) = get_data_split_indeces(N, params)
    
    # Training data
    xtrain = x[idx_train, None]
    ytrain = y[idx_train, None]
    rhotrain = rho[idx_train, None]
    utrain = u[idx_train, None]
    vtrain = v[idx_train, None]
    ptrain = p[idx_train, None]
    muttrain = mut[idx_train, None]

    # Validation data
    xval = None
    yval = None
    rhoval = None
    uval = None
    vval = None
    pval = None
    mutval = None

    if N_val_data > 0:
        xval = x[idx_val, None]
        yval = y[idx_val, None]
        rhoval = rho[idx_val, None]
        uval = u[idx_val, None]
        vval = v[idx_val, None]
        pval = p[idx_val, None]
        mutval = mut[idx_val, None]

    # Test data
    xtest = None
    ytest = None
    rhotest = None
    utest = None
    vtest = None
    ptest = None
    muttest = None

    if N_test_data > 0:
        xtest = x[idx_test, None]
        ytest = y[idx_test, None]
        rhotest = rho[idx_test, None]
        utest = u[idx_test, None]
        vtest = v[idx_test, None]
        ptest = p[idx_test, None]
        muttest = mut[idx_test, None]

    # Collocation points initialisation
    xf = None
    yf = None
    xftrain = None
    yftrain = None

    # Collocation points
    if Xf is not None:  
        # Collocation points
        N_coll = Xf.shape[0]
        xf = Xf[:,0]   # N 
        yf = Xf[:,1]   # N

        # Get collocation split indeces 
        idxc = get_collocation_indeces(N_coll, params)

        xftrain = xf[idxc, None]
        yftrain = yf[idxc, None]

    # All data points
    all = {
        "x": x,
        "y": y,
        "rho": rho,
        "u": u,
        "v": v,
        "p": p,
        "mut": mut,
    }

    training_data = {
        "xtrain": xtrain,
        "ytrain": ytrain,
        "rhotrain": rhotrain,
        "utrain": utrain,
        "vtrain": vtrain,
        "ptrain": ptrain,
        "muttrain": muttrain,
    }

    validation_data = {
        "xval": xval,
        "yval": yval,
        "rhoval": rhoval,
        "uval": uval,
        "vval": vval,
        "pval": pval,
        "mutval": mutval,
    }


        # Test data
    test_data = {
        "xtest": xtest,
        "ytest": ytest,
        "rhotest": rhotest,
        "utest": utest,
        "vtest": vtest,
        "ptest": ptest,
        "muttest": muttest,
    }

    collocation_data = {
        "xf": xf,
        "yf": yf,
        "xftrain": xftrain,
        "yftrain": yftrain,
    }

    # Print dataset information
    print("---------------------------------------")
    print("Dataset information")
    print(f"  Training data points           : {N_train_data}")
    print(f"  Validation data points         : {N_val_data}")
    print(f"  Test data points               : {N_test_data}")
    if xftrain is not None:
        print(f"  Training collocation points    : {N_coll}")
    
    return {
        "all": all,
        "training": training_data,
        "validation": validation_data,
        "test": test_data,
        "collocation": collocation_data,
    }