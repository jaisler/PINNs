# SPDX-License-Identifier: MIT

def training_is_enabled(params):
    """
    Determine whether training is enabled for at least one optimizer.

    Training is considered enabled when either Adam or L-BFGS is enabled
    and its configured number of iterations is greater than zero.

    Parameters
    ----------
    params : dict
        Configuration parameters containing the optimizer settings.

    Returns
    -------
    bool
        ``True`` if Adam or L-BFGS training is enabled with a positive
        number of iterations; otherwise, ``False``.
    """

    adam_enabled = params["optimizer"]["adam"].get("enabled", False)
    lbfgs_enabled = params["optimizer"]["lbfgs"].get("enabled", False)

    adam_iterations = int(
        params["optimizer"]["adam"].get("iterations", 0)
    )

    lbfgs_iterations = int(
        params["optimizer"]["lbfgs"].get("iterations", 0)
    )

    return (
        adam_enabled and adam_iterations > 0
    ) or (
        lbfgs_enabled and lbfgs_iterations > 0
    )


