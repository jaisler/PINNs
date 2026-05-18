# SPDX-License-Identifier: MIT
import torch

def compute_metrics(y_pred, y_true, eps=1.0e-12):
    """
    Compute MSE, RMSE, relative L2 error, and R2 score.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted values.

    y_true : torch.Tensor
        Reference values.

    eps : float, optional
        Small value to avoid division by zero.

    Returns
    -------
    dict
        Dictionary with mse, rmse, rel_l2, and r2.
    """

    error = y_pred - y_true

    mse = torch.mean(error**2)
    rmse = torch.sqrt(mse)

    rel_l2 = (
        torch.linalg.norm(error)
        /
        (torch.linalg.norm(y_true) + eps)
    )

    ss_res = torch.sum(error**2)
    ss_tot = torch.sum((y_true - torch.mean(y_true))**2)

    r2 = 1.0 - ss_res / (ss_tot + eps)

    return {
        "mse": mse.item(),
        "rmse": rmse.item(),
        "rel_l2": rel_l2.item(),
        "r2": r2.item(),
    }

def print_metrics_table(metrics, title="Metrics", rel_l2_percent=True):
    """
    Print a formatted table with the metrics for each variable.

    Parameters
    ----------
    metrics : dict
        Dictionary containing the metrics for each variable.

    title : str, optional
        Title of the printed table.

    rel_l2_percent : bool, optional
        If True, print the relative L2 error as a percentage.
    """

    rel_l2_name = "Rel. L2 (%)" if rel_l2_percent else "Rel. L2"

    header = (
        f"{'Variable':<10}"
        f"{'MSE':>14}"
        f"{'RMSE':>14}"
        f"{rel_l2_name:>16}"
        f"{'R2':>14}"
    )

    line_width = len(header)

    print("\n" + "=" * line_width)
    print(f"{title:}")
    print("-" * line_width)
    print(header)
    print("-" * line_width)

    preferred_order = ["rho", "u", "v", "p", "mut"]

    for var in preferred_order:
        if var not in metrics:
            continue

        values = metrics[var]

        mse = values["mse"]
        rmse = values["rmse"]
        rel_l2 = values["rel_l2"]
        r2 = values["r2"]

        if rel_l2_percent:
            rel_l2 = 100.0 * rel_l2

        print(
            f"{var:<10}"
            f"{mse:>14.4e}"
            f"{rmse:>14.4e}"
            f"{rel_l2:>16.4f}"
            f"{r2:>14.4f}"
        )

    print("=" * line_width + "\n")