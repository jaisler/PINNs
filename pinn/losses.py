import torch

from .residuals import steady_euler
from .residuals import steady_compressible_rans


def zero_loss(pinn):
    """
    Create a scalar zero tensor on the same device as the PINN model.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Physics-informed neural network object.

    Returns
    -------
    torch.Tensor
        Scalar zero tensor on `pinn.device`.
    """

    return torch.tensor(0.0, dtype=torch.float32, device=pinn.device)