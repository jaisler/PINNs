# SPDX-License-Identifier: MIT
from ..pinn.losses import loss_fn

def print_loss(pinn, it):
    """Print the current training-loss components.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Model containing data and physical loss terms.
    it : int
        Training iteration.

    Returns
    -------
    None
        Losses are written to standard output.
    """
    if pinn.model == 'pinn':

        if pinn.eq == 'euler':
            loss_val, l_rho, l_u, l_v, l_p, l_mut, l_f1, l_f2, l_f3, l_f4 = \
                loss_fn(pinn, return_terms=True)

            print(
                f"It: {it:6d} | "
                f"Loss: {loss_val.item():.3e} | "
                f"rho: {l_rho.item():.3e} | "
                f"u: {l_u.item():.3e} | "
                f"v: {l_v.item():.3e} | "
                f"p: {l_p.item():.3e} | "
                f"f1: {l_f1.item():.3e} | "
                f"f2: {l_f2.item():.3e} | "
                f"f3: {l_f3.item():.3e} | "
                f"f4: {l_f4.item():.3e}"
            )

        elif pinn.eq == 'rans':
            loss_val, l_rho, l_u, l_v, l_p, l_mut, l_f1, l_f2, l_f3, l_f4 = \
                loss_fn(pinn, return_terms=True)

            print(
                f"It: {it:6d} | "
                f"Loss: {loss_val.item():.3e} | "
                f"rho: {l_rho.item():.3e} | "
                f"u: {l_u.item():.3e} | "
                f"v: {l_v.item():.3e} | "
                f"p: {l_p.item():.3e} | "
                f"mut: {l_mut.item():.3e} | "
                f"f1: {l_f1.item():.3e} | "
                f"f2: {l_f2.item():.3e} | "
                f"f3: {l_f3.item():.3e} | "
                f"f4: {l_f4.item():.3e}"
            )
    
    elif pinn.model == 'supervised':
        loss_val, l_rho, l_u, l_v, l_p, l_mut, l_f1, l_f2, l_f3, l_f4 = \
            loss_fn(pinn, return_terms=True)

        if pinn.eq == 'euler':

            print(
                f"It: {it:6d} | "
                f"Loss: {loss_val.item():.3e} | "
                f"rho: {l_rho.item():.3e} | "
                f"u: {l_u.item():.3e} | "
                f"v: {l_v.item():.3e} | "
                f"p: {l_p.item():.3e} "
            )

        elif pinn.eq == 'rans':

            print(
                f"It: {it:6d} | "
                f"Loss: {loss_val.item():.3e} | "
                f"rho: {l_rho.item():.3e} | "
                f"u: {l_u.item():.3e} | "
                f"v: {l_v.item():.3e} | "
                f"p: {l_p.item():.3e} | "
                f"mut: {l_mut.item():.3e}"
            )
