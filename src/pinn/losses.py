import torch

from .residuals import steady_euler_residuals
from .residuals import steady_compressible_rans_residuals

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

def data_loss_terms(pinn, x, y, rho_true, u_true, v_true, p_true,
                   mut_true=None, use_dropout=False):
    """
    Compute supervised data loss terms.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Physics-informed neural network object. It must provide the method
        `net_fields`.

    x, y : torch.Tensor
        Coordinate tensors.

    rho_true, u_true, v_true, p_true : torch.Tensor
        Reference non-dimensional density, velocity components and pressure.

    mut_true : torch.Tensor or None, optional
        Reference scaled turbulent viscosity. Required only for RANS.

    use_dropout : bool, optional
        If True, data predictions are computed with dropout activated in the
        selected layers.

    Returns
    -------
    l_rho, l_u, l_v, l_p, l_mut : torch.Tensor
        Individual mean-squared-error loss terms.
    """

    if pinn.eq == 'Euler': 
        rho_pred, u_pred, v_pred, p_pred = \
            pinn.net_fields(x, y, use_dropout)        
        # is not RANS
        l_mut = zero_loss(pinn)

    elif pinn.eq == 'RANS':
        rho_pred, u_pred, v_pred, p_pred, mut_pred \
            = pinn.net_fields(x, y, use_dropout)

        if mut_true is None:
            raise ValueError("For RANS, mut_t must be provided in loss_fn.")
        # Loss of the turbulent viscosity         
        l_mut = torch.mean((mut_true - mut_pred) ** 2)

    # Data losses terms
    l_rho = torch.mean((rho_true - rho_pred) ** 2)
    l_u   = torch.mean((u_true   - u_pred)   ** 2)
    l_v   = torch.mean((v_true   - v_pred)   ** 2)
    l_p   = torch.mean((p_true   - p_pred)   ** 2)

    return l_rho, l_u, l_v, l_p, l_mut

def residual_loss_terms(pinn):
    """
    Compute PDE residual loss terms.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Physics-informed neural network object. It must provide the model type,
        equation type, collocation tensors, and the physical parameters needed
        by the residual functions.

    Returns
    -------
    l_f1, l_f2, l_f3, l_f4 : torch.Tensor
        Individual PDE residual mean-squared-error loss terms.

    Notes
    -----
    If `pinn.model == "supervised"`, all PDE residual losses are zero.
    """

    if pinn.model == 'supervised':
        z = zero_loss(pinn)
        return z, z, z, z
    
    if pinn.model != 'pinn':
        ValueError(f"Unknown model type: {pinn.model}")

    if pinn.xf is None or pinn.yf is None:
        raise ValueError("PINN mode requires collocation points xf and yf.")

    # Residuals for each equation
    if pinn.eq == 'Euler':
        f1_res, f2_res, f3_res, f4_res \
            = steady_euler_residuals(pinn, pinn.xf, pinn.yf)
        
    elif pinn.eq == 'RANS':
        f1_res, f2_res, f3_res, f4_res \
            = steady_compressible_rans_residuals(pinn, pinn.xf, pinn.yf)
    
    else:
        raise ValueError(f"Unknown equation type: {pinn.eq}")
        
    # PDE residual loss terms
    l_f1  = torch.mean(f1_res ** 2)
    l_f2  = torch.mean(f2_res ** 2)
    l_f3  = torch.mean(f3_res ** 2)
    l_f4  = torch.mean(f4_res ** 2)

    return l_f1, l_f2, l_f3, l_f4

def loss_fn(pinn, return_terms=False):
    """
    Compute the full training loss.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Physics-informed neural network object.

    return_terms : bool, optional
        If True, return the total loss and all individual loss terms.
        If False, return the total loss, data loss and residual loss.

    Returns
    -------
    If return_terms is False:
        loss, data_loss, res_loss : torch.Tensor
            Total loss, weighted supervised data loss, and weighted PDE
            residual loss.

    If return_terms is True:
        loss, l_rho, l_u, l_v, l_p, l_mut, l_f1, l_f2, l_f3, l_f4
            Total loss and individual unweighted loss terms.
    """
    
    # Define dropout
    use_data_dropout = pinn.enable_data_dropout

    # Data loss terms
    l_rho, l_u, l_v, l_p, l_mut = \
        data_loss_terms(pinn, pinn.x, pinn.y, pinn.rho, pinn.u,
                        pinn.v,pinn.p, pinn.mut, use_data_dropout)

    # Residuals loss terms
    l_f1, l_f2, l_f3, l_f4 = residual_loss_terms(pinn)

    # Data loss
    data_loss = (
        pinn.w_rho * l_rho +
        pinn.w_u   * l_u   +
        pinn.w_v   * l_v   +
        pinn.w_p   * l_p   +
        pinn.w_mut * l_mut
    )
    # Residual loss
    res_loss = (
        pinn.w_f1 * l_f1 + 
        pinn.w_f2 * l_f2 + 
        pinn.w_f3 * l_f3 + 
        pinn.w_f4 * l_f4
    )

    # Total loss
    loss = data_loss + res_loss  

    if return_terms:
        return loss, l_rho, l_u, l_v, l_p, l_mut, l_f1, l_f2, l_f3, l_f4

    return loss, data_loss, res_loss

def validation_loss_fn(pinn):
    """
    Validation data loss function based only on supervised data.

    The validation set should not be used to update the weights.
    It is only used to decide when to stop training.
    """

    if not pinn.has_validation:
        return None

    # Use the model in prediction/evaluation mode
    pinn.eval()

    # Do not compute gradients
    with torch.no_grad():
        l_val_rho, l_val_u, l_val_v, l_val_p, l_val_mut = \
            data_loss_terms(pinn, pinn.xval, pinn.yval, pinn.rhoval, pinn.uval,
                            pinn.vval,pinn.pval, pinn.mutval, False)

        l_val = (
            pinn.w_rho * l_val_rho +
            pinn.w_u   * l_val_u   +
            pinn.w_v   * l_val_v   +
            pinn.w_p   * l_val_p   +
            pinn.w_mut * l_val_mut
        )

    # Return to training mode
    pinn.train()

    return l_val
