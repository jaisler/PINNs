# SPDX-License-Identifier: MIT
import torch

from .residuals import steady_euler_residuals
from .residuals import steady_compressible_rans_residuals

def zero_loss(pinn):
    """Create a scalar zero on the model device.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Model providing the target device.

    Returns
    -------
    torch.Tensor
        Scalar zero tensor on `pinn.device`.
    """

    return torch.tensor(0.0, dtype=torch.float32, device=pinn.device)

def data_loss_terms(pinn, x, y, rho_true, u_true, v_true, p_true,
                    mut_true=None, use_dropout=False, role="data"):
    """Compute mean-squared errors for the supervised flow fields.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Model used to predict the fields.
    x, y : torch.Tensor
        Coordinate tensors.
    rho_true, u_true, v_true, p_true : torch.Tensor
        Reference nondimensional flow fields.
    mut_true : torch.Tensor or None, optional
        Reference scaled turbulent viscosity. Required only for RANS.
    use_dropout : bool, optional
        Whether to enable data dropout.
    role : {"data", "validation", "query"} or None
        GNN graph role; ignored by the MLP.

    Returns
    -------
    tuple of torch.Tensor
        Density, velocity, pressure, and viscosity loss terms.
    """

    if pinn.eq == 'euler': 
        rho_pred, u_pred, v_pred, p_pred = \
            pinn.net_fields(x, y, use_dropout, role=role)        
        # is not rans
        l_mut = zero_loss(pinn)

    elif pinn.eq == 'rans':
        rho_pred, u_pred, v_pred, p_pred, mut_pred \
            = pinn.net_fields(x, y, use_dropout, role=role)

        if mut_true is None:
            raise ValueError("For RANS, mut_t must be provided in loss_fn.")
        # Loss of the turbulent viscosity         
        l_mut = torch.mean((mut_true - mut_pred) ** 2)
        
    else:
        raise ValueError(f"Unknown equation type: {pinn.eq}")

    # Data losses terms
    l_rho = torch.mean((rho_true - rho_pred) ** 2)
    l_u   = torch.mean((u_true   - u_pred)   ** 2)
    l_v   = torch.mean((v_true   - v_pred)   ** 2)
    l_p   = torch.mean((p_true   - p_pred)   ** 2)

    return l_rho, l_u, l_v, l_p, l_mut

def residual_loss_terms(pinn):
    """Compute mean-squared PDE residual terms.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Model containing collocation points and physical parameters.

    Returns
    -------
    tuple of torch.Tensor
        Mass, momentum, and energy residual losses. All values are zero for a
        supervised model.
    """

    if pinn.model == 'supervised':
        z = zero_loss(pinn)
        return z, z, z, z
    
    if pinn.model != 'pinn':
        raise ValueError(f"Unknown model type: {pinn.model}")

    if pinn.xf is None or pinn.yf is None:
        raise ValueError("PINN mode requires collocation points xf and yf.")

    # Need gradients wrt x,y
    # Independent coordinate tensors for automatic differentiation
    x = pinn.xf.clone().detach().requires_grad_(True)
    y = pinn.yf.clone().detach().requires_grad_(True)

    # For the GNN, net_fields() inserts these collocation
    # coordinates into the complete training graph.
    if pinn.net_arch == "mlp":
        role = None

    elif pinn.net_arch == "gnn":
        role = "residual"

    else:
        raise ValueError(
            f"Unknown network architecture: {pinn.net_arch}"
        )

    # Residuals for each equation
    if pinn.eq == 'euler':
        rho_pred, u_pred, v_pred, p_pred = \
            pinn.net_fields(x, y, use_dropout=False, role=role)        

        f1_res, f2_res, f3_res, f4_res \
            = steady_euler_residuals(pinn, x, y, rho_pred, u_pred, v_pred, p_pred)
        
    elif pinn.eq == 'rans':
        rho_pred, u_pred, v_pred, p_pred, mut_pred = \
            pinn.net_fields(x, y, use_dropout=False, role=role)                
        
        f1_res, f2_res, f3_res, f4_res \
            = steady_compressible_rans_residuals(pinn, x, y, rho_pred, u_pred, \
                                                 v_pred, p_pred, mut_pred)

    else:
        raise ValueError(f"Unknown equation type: {pinn.eq}")
        
    # PDE residual loss terms
    l_f1  = torch.mean(f1_res ** 2)
    l_f2  = torch.mean(f2_res ** 2)
    l_f3  = torch.mean(f3_res ** 2)
    l_f4  = torch.mean(f4_res ** 2)

    return l_f1, l_f2, l_f3, l_f4

def loss_fn(pinn, return_terms=False):
    """Compute the weighted data and physics training objective.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Model containing predictions, data, and loss weights.
    return_terms : bool, optional
        Whether to return every unweighted component.

    Returns
    -------
    tuple of torch.Tensor
        Total, data, and residual losses, or the total followed by every
        component when ``return_terms`` is true.
    """
    
    # Define dropout
    use_data_dropout = pinn.enable_data_dropout

    # Data loss terms
    l_rho, l_u, l_v, l_p, l_mut = \
        data_loss_terms(pinn, pinn.x, pinn.y, pinn.rho, pinn.u,
                        pinn.v,pinn.p, pinn.mut, use_data_dropout,
                        role="data")

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
    """Compute the weighted supervised validation loss.
   
    Parameters
    ----------
    pinn : PhysicsInformedNN
        Model containing validation data and loss weights.

    Returns
    -------
    torch.Tensor or None
        Validation loss, or ``None`` when validation data are unavailable.
    """

    if not pinn.has_validation:
        return None

    # Save the current PyTorch module mode.
    was_training = pinn.training
    # Use the model in prediction/evaluation mode
    pinn.eval()

    # Do not compute gradients
    try:
        with torch.no_grad():
            l_val_rho, l_val_u, l_val_v, l_val_p, l_val_mut = \
                data_loss_terms(pinn, pinn.xval, pinn.yval, pinn.rhoval, 
                                pinn.uval, pinn.vval, pinn.pval, 
                                pinn.mutval, use_dropout=False, 
                                role="validation")

            l_val = (
                pinn.w_rho * l_val_rho +
                pinn.w_u   * l_val_u   +
                pinn.w_v   * l_val_v   +
                pinn.w_p   * l_val_p   +
                pinn.w_mut * l_val_mut
            )

    finally:
        # Return to training mode
        pinn.train(was_training)

    return l_val
