import torch

def grad(y, x):
    """
    Compute dy/dx using PyTorch automatic differentiation.

    Parameters
    ----------
    y : torch.Tensor
        Dependent variable or flux term. It must depend on `x`.

    x : torch.Tensor
        Independent variable. It must have `requires_grad=True`.

    Returns
    -------
    torch.Tensor
        Derivative of `y` with respect to `x`.

    Notes
    -----
    The computational graph is retained to allow multiple derivative
    evaluations and possible higher-order derivatives.
    """

    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def steady_euler_residuals(pinn, x, y):
    """
    Steady compressible Euler residuals.

    Parameters
    ----------
    pinn :
        PhysicsInformedNN object.
    x, y :
        Collocation coordinates.

    Returns
    -------
    f1, f2, f3, f4
    """

    # Need gradients wrt x,y
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)

    rho, u, v, p = pinn.net_fields(x, y)
    
    # Heat capacity ratio
    gamma = pinn.gamma
    # Internal energy
    e = p / ((gamma - 1.0) * rho)
    # Total Energy
    E = e + 0.5 * (u**2 + v**2)
    # Enthalpy
    H = rho * E + p

    # fluxes
    # Derivative wrt x
    F1 = rho * u 
    F2 = rho * u**2 + p 
    F3 = rho * u * v 
    F4 = u * H
    # Derivative wrt y
    G1 = rho * v
    G2 = rho * u * v
    G3 = rho * v**2 + p
    G4 = v * H
    # Residual
    f1 = grad(F1, x) + grad(G1, y)
    f2 = grad(F2, x) + grad(G2, y)
    f3 = grad(F3, x) + grad(G3, y)
    f4 = grad(F4, x) + grad(G4, y)

    return f1, f2, f3, f4

def steady_compressible_rans_residuals(pinn, x, y):
    """
    Steady compressible RANS residuals. 
    Equations in non-dimensional formulation.

    Parameters
    ----------
    pinn :
        PhysicsInformedNN object.
    x, y :
        Collocation coordinates.

    Returns
    -------
    f1, f2, f3, f4
    """

    # Need gradients wrt x,y
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)

    # Get forward pass: starred fields
    rho, u, v, p, muthat = pinn.net_fields(x, y, False)

    # Recover physical nondimensional eddy viscosity
    mutstar = pinn.mut_scale * muthat

    # Heat capacity ratio
    gamma = pinn.gamma
    # Internal energy
    estar = p / ((gamma - 1.0) * rho)
    # Total Energy
    Estar = estar + 0.5 * (u**2 + v**2)
    # Enthalpy
    Hstar = rho * Estar + p
    # Temperature (non-dimensional form)
    Tstar = p / rho

    # Dynamic viscosity (Sutherland)
    mustar = pinn.mu0star * (Tstar / pinn.T0star) \
        ** 1.5 * ((pinn.T0star + pinn.Sstar) \
        / (Tstar + pinn.Sstar))
    
    # Effective viscosity
    mueffstar = mustar + mutstar

    # Effective conductivity (requires some calculations)
    keffstar = (pinn.gamma / (pinn.gamma - 1.0)) * ((mustar / pinn.Pr) \
        + (mutstar / pinn.Prt))

    # Derivatives
    ux = grad(u, x)
    vx = grad(v, x)
    uy = grad(u, y)
    vy = grad(v, y)

    # Viscous stress tensor
    tauxx = (mueffstar / pinn.Re) * ((4.0/3.0) * ux - (2.0/3.0) * vy)
    tauyy = (mueffstar / pinn.Re) * ((4.0/3.0) * vy - (2.0/3.0) * ux)
    tauxy = (mueffstar / pinn.Re) * (uy + vx)

    # Conductivity heat        
    qx = - (keffstar / pinn.Re) * pinn.grad(Tstar, x)
    qy = - (keffstar / pinn.Re) * pinn.grad(Tstar, y)

    # Convective fluxes 
    # Derivative wrt x
    Fc1 = rho * u 
    Fc2 = rho * u**2 + p 
    Fc3 = rho * u * v 
    Fc4 = u * Hstar
    # Derivative wrt y
    Gc1 = rho * v
    Gc2 = rho * u * v
    Gc3 = rho * v**2 + p
    Gc4 = v * Hstar

    # Viscous fluxes 
    # Derivative wrt x
    Fv1 = torch.zeros_like(rho)
    Fv2 = tauxx 
    Fv3 = tauxy 
    Fv4 = u * tauxx + v * tauxy - qx
    # Derivative wrt y
    Gv1 = torch.zeros_like(rho)
    Gv2 = tauxy 
    Gv3 = tauyy
    Gv4 = u * tauxy + v * tauyy - qy

    # Residual
    f1 = pinn.grad(Fc1, x) + pinn.grad(Gc1, y)
    f2 = pinn.grad(Fc2 - Fv2, x) + pinn.grad(Gc2 - Gv2, y)
    f3 = pinn.grad(Fc3 - Fv3, x) + pinn.grad(Gc3 - Gv3, y)
    f4 = pinn.grad(Fc4 - Fv4, x) + pinn.grad(Gc4 - Gv4, y)

    return f1, f2, f3, f4
