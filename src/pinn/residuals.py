# SPDX-License-Identifier: MIT
import torch

def grad(y, x):
    """Differentiate a tensor with respect to an input tensor.

    Parameters
    ----------
    y : torch.Tensor
        Dependent variable or flux term.
    x : torch.Tensor
        Independent variable with gradients enabled.

    Returns
    -------
    torch.Tensor
        Derivative of ``y`` with respect to ``x`` with its graph retained.
    """

    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def steady_euler_residuals(pinn, x, y, rho, u, v, p):
    """Compute nondimensional steady Euler residuals.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Model providing the heat-capacity ratio.
    x, y : torch.Tensor
        Collocation coordinates.
    rho, u, v, p : torch.Tensor
        Predicted flow fields.
        
    Returns
    -------
    tuple of torch.Tensor
        Mass, x-momentum, y-momentum, and energy residuals.
    """

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

def steady_compressible_rans_residuals(pinn, x, y, rho, u, v, p, muthat):
    """Compute nondimensional steady compressible RANS residuals.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Model providing physical constants and viscosity scaling.
    x, y : torch.Tensor
        Collocation coordinates.
    rho, u, v, p, muthat : torch.Tensor
        Predicted flow fields.

    Returns
    -------
    tuple of torch.Tensor
        Mass, x-momentum, y-momentum, and energy residuals.
    """

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
    qx = - (keffstar / pinn.Re) * grad(Tstar, x)
    qy = - (keffstar / pinn.Re) * grad(Tstar, y)

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
    f1 = grad(Fc1, x) + grad(Gc1, y)
    f2 = grad(Fc2 - Fv2, x) + grad(Gc2 - Gv2, y)
    f3 = grad(Fc3 - Fv3, x) + grad(Gc3 - Gv3, y)
    f4 = grad(Fc4 - Fv4, x) + grad(Gc4 - Gv4, y)

    return f1, f2, f3, f4
