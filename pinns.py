import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(1234)
np.random.seed(1234)

class PhysicsInformedNN(nn.Module):
    # Initialize the class (Constructor)
    def __init__(
        self,
        x_data, y_data, rho_data, u_data, v_data, p_data, # data points
        x_f, y_f, # Collocation points (only coordinates)
        params,
        mut_data=None # for RANS
    ):
        super().__init__()

        device = torch.device(params.get("device", "cpu"))
        self.device = device

        # Equation
        self.eq = params['equation']
        # Layers
        self.layers = params["layers"]
        # Heat capacity ratio
        self.gamma = float(params["gamma"])
        # Reference parameters (Euler and RANS)
        self.Lref    = float(params["Lref"])
        self.rhoref = float(params["rho"])
        self.Uref   = float(params["U_0"])
        self.pref   = self.rhoref * self.Uref * self.Uref

        if self.eq == 'RANS':
            # Universal gas constant
            self.R = float(params["R"])
            # Prandtl number
            self.Pr = float(params["Pr"])
            # Turbulent Prandtl number 
            self.Prt = float(params["Prt"])
            # Molecular dynamic viscosity
            self.muref  = float(params["mu"])
            # Temperature
            self.Tref = self.Uref**2 / self.R
            # Reynolds number
            self.Re = self.rhoref * self.Uref * self.Lref / self.muref
            # Starred quatities (Sutherland's)
            self.T0star = float(params["T0"]) / self.Tref
            self.Sstar  = float(params["S"]) / self.Tref
            self.mu0star = float(params["mu0"]) / self.muref

        # Check the output layer
        if self.eq == 'Euler':
            expected_out = 4
        elif self.eq == 'RANS':
            expected_out = 5
        else:
            raise ValueError(f"Unknown equation type: {self.eq}")

        if self.layers[-1] != expected_out:
            raise ValueError(
                f"For equation='{self.eq}', last layer must be {expected_out}, "
                f"but got {self.layers[-1]}")
        
        # Non-dimensional data for optimisation performance
        xstar, ystar, rhostar, ustar, vstar, pstar, mutstar = \
            self.GetNonDimensionalData(x, y, rho, u, v , p, mut)

        # Rescaling mut (turbulent viscosity for marchine learning)
        if self.eq == 'RANS':
            if mutstar is None:
                raise ValueError("For equation='RANS', mut must be provided.")

            # Second scaling for ML only
            self.mut_scale = float(np.percentile(mutstar, 95.0))
            if self.mut_scale <= 0.0:
                self.mut_scale = 1.0

            # Scaled target used in the supervised loss
            muthat = mutstar / self.mut_scale   

        X = np.concatenate([xstar, ystar], 1)

        self.lb = torch.tensor(X.min(0), dtype=torch.float32, device=device)  # (2,)
        self.ub = torch.tensor(X.max(0), dtype=torch.float32, device=device)  # (2,)

        # Spatial coordinates
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.x = self.X[:,0:1]
        self.y = self.X[:,1:2]

        # Variables
        self.u = torch.tensor(ustar, dtype=torch.float32, device=device)
        self.v = torch.tensor(vstar, dtype=torch.float32, device=device)
        self.rho = torch.tensor(rhostar, dtype=torch.float32, device=device)
        self.p = torch.tensor(pstar, dtype=torch.float32, device=device)
        if self.eq == 'RANS': 
            self.mut = torch.tensor(muthat, dtype=torch.float32, device=device)        

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(self.layers)

        # Optional LBFGS
        self.optimizer = torch.optim.LBFGS(
            self.trainable_parameters(),
            max_iter=params.get("lbfgs_maxiter", 50000),
            history_size=params.get("lbfgs_history", 50),
            line_search_fn="strong_wolfe",
            tolerance_grad=params.get("lbfgs_tol_grad", 1e-12),
            tolerance_change=params.get("lbfgs_tol_change", 1e-12))

        # Optimizers
        self.optimizer_Adam = torch.optim.Adam(self.trainable_parameters(), 
            lr=params.get("lr", 1e-3))

    def GetNonDimensionalData(self, x, y, rho, u, v, p, mut=None):
        """
        Generate non-dimensional data
        """

        xstar = x / self.Lref
        ystar = y / self.Lref
        rhostar = rho / self.rhoref
        ustar = u / self.Uref
        vstar = v / self.Uref
        pstar = p / (self.rhoref * self.Uref * self.Uref)
        # Turbulent dynamic viscosity (this quatity came form CFD RANS)
        if self.eq == 'RANS':
            if mut is None:
                raise ValueError("For equation='RANS', mut must be provided.")
            mutstar = mut / self.muref
        elif self.eq == 'Euler':
            mutstar = None
 
        return xstar, ystar, rhostar, ustar, vstar, pstar, mutstar

    def initialize_NN(self, layers):
        """
        Returns (weights, biases) 
        weights[i]: (layers[i], layers[i+1])
        biases[i]:  (1, layers[i+1])
        """
        
        weights = nn.ParameterList()
        biases = nn.ParameterList()

        for l in range(len(layers) - 1):
            
            in_dim, out_dim = layers[l], layers[l + 1]
            W = nn.Parameter(torch.empty(in_dim, out_dim, device=self.device))
            b = nn.Parameter(torch.zeros(1, out_dim, device=self.device))

            # Xavier init
            nn.init.xavier_normal_(W)
            weights.append(W)
            biases.append(b)

        return weights, biases

    def neural_net(self, X, weights, biases):
        """
        H = 2*(X-lb)/(ub-lb) - 1
        for each layer: H = tanh(HW + b), last layer linear
        """
        
        # Scale inputs to [-1, 1]
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

        num_layers = len(weights) + 1
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = torch.tanh(H @ W + b)

        W = weights[-1]
        b = biases[-1]
        Y = H @ W + b
        return Y

    def grad(self, y, x):
        """
        dy/dx with graph retention.
        """
        return torch.autograd.grad(
            y, x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

    def net_fields(self, x, y):
        """
        Returns:
        Euler -> rho, u, v, p
        RANS  -> rho, u, v, p, mut
        No deriviatives.
        Used by predict(...) and net_steady_euler(...)
        """
        X = torch.cat([x, y], dim=1)
        out = self.neural_net(X, self.weights, self.biases)

        # Common variables
        raw_rho = out[:,0:1]
        u       = out[:,1:2]
        v       = out[:,2:3]
        raw_p   = out[:,3:4]

        # Enforce positivity of rho and p
        rho = torch.nn.functional.softplus(raw_rho) + 1e-9
        p   = torch.nn.functional.softplus(raw_p) + 1e-9

        if self.eq == 'Euler':
            return rho, u, v, p
        
        elif self.eq == 'RANS':
            raw_mut = out[:,4:5]
            muthat = torch.nn.functional.softplus(raw_mut) + 1e-9
            return rho, u, v, p, muthat
        
        else:
            raise ValueError(f"Unknown equation type: {self.eq}")

    def net_steady_euler(self, x, y):
        """
        Network outputs: rho, u, v, p (primitive variables).
        PDE residuals are steady compressible Euler in conservative form:
          div([rho*u, rho*v]) = 0
          d/dx(rho u^2 + p) + d/dy(rho u v) = 0
          d/dx(rho u v) + d/dy(rho v^2 + p) = 0
          d/dx(u(rhoE+p)) + d/dy(v(rhoE+p)) = 0
        """
        
        # Need gradients wrt x,y
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)

        # Get forward pass: starred quantities 
        rho, u, v, p = self.net_fields(x, y)

        # Heat capacity ratio
        gamma = self.gamma
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
        f1 = self.grad(F1, x) + self.grad(G1, y)
        f2 = self.grad(F2, x) + self.grad(G2, y)
        f3 = self.grad(F3, x) + self.grad(G3, y)
        f4 = self.grad(F4, x) + self.grad(G4, y)

        return rho, u, v, p, f1, f2, f3, f4

    def net_steady_compressible_rans(self, x, y):
        """
        Steady compressible Navier–Stokes with prescribed eddy viscosity, or
        RANS residual with frozen turbulent viscosity field.
        Non-dimensional formulation.
        """

        # Need gradients wrt x,y
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)

        # Get forward pass: starred fields
        rho, u, v, p, muthat = self.net_fields(x, y)

        # Recover physical nondimensional eddy viscosity
        mutstar = self.mut_scale * muthat

        # Heat capacity ratio
        gamma = self.gamma
        # Internal energy
        estar = p / ((gamma - 1.0) * rho)
        # Total Energy
        Estar = estar + 0.5 * (u**2 + v**2)
        # Enthalpy
        Hstar = rho * Estar + p
        # Temperature (non-dimensional form)
        Tstar = p / rho

        # Dynamic viscosity (Sutherland)
        mustar = self.mu0star * (Tstar / self.T0star) ** 1.5 * ((self.T0star + self.Sstar) \
            / (Tstar + self.Sstar))
        
        # Effective viscosity
        mueffstar = mustar + mutstar

        # Effective conductivity (requires some calculations)
        keffstar = (self.gamma / (self.gamma - 1.0)) * ((mustar / self.Pr) \
            + (mutstar / self.Prt))

        # Derivatives
        ux = self.grad(u, x)
        vx = self.grad(v, x)
        uy = self.grad(u, y)
        vy = self.grad(v, y)

        # Viscous stress tensor
        tauxx = (mueffstar / self.Re) * ((4.0/3.0) * ux - (2.0/3.0) * vy)
        tauyy = (mueffstar / self.Re) * ((4.0/3.0) * vy - (2.0/3.0) * ux)
        tauxy = (mueffstar / self.Re) * (uy + vx)

        # Conductivity heat        
        qx = - (keffstar / self.Re) * self.grad(Tstar, x)
        qy = - (keffstar / self.Re) * self.grad(Tstar, y)

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
        f1 = self.grad(Fc1, x) + self.grad(Gc1, y)
        f2 = self.grad(Fc2 - Fv2, x) + self.grad(Gc2 - Gv2, y)
        f3 = self.grad(Fc3 - Fv3, x) + self.grad(Gc3 - Gv3, y)
        f4 = self.grad(Fc4 - Fv4, x) + self.grad(Gc4 - Gv4, y)

        # Note that, at the end, it returns muthat, not mutstar, 
        # because the loss is on the scaled variable
        return rho, u, v, p, muthat, f1, f2, f3, f4

    def loss_fn(self, x, y, rho_t, u_t, v_t, p_t, mut_t=None, return_terms=False):
        """
        Loss function for PINNs regarding the steady euler and RANS equations
        """
        if (self.eq == 'Euler'):
            rho_pred, u_pred, v_pred, p_pred, \
                f1_res, f2_res, f3_res, f4_res \
                = self.net_steady_euler(x, y)
            # Loss of the turbulent viscosity is zero for Euler
            l_mut = torch.tensor(0.0, device=self.device)
            
        elif (self.eq == 'RANS'):        
            rho_pred, u_pred, v_pred, p_pred, mut_pred, \
                f1_res, f2_res, f3_res, f4_res \
                = self.net_steady_compressible_rans(x, y)
            
            if mut_t is None:
                raise ValueError("For RANS, mut_t must be provided in loss_fn.")
            # Loss of the turbulent viscosity         
            l_mut = torch.mean((mut_t - mut_pred) ** 2)
        
        else:
            raise ValueError(f"Unknown equation type: {self.eq}")

        # Data losses
        l_rho = torch.mean((rho_t - rho_pred) ** 2)
        l_u   = torch.mean((u_t   - u_pred)   ** 2)
        l_v   = torch.mean((v_t   - v_pred)   ** 2)
        l_p   = torch.mean((p_t   - p_pred)   ** 2)

        # PDE residual losses
        l_f1  = torch.mean(f1_res ** 2)
        l_f2  = torch.mean(f2_res ** 2)
        l_f3  = torch.mean(f3_res ** 2)
        l_f4  = torch.mean(f4_res ** 2)

        # weights
        w_mut = 1.0 if self.eq == 'RANS' else 0.0

        # weights: chosen based on residuals
        if self.eq == 'Euler':
            w_f1, w_f2, w_f3, w_f4 = 1.0, 1.0, 1.0, 1.0
        else:
            w_f1, w_f2, w_f3, w_f4 = 1.0, 1.0, 1.0, 1.0
        
        # Total loss
        loss = (
            l_rho
            + l_u
            + l_v
            + l_p
            + w_mut * l_mut
            + w_f1 * l_f1
            + w_f2 * l_f2
            + w_f3 * l_f3
            + w_f4 * l_f4
        )

        if return_terms:
            return loss, l_rho, l_u, l_v, l_p, l_mut, l_f1, l_f2, l_f3, l_f4

        return loss

    def trainable_parameters(self):
        """
        Return something optimizers can consume, keeping the 
        'weights/biases' structure.
        """
        return list(self.weights) + list(self.biases)

    def train(self, nIter, use_lbfgs=False, print_every=10):
        """
        Train the Physics-Informed Neural Network (PINN) parameters using Adam,
        with an optional L-BFGS refinement stage.

        Parameters
        ----------
        nIter : int
            Number of Adam optimization iterations (gradient steps).
        use_lbfgs : bool, optional
            If True, run an L-BFGS refinement stage after Adam (default: False).
            Requires `self.optimizer` to be a torch.optim.LBFGS instance.
        print_every : int, optional
            Print training progress every `print_every` Adam iterations (default: 10).

        Returns
        -------
        None
            The function updates model parameters in-place. If L-BFGS is enabled, it prints
            the final L-BFGS loss.
            
        """

        # Adam loop
        for it in range(nIter):
            self.optimizer_Adam.zero_grad()
            
            if self.eq == 'Euler':
                loss = self.loss_fn(self.x, self.y, self.rho, self.u, self.v, self.p)
            elif self.eq == 'RANS':
                loss = self.loss_fn(self.x, self.y, self.rho, self.u, self.v, self.p, \
                                    self.mut)
            
            loss.backward()
            self.optimizer_Adam.step()

            if it % print_every == 0:
                if self.eq == 'Euler':
                    loss_val, l_rho, l_u, l_v, l_p, l_mut, l_f1, l_f2, l_f3, l_f4 = \
                        self.loss_fn(self.x, self.y, self.rho, self.u, self.v, self.p, 
                                     return_terms=True)

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

                if self.eq == 'RANS':
                    loss_val, l_rho, l_u, l_v, l_p, l_mut, l_f1, l_f2, l_f3, l_f4 = \
                        self.loss_fn(self.x, self.y, self.rho, self.u, self.v, self.p,
                                     self.mut, return_terms=True)

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

        # LBFGS refinement (optional)
        if use_lbfgs:
            def closure():
                self.optimizer.zero_grad()
                if self.eq == 'Euler':
                    loss = self.loss_fn(self.x, self.y, self.rho, self.u, self.v, self.p)
                elif self.eq == 'RANS':
                    loss = self.loss_fn(self.x, self.y, self.rho, self.u, self.v, self.p, self.mut)
                loss.backward()
                return loss

            loss = self.optimizer.step(closure)
            print(f"LBFGS final loss: {float(loss):.3e}")

    @torch.no_grad()
    def predict(self, x, y):
        """
        Decorator
        ---------
        @torch.no_grad()
            Disables PyTorch autograd inside this function. This:
            - reduces memory usage,
            - speeds up inference,
            - prevents building computation graphs (no backward possible).

        Parameters
        ----------
        x : array_like
            x-coordinates of query points. Accepted shapes:
            - (N,)  : 1D array of N points
            - (N, 1): column vector of N points
            Will be converted internally to a NumPy array, then to a torch.Tensor
            on `self.device`.
        y : array_like
            y-coordinates of query points. Accepted shapes:
            - (N,)
            - (N, 1)
            Must contain the same number of points N as `x`.

        Returns
        -------
        rho : numpy.ndarray
            Predicted density ρ̂ at the input points, shape (N, 1).
        u : numpy.ndarray
            Predicted x-velocity û at the input points, shape (N, 1).
        v : numpy.ndarray
            Predicted y-velocity v̂ at the input points, shape (N, 1).
        p : numpy.ndarray
            Predicted pressure p̂ at the input points, shape (N, 1).
        """

        x = np.asarray(x) / self.Lref
        y = np.asarray(y) / self.Lref
        if x.ndim == 1: x = x[:, None]
        if y.ndim == 1: y = y[:, None]

        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)

        if self.eq == 'Euler':
            rho, u, v, p = self.net_fields(x_t, y_t)

            rhod, ud, vd, pd = self.GetDimensionalData(rho, u, v, p)
            return (
                rhod.cpu().numpy(), 
                ud.cpu().numpy(), 
                vd.cpu().numpy(), 
                pd.cpu().numpy()
            )

        elif self.eq == 'RANS':
            rho, u, v, p, muthat = self.net_fields(x_t, y_t)
            
            # Rescaling back to mutstar
            mutstar = self.mut_scale * muthat

            rhod, ud, vd, pd, mutd = self.GetDimensionalData(rho, u, v, p, mutstar)
            
            return (
                rhod.cpu().numpy(),
                ud.cpu().numpy(),
                vd.cpu().numpy(),
                pd.cpu().numpy(),
                # Change to also return mutd
                #mutd.cpu().numpy()
            )

    def GetDimensionalData(self, rho, u, v, p, mut=None):
        """
        Make the data dimensional again
        """
        
        rhod = rho * self.rhoref
        ud = u * self.Uref
        vd = v * self.Uref
        pd = p * (self.rhoref * self.Uref * self.Uref)

        if self.eq == 'Euler':
            return rhod, ud, vd, pd
        elif self.eq == 'RANS':
            mutd = mut * self.muref
            return rhod, ud, vd, pd, mutd 

    def callback(self, it, loss_value):
        print(f"It: {it}, Loss: {loss_value:.3e}")