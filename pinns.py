import numpy as np
import torch
import torch.nn as nn
import os

torch.manual_seed(1234)
np.random.seed(1234)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

class PhysicsInformedNN(nn.Module):
    # Initialize the class (Constructor)
    def __init__(
        self,
        xdata, ydata, rhodata, udata, vdata, pdata, # data points
        xf, yf, # Collocation points (only coordinates)
        params,
        mutdata=None, # for RANS
        xval=None,
        yval=None,
        rhoval=None,
        uval=None,
        vval=None,
        pval=None,
        mutval=None,
    ):
        super().__init__()

        # Device selection
        device_str = params.get("device", None)
        if device_str is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if "cuda" in device_str and not torch.cuda.is_available():
                print("---------------------------------------")
                print("CUDA requested but not available. Falling back to CPU")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device_str)

        # Architeture
        self.layers = params["layers"]
        # Activation function
        activation = params.get("activation", "tanh").lower()
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Initialise NN - weights and biases
        self.initialise_nn()

        # Model
        self.model = params['model']
        # Equation
        self.eq = params['equation']

        # Check model
        if self.model not in ('supervised', 'pinn'):
            raise ValueError(f"Unknown model type: {self.model}")

        # Check the equation
        if self.eq == 'Euler':
            expected_out = 4
        elif self.eq == 'RANS':
            expected_out = 5
        else:
            raise ValueError(f"Unknown equation type: {self.eq}")

        # Check the output layer
        if self.layers[-1] != expected_out:
            raise ValueError(
                f"For equation='{self.eq}', last layer must be {expected_out}, "
                f"but got {self.layers[-1]}")

        # IO loss function
        self.io_loss = int(params.get('io_loss', 999999))
        if self.io_loss < 0:
            raise ValueError(f"io_loss must be non-negative")

        # Initialisation: 
        # Collocation points
        Xf = None
        # Losses
        self.ldata = []
        self.lres = []
        self.loss = []
        # Epochs
        self.n_epoch = 0
        # Turbulent viscosity
        self.mut = None 

        # Physical parameters
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

        # Non-dimensional data (data points)
        xstar, ystar, rhostar, ustar, vstar, pstar, mutstar = \
            self.get_nondimensional_data(xdata, ydata, rhodata, udata, vdata, pdata,
                                       mutdata)
        
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

        # Data points
        # Data coordiantes
        Xdata = np.concatenate([xstar, ystar], 1)
        # Spatial coordinates
        self.Xdata = torch.tensor(Xdata, dtype=torch.float32, device=self.device)
        self.x = self.Xdata[:,0:1]
        self.y = self.Xdata[:,1:2]
        # Physical variables points (training physical information)
        self.u = torch.tensor(ustar, dtype=torch.float32, device=self.device)
        self.v = torch.tensor(vstar, dtype=torch.float32, device=self.device)
        self.rho = torch.tensor(rhostar, dtype=torch.float32, device=self.device)
        self.p = torch.tensor(pstar, dtype=torch.float32, device=self.device)
        if self.eq == 'RANS': 
            self.mut = torch.tensor(muthat, dtype=torch.float32, device=self.device)        
        else:
            self.mut = None

        if xf is not None and yf is not None and self.model == 'pinn':
            # Non-dimensional coordiantes (collocation points for PINNs)
            xfstar, yfstar = self.get_nondimensional_coord(xf, yf)
            # Data coordiantes
            Xf = np.concatenate([xfstar, yfstar], 1)
            # Spatial coordinates
            self.Xf = torch.tensor(Xf, dtype=torch.float32, device=self.device)
            self.xf = self.Xf[:,0:1]
            self.yf = self.Xf[:,1:2]
        else:
            self.Xf = None
            self.xf = None
            self.yf = None

        # Input bounds for normalization
        # Calculate the lower and upper bound from the union of all training coord.
        if Xf is not None:
            Xall = np.concatenate([Xdata, Xf], axis=0)
        else:
            Xall = Xdata.copy()
        self.lb = torch.tensor(Xall.min(0), dtype=torch.float32, device=self.device)  # (2,)
        self.ub = torch.tensor(Xall.max(0), dtype=torch.float32, device=self.device)  # (2,)

		# Optimizers
        # Adam
        self.use_adam = params.get('use_adam', False)
        if self.use_adam:
            self.n_adam_iter = int(params.get('n_adam_iter', 50000))
            if self.n_adam_iter <= 0:
                raise ValueError("n_adam_iter must be greater than zero")
            learning_rate_adam = float(params.get('lr_adam', 5e-4))
            if learning_rate_adam <= 0:
                raise ValueError("learning rate must be greater than zero")        
            scheduler_size = int(params.get('scheduler_size', 10000))
            scheduler_gamma = float(params.get('scheduler_gamma', 0.5))
             
            # Optimizer
            self.optimizer_adam = torch.optim.Adam(self.parameters(), 
                lr=learning_rate_adam)

            # Scheduler
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer_adam, step_size=scheduler_size, gamma=scheduler_gamma)
            
        else:
            self.n_adam_iter = 0

        # LBFGS
        self.use_lbfgs = params.get('use_lbfgs', False)
        if self.use_lbfgs:
            self.n_lbfgs_iter = int(params.get('n_lbfgs_iter', 10000))
            if self.n_lbfgs_iter <= 0:
                raise ValueError("n_adam_iter must be greater than zero")
            max_iter_lbfgs = int(params.get('max_iter_lbfgs', 20))
            if self.n_lbfgs_iter <= 0:
                raise ValueError("max_iter_lbfgs must be greater than zero")
            learning_rate_lbfgs = int(params.get('lr_lbfgs', 1.0))
            if learning_rate_lbfgs <= 0:
                raise ValueError("L-BFGS learning rate must be greater than zero")

            # Optimizer
            self.optimizer_lbfgs = torch.optim.LBFGS(
                self.parameters(),
                lr=learning_rate_lbfgs,
                max_iter=max_iter_lbfgs,
                history_size=params.get("lbfgs_history", 50),
                line_search_fn="strong_wolfe",
                tolerance_grad=params.get("lbfgs_tol_grad", 1e-7),
                tolerance_change=params.get("lbfgs_tol_change", 1e-9))
        else:
            self.n_lbfgs_iter = 0

        # Validation
        self.has_validation = xval is not None
        self.lval = [] # loss

        if self.has_validation:
            # Non-dimensional data (data points)
            (
                x_valstar, y_valstar, rho_valstar,
                u_valstar, v_valstar, p_valstar,
                mut_valstar,
            ) = self.get_nondimensional_data(
                xval, yval, rhoval,uval, vval, pval, mutval,
            )
            # Rescaling mut (turbulent viscosity for marchine learning)
            if self.eq == 'RANS':
                if mut_valstar is None:
                    raise ValueError("For equation='RANS', mut must be provided.")

                # Second scaling for ML only
                #self.mut_valstarscale = float(np.percentile(mut_valstar, 95.0))
                #if self.mut_valstarscale <= 0.0:
                #    self.mut_valstarscale = 1.0

                # Scaled target used in the supervised loss
                mut_valhat = mut_valstar / self.mut_scale   

            # Data coordiantes
            Xval = np.concatenate([x_valstar, y_valstar], 1)
            # Spatial coordinates
            self.Xval = torch.tensor(Xval, dtype=torch.float32, device=self.device)
            self.xval = self.Xval[:,0:1]
            self.yval = self.Xval[:,1:2]
            self.rhoval = torch.tensor(rho_valstar, dtype=torch.float32, device=self.device)
            self.uval = torch.tensor(u_valstar, dtype=torch.float32, device=self.device)
            self.vval = torch.tensor(v_valstar, dtype=torch.float32, device=self.device)
            self.pval = torch.tensor(p_valstar, dtype=torch.float32, device=self.device)
            if self.eq == 'RANS':
                self.mutval = torch.tensor(
                    mut_valhat, dtype=torch.float32, device=self.device
                )
            else:
                self.mutval = None

        # Load model
        if params['load_model']:
            self.load_model(params['pathModel'], params['model_name'])

        # Verbose
        self.verbose = params.get("verbose", False)
        # PhysicsInformedNN setup
        if self.verbose:
            print("---------------------------------------")
            print("PhysicsInformedNN initialized")
            print(f"  Device                       : {self.device}")
            print(f"  Model                        : {self.model}")
            print(f"  Equation                     : {self.eq}")
            print(f"  MLP                          : {self.layers}")
            print(f"  Activation function          : {self.activation}")
            print(f"  Training data points         : {Xdata.shape[0]}")
            if Xf is not None:
                print(f"  Training collocation points  : {Xf.shape[0]}")
            print(f"  Use Adam                     : {self.use_adam}") 
            if self.use_adam:
                print(f"    Number of Adam iteration   : {self.n_adam_iter}")
                print(f"    Adam learning rate         : {learning_rate_adam}")
                print(f"    Scheduler size             : {scheduler_size}")
                print(f"    Learning Reduction rate    : {scheduler_gamma}")
            print(f"  Use L-BFGS                   : {self.use_lbfgs}")
            if self.use_lbfgs:
                print(f"    Number of L-BFGS iteration : {self.n_lbfgs_iter}")
                print(f"    Max iterration for L-BFGS  : {max_iter_lbfgs}")
                print(f"    L-BFGS learning rate       : {learning_rate_lbfgs}")

        # Move the whole module to the selected device
        self.to(self.device)

    def initialise_nn(self):
        
        # Fully connected layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(self.layers) - 1):
            self.hidden_layers.append(
                nn.Linear(self.layers[i], self.layers[i + 1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
       
        if isinstance(m, nn.Linear):
            # Xavier initialization
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)    
    
    def forward(self, X):
        """
        Forward pass
        Normalise input: H = 2*(X-lb)/(ub-lb) - 1
        for each layer: H = tanh(HW + b)
        last layer linear
        """

        # Scale inputs to [-1, 1]
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        
        # Hidden layers with activation
        for layer in self.hidden_layers[:-1]:
            H = self.activation(layer(H))

        # Last layer without activation: linear
        Y = self.hidden_layers[-1](H)

        return Y

    def neural_net(self, X):
        return self.forward(X)

    def net_fields(self, x, y):
        """
        Returns:
        Euler -> rho, u, v, p
        RANS  -> rho, u, v, p, mut
        No deriviatives.
        Used by predict(...) and net_steady_euler(...) and 
        net_steady_compressible_rans(...)
        """
        X = torch.cat([x, y], dim=1)
        out = self.neural_net(X)

        # Common variables
        raw_rho = out[:,0:1]
        u       = out[:,1:2]
        v       = out[:,2:3]
        raw_p   = out[:,3:4]

        # Enforce positivity of rho and p
        rho = torch.nn.functional.softplus(raw_rho) + 1e-8
        p   = torch.nn.functional.softplus(raw_p) + 1e-8

        if self.eq == 'Euler':
            return rho, u, v, p
        
        elif self.eq == 'RANS':
            raw_mut = out[:,4:5]
            # Enforce positivity of muthat
            muthat = torch.nn.functional.softplus(raw_mut) + 1e-8
            return rho, u, v, p, muthat

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

    def loss_fn(self, return_terms=False):
        """
        Loss function for data and PDE
        """

        if self.model == 'pinn':
            
            if self.xf is None or self.yf is None:
                raise ValueError("PINN mode requires collocation points xf and yf.")

            if self.eq == 'Euler':
                # Data
                rho_pred, u_pred, v_pred, p_pred = self.net_fields(self.x, self.y)        
                # Residuals
                _, _, _, _, \
                    f1_res, f2_res, f3_res, f4_res \
                    = self.net_steady_euler(self.xf, self.yf)
                # Loss of the turbulent viscosity is zero for Euler
                l_mut = torch.tensor(0.0, device=self.device)
                
            elif self.eq == 'RANS':
                # Data        
                rho_pred, u_pred, v_pred, p_pred, mut_pred \
                    = self.net_fields(self.x, self.y)
                # Residuals
                _, _, _, _, _, \
                    f1_res, f2_res, f3_res, f4_res \
                    = self.net_steady_compressible_rans(self.xf, self.yf)
                
                if self.mut is None:
                    raise ValueError("For RANS, mut_t must be provided in loss_fn.")
                # Loss of the turbulent viscosity         
                l_mut = torch.mean((self.mut - mut_pred) ** 2)

            # PDE residual losses
            l_f1  = torch.mean(f1_res ** 2)
            l_f2  = torch.mean(f2_res ** 2)
            l_f3  = torch.mean(f3_res ** 2)
            l_f4  = torch.mean(f4_res ** 2)

        elif self.model == 'supervised':
            
            if self.eq == 'Euler': 
                # Data
                rho_pred, u_pred, v_pred, p_pred = self.net_fields(self.x, self.y)        
                # is not RANS
                l_mut = torch.tensor(0.0, device=self.device)

            elif self.eq == 'RANS':
                # Data        
                rho_pred, u_pred, v_pred, p_pred, mut_pred \
                    = self.net_fields(self.x, self.y)
    
                if self.mut is None:
                    raise ValueError("For RANS, mut_t must be provided in loss_fn.")
                # Loss of the turbulent viscosity         
                l_mut = torch.mean((self.mut - mut_pred) ** 2)
 

            # PDE residual losses
            l_f1 = torch.tensor(0.0, device=self.device)
            l_f2 = torch.tensor(0.0, device=self.device)
            l_f3 = torch.tensor(0.0, device=self.device)
            l_f4 = torch.tensor(0.0, device=self.device)
                
                
        # Data losses
        l_rho = torch.mean((self.rho - rho_pred) ** 2)
        l_u   = torch.mean((self.u   - u_pred)   ** 2)
        l_v   = torch.mean((self.v   - v_pred)   ** 2)
        l_p   = torch.mean((self.p   - p_pred)   ** 2)

        # weights: chosen based on residuals
        if self.eq == 'Euler':
            w_f1, w_f2, w_f3, w_f4 = 1.0, 1.0, 1.0, 1.0
            w_mut = 0.0
            
        elif self.eq == 'RANS':
            w_f1, w_f2, w_f3, w_f4 = 1.0, 1.0, 1.0, 1.0
            w_mut = 1.0        

        # data weights
        w_rho = 1.0
        w_u   = 1.0
        w_v   = 8.0
        w_p   = 1.0

        # Data loss
        data_loss = (
            w_rho * l_rho +
            w_u   * l_u   +
            w_v   * l_v   +
            w_p   * l_p   +
            w_mut * l_mut
        )
        # Residual loss
        res_loss = (
            w_f1 * l_f1 + 
            w_f2 * l_f2 + 
            w_f3 * l_f3 + 
            w_f4 * l_f4
        )
        # Total loss
        loss = data_loss + res_loss  

        if return_terms:
            return loss, l_rho, l_u, l_v, l_p, l_mut, l_f1, l_f2, l_f3, l_f4

        return loss, data_loss, res_loss

    def validation_loss_fn(self):
        """
        Validation data loss function based only on supervised data.

        The validation set should not be used to update the weights.
        It is only used to decide when to stop training.
        """

        if not self.has_validation:
            return None

        # Use the model in prediction/evaluation mode
        self.eval()

        # Do not compute gradients
        with torch.no_grad():
            # compute validation data loss only
            if self.eq == 'Euler': 
                # Data
                rho_val_pred, u_val_pred, v_val_pred, p_val_pred = \
                    self.net_fields(self.xval, self.yval)        
                # is not RANS
                l_val_mut = torch.tensor(0.0, device=self.device)
                w_mut = 0.0

            elif self.eq == 'RANS':
                # Data        
                rho_val_pred, u_val_pred, v_val_pred, p_val_pred, mut_val_pred \
                    = self.net_fields(self.xval, self.yval)
    
                if self.mutval is None:
                    raise ValueError("For RANS, mut_t must be provided in loss_fn.")
                # turbulent viscosity data loss       
                l_val_mut = torch.mean((self.mutval - mut_val_pred) ** 2)
                w_mut = 1.0

            # Data losses
            l_val_rho = torch.mean((self.rhoval - rho_val_pred) ** 2)
            l_val_u   = torch.mean((self.uval   - u_val_pred)   ** 2)
            l_val_v   = torch.mean((self.vval   - v_val_pred)   ** 2)
            l_val_p   = torch.mean((self.pval   - p_val_pred)   ** 2)

            # data weights
            w_rho = 1.0
            w_u   = 1.0
            w_v   = 8.0
            w_p   = 1.0

            l_val = (
                w_rho * l_val_rho +
                w_u   * l_val_u   +
                w_v   * l_val_v   +
                w_p   * l_val_p   +
                w_mut * l_val_mut
            )

        # Return to training mode
        self.train()

        return l_val

    def fit(self):
        """
        Train the Physics-Informed Neural Network (PINN) parameters using Adam,
        with L-BFGS refinement stage.

        Returns
        -------
        None
            The function updates model parameters in-place. If L-BFGS is enabled, it prints
            the final L-BFGS loss.            
        """

        # Training=True
        self.train()

        if self.use_adam:
            # Adam loop
            if self.verbose:
                print("---------------------------------------")
                print("Adam optimization")
            
            for it in range(1, self.n_adam_iter + 1):
                self.optimizer_adam.zero_grad()
                # Loss function
                loss, data_loss, res_loss = self.loss_fn()
                # Backward propagation
                loss.backward()
                # Adam step
                self.optimizer_adam.step()
                # Scheduler
                self.scheduler.step()
                
                #Store losses
                self.ldata.append(data_loss.item())
                self.lres.append(res_loss.item())
                self.loss.append(loss.item())
                self.n_epoch += 1

                # validation data loss function
                if self.has_validation:
                    self.lval.append(self.validation_loss_fn().item())

                # Print
                if it % self.io_loss == 0:
                    self.print_loss_fn(it)

        if self.use_lbfgs:
            # L-BFGS loop
            if self.verbose:
                print("---------------------------------------")
                print("L-BFGS optimization")
                    
            for it in range(1, self.n_lbfgs_iter + 1):
                def closure():
                    self.optimizer_lbfgs.zero_grad()
                    loss, _, _ = self.loss_fn()
                    loss.backward()
                    return loss

                self.optimizer_lbfgs.step(closure)

                # recompute once for logging
                loss, data_loss, res_loss = self.loss_fn()
                            
                self.ldata.append(data_loss.item())
                self.lres.append(res_loss.item())
                self.loss.append(loss.item())

                self.n_epoch += 1   # increment once per LBFGS outer step

                # validation data loss function
                if self.has_validation:
                    self.lval.append(self.validation_loss_fn().item())

                # Print
                if it % self.io_loss == 0:
                    self.print_loss_fn(it)
 
    def print_loss_fn(self, it):
    
        if self.model == 'pinn':

            if self.eq == 'Euler':
                loss_val, l_rho, l_u, l_v, l_p, l_mut, l_f1, l_f2, l_f3, l_f4 = \
                    self.loss_fn(return_terms=True)

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

            elif self.eq == 'RANS':
                loss_val, l_rho, l_u, l_v, l_p, l_mut, l_f1, l_f2, l_f3, l_f4 = \
                    self.loss_fn(return_terms=True)

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
        
        elif self.model == 'supervised':
            loss_val, l_rho, l_u, l_v, l_p, l_mut, l_f1, l_f2, l_f3, l_f4 = \
                self.loss_fn(return_terms=True)

            if self.eq == 'Euler':

                print(
                    f"It: {it:6d} | "
                    f"Loss: {loss_val.item():.3e} | "
                    f"rho: {l_rho.item():.3e} | "
                    f"u: {l_u.item():.3e} | "
                    f"v: {l_v.item():.3e} | "
                    f"p: {l_p.item():.3e} "
                )

            elif self.eq == 'RANS':

                print(
                    f"It: {it:6d} | "
                    f"Loss: {loss_val.item():.3e} | "
                    f"rho: {l_rho.item():.3e} | "
                    f"u: {l_u.item():.3e} | "
                    f"v: {l_v.item():.3e} | "
                    f"p: {l_p.item():.3e} | "
                    f"mut: {l_mut.item():.3e}"
                )

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

        # Training=False
        self.eval()

        x = np.asarray(x) / self.Lref
        y = np.asarray(y) / self.Lref
        if x.ndim == 1: x = x[:, None]
        if y.ndim == 1: y = y[:, None]

        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)

        if self.eq == 'Euler':
            rho, u, v, p = self.net_fields(x_t, y_t)

            rhod, ud, vd, pd = self.get_dimensional_data(rho, u, v, p)
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

            rhod, ud, vd, pd, mutd = self.get_dimensional_data(rho, u, v, p, mutstar)
            
            return (
                rhod.cpu().numpy(),
                ud.cpu().numpy(),
                vd.cpu().numpy(),
                pd.cpu().numpy(),
                # Change to also return mutd
                #mutd.cpu().numpy()
            )

    def get_dimensional_data(self, rho, u, v, p, mut=None):
        """
        Make the data dimensional
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

    def get_nondimensional_data(self, x, y, rho, u, v, p, mut=None):
        """
        Generate non-dimensional data (data points)
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

    def get_nondimensional_coord(self, x, y):
        """
        Generate non-dimensional coordinates (collocation points)
        """

        xstar = x / self.Lref
        ystar = y / self.Lref
 
        return xstar, ystar

    def save_model(self, filepath, filename):

        # Save adam optimzer
        optimizer_adam_state = (
            self.optimizer_adam.state_dict()
            if hasattr(self, "optimizer_adam")
            else None
        )

        # Save lbfgs optimizer
        optimizer_lbfgs_state = (
            self.optimizer_lbfgs.state_dict()
            if hasattr(self, "optimizer_lbfgs") and self.optimizer_lbfgs is not None
            else None
        )

        # Save scheduler for adam optimizer
        scheduler_state = (
            self.scheduler.state_dict()
            if hasattr(self, "scheduler") and self.scheduler is not None
            else None
        )

        # Create checkpoint
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_adam_state_dict": optimizer_adam_state,
            "optimizer_lbfgs_state_dict": optimizer_lbfgs_state,
            "scheduler_state_dict": scheduler_state,
            "n_epoch": getattr(self, "n_epoch", 0),
            "loss": getattr(self, "loss", []),
            "ldata": getattr(self, "ldata", []),
            "lres": getattr(self, "lres", []),
            "params": self.params if hasattr(self, "params") else None,
            "lb": self.lb.detach().cpu() if torch.is_tensor(self.lb) else self.lb,
            "ub": self.ub.detach().cpu() if torch.is_tensor(self.ub) else self.ub,
        }

        # Path
        fullpath = os.path.join(filepath, filename)
        # Save model
        torch.save(checkpoint, fullpath + ".pth")
        print("---------------------------------------")                
        print(f"Model saved to: {fullpath}")

    def load_model(self, filepath, filename):
        
        # Path
        fullpath = os.path.join(filepath, filename)
        # Load checkpoint
        checkpoint = torch.load(fullpath + '.pth',
                                map_location=self.device)

        self.load_state_dict(checkpoint["model_state_dict"])

        if checkpoint.get("optimizer_adam_state_dict") is not None and hasattr(self, "optimizer_adam"):
            self.optimizer_adam.load_state_dict(checkpoint["optimizer_adam_state_dict"])

        if (checkpoint.get("optimizer_lbfgs_state_dict") is not None 
            and hasattr(self, "optimizer_lbfgs") 
            and self.optimizer_lbfgs is not None):
            self.optimizer_lbfgs.load_state_dict(checkpoint["optimizer_lbfgs_state_dict"])

        if (checkpoint.get("scheduler_state_dict") is not None 
            and hasattr(self, "scheduler") 
            and self.scheduler is not None):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.n_epoch = checkpoint.get("n_epoch", 0)
        self.loss = checkpoint.get("loss", [])
        self.ldata = checkpoint.get("ldata", [])
        self.lres = checkpoint.get("lres", [])

        if "lb" in checkpoint:
            self.lb = checkpoint["lb"].to(self.device) if torch.is_tensor(checkpoint["lb"]) else checkpoint["lb"]
        if "ub" in checkpoint:
            self.ub = checkpoint["ub"].to(self.device) if torch.is_tensor(checkpoint["ub"]) else checkpoint["ub"]

        print("---------------------------------------")
        print(f"Model loaded from: {filepath}")

    def get_data_loss(self):
        return self.ldata

    def get_residual_loss(self):
        return self.lres

    def get_total_loss(self):
        return self.loss
    
    def get_validation_data_loss(self):
        return self.lval
    
    def get_n_epoch(self):
        return self.n_epoch
    
    def callback(self, it, loss_value):
        print(f"It: {it}, Loss: {loss_value:.3e}")