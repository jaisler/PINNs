# SPDX-License-Identifier: MIT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .losses import loss_fn, validation_loss_fn
from ..utils import print_loss, compute_metrics

torch.manual_seed(1234)
np.random.seed(1234)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

class PhysicsInformedNN(nn.Module):
    # Initialize the class (Constructor)
    def __init__(
        self,
        network,
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

        # Model
        self.model = params['model']
        # Equation
        self.eq = params['equation']

        # Copy network object
        self.network = network

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
        if network.layers[-1] != expected_out:
            raise ValueError(
                f"For equation='{self.eq}', last layer must be {expected_out}, "
                f"but got {self.layers[-1]}")

        # IO loss function
        self.io_loss = int(params.get('io_loss', 999999))
        if self.io_loss < 0:
            raise ValueError(f"io_loss must be non-negative")

        # Loss weights
        loss_weights = params.get("loss_weights", {})
        # Data
        self.w_rho = float(loss_weights.get("w_rho", 1.0))
        self.w_u   = float(loss_weights.get("w_u", 1.0))
        self.w_v   = float(loss_weights.get("w_v", 1.0))
        self.w_p   = float(loss_weights.get("w_p", 1.0))
        # Residual
        self.w_f1  = float(loss_weights.get("w_f1", 1.0))
        self.w_f2  = float(loss_weights.get("w_f2", 1.0))
        self.w_f3  = float(loss_weights.get("w_f3", 1.0))
        self.w_f4  = float(loss_weights.get("w_f4", 1.0))
        # Equation
        if self.eq == "Euler":
            self.w_mut = 0.0
        elif self.eq == "RANS":
            self.w_mut = float(loss_weights.get("w_mut", 1.0))

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

        # Training data
        Xdata, self.x, self.y, self.rho, self.u, self.v, self.p, self.mut = \
            self.prepare_torch_supervised_data(xdata, ydata, rhodata, udata, 
                                               vdata, pdata, mutdata, True)

        # Validation
        self.has_validation = xval is not None
        if self.has_validation:
            self.lval = [] # loss
            (
                _, self.xval, self.yval, 
                self.rhoval, self.uval, self.vval, 
                self.pval, self.mutval
            ) = self.prepare_torch_supervised_data(xval, yval, rhoval, uval, 
                                                   vval, pval, mutval, False)

        # Collocation
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
            self.optimizer_adam = torch.optim.Adam(network.parameters(), 
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
            learning_rate_lbfgs = float(params.get('lr_lbfgs', 1.0))
            if learning_rate_lbfgs <= 0:
                raise ValueError("L-BFGS learning rate must be greater than zero")

            # Optimizer
            self.optimizer_lbfgs = torch.optim.LBFGS(
                network.parameters(),
                lr=learning_rate_lbfgs,
                max_iter=max_iter_lbfgs,
                history_size=params.get("lbfgs_history", 50),
                line_search_fn="strong_wolfe",
                tolerance_grad=params.get("lbfgs_tol_grad", 1e-7),
                tolerance_change=params.get("lbfgs_tol_change", 1e-9))
        else:
            self.n_lbfgs_iter = 0

        # Load model
        if params['load_model']:
            self.load_model(params['pathModel'], params['model_name'])

        # Verbose
        self.verbose = params.get("verbose", False)
        # PhysicsInformedNN setup
        if self.verbose:
            print("---------------------------------------")
            print("Neural Network initialized")
            print(f"  Device                         : {self.device}")
            print(f"  Learning formulation           : {self.model}")
            print(f"  Equation                       : {self.eq}")
            print(f"  MLP                            : {network.layers}")
            print(f"  Activation function            : {network.activation}")
            print(f"  Training data points           : {Xdata.shape[0]}")
            if Xf is not None:
                print(f"  Training collocation points    : {Xf.shape[0]}")
            if self.has_validation:
                print(f"  Validation data points         : {xval.shape[0]}")
            print(f"  Use Adam                       : {self.use_adam}") 
            if self.use_adam:
                print(f"    Number of Adam iteration     : {self.n_adam_iter}")
                print(f"    Adam learning rate           : {learning_rate_adam}")
                print(f"    Scheduler size               : {scheduler_size}")
                print(f"    Learning Reduction rate      : {scheduler_gamma}")
            print(f"  Use L-BFGS                     : {self.use_lbfgs}")
            if self.use_lbfgs:
                print(f"    Number of L-BFGS iteration   : {self.n_lbfgs_iter}")
                print(f"    Max iterration for L-BFGS    : {max_iter_lbfgs}")
                print(f"    L-BFGS learning rate         : {learning_rate_lbfgs}")
            if network.dropout_p > 0.0:
                print(f"  Dropout:")
                print(f"    Probability                  : {network.dropout_p}")
                print(f"    Hidden layer indices         : {network.dropout_indices}")
            print(f"  Loss weights:")
            print(f"    w_rho                        : {self.w_rho}")
            print(f"    w_u                          : {self.w_u}")
            print(f"    w_v                          : {self.w_v}")
            print(f"    w_p                          : {self.w_p}")
            if self.eq == 'RANS':
                print(f"    w_mut                        : {self.w_mut}")
            print(f"    w_f1                         : {self.w_f1}")
            print(f"    w_f2                         : {self.w_f2}")
            print(f"    w_f3                         : {self.w_f3}")
            print(f"    w_f4                         : {self.w_f4}")
            print(f"  Model I/O:")
            print(f"    Load                         : {params['load_model']}")
            print(f"    Save                         : {params['save_model']}")

        # Move the whole module to the selected device
        self.to(self.device)

    def normalize_input(self, X):
        """
        Input normalisation between [-1,1]

        Parameters
        ----------
        X : torch.Tensor
            Input coordinates.
        """
        return 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, X, use_dropout=False):
        """
        Evaluate the neural network inside the PINN.

        The input coordinates are first normalized using the bounds stored in
        the PINN class. The normalized coordinates are then passed to the
        selected network, for example an MLP or GNN.

        Parameters
        ----------
        X : torch.Tensor
            Input coordinates with shape (N, input_dim).

        use_dropout : bool, optional
            If True, enables dropout inside the network.

        Returns
        -------
        torch.Tensor
            Raw network output.
        """
        # Normalised input coordinates
        X_norm = self.normalize_input(X)

        # Note that, this is self.network.forward(...)
        output = self.network(X_norm, use_dropout=use_dropout)

        return output

    def net_fields(self, x, y, use_dropout=False):
        """
        Predict the physical flow variables at the given coordinates.

        The coordinates are combined, normalized in forward(...), and passed
        through the neural network. Positivity is enforced for density,
        pressure, and, in the RANS case, turbulent viscosity.

        Parameters
        ----------
        x, y : torch.Tensor
            Coordinate tensors with shape (N, 1).

        use_dropout : bool, optional
            If True, enables dropout during the network evaluation.

        Returns
        -------
        tuple of torch.Tensor
            For Euler: rho, u, v, p.
            For RANS: rho, u, v, p, muthat.
        """

        X = torch.cat([x, y], dim=1)
        out = self.forward(X, use_dropout)

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

    def fit(self):
        """
        Train the Physics-Informed Neural Network (PINN) parameters using Adam,
        with L-BFGS refinement stage.

        Returns
        -------
        None
            The function updates model parameters in-place. If L-BFGS is enabled, 
            it prints
            the final L-BFGS loss.            
        """

        # Training=True
        self.train()

        if self.use_adam:
            # Adam loop
            if self.verbose:
                print("---------------------------------------")
                print("Adam optimization")

            # Use dropout for Adam optimisation
            self.enable_data_dropout = True

            for it in range(1, self.n_adam_iter + 1):
                self.optimizer_adam.zero_grad()
                # Loss function
                loss, data_loss, res_loss = loss_fn(self)                
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
                    self.lval.append(validation_loss_fn(self).item())

                # Print
                if it % self.io_loss == 0:
                    print_loss(self, it)

        if self.use_lbfgs:
            # L-BFGS loop
            if self.verbose:
                print("---------------------------------------")
                print("L-BFGS optimization")

            # Do not use dropout for L-FBGS optimisation
            self.enable_data_dropout = False

            for it in range(1, self.n_lbfgs_iter + 1):
                def closure():
                    self.optimizer_lbfgs.zero_grad()
                    loss, _, _ = loss_fn(self)
                    loss.backward()
                    return loss

                self.optimizer_lbfgs.step(closure)

                # recompute once for logging
                loss, data_loss, res_loss = loss_fn(self)
                            
                self.ldata.append(data_loss.item())
                self.lres.append(res_loss.item())
                self.loss.append(loss.item())

                self.n_epoch += 1   # increment once per LBFGS outer step

                # validation data loss function
                if self.has_validation:
                    self.lval.append(validation_loss_fn(self).item())

                # Print
                if it % self.io_loss == 0:
                    print_loss(self, it)

        # After training, disable dropout by default
        self.enable_data_dropout = False
        self.eval()
    
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
                mutd.cpu().numpy()
            )

    def prepare_torch_supervised_data(self, xdata, ydata, rhodata, udata,
                                      vdata, pdata, mutdata=None, 
                                      fit_scale=False):
        """
        Non-dimensionalise physical data, scale mut if needed,
        and convert data to PyTorch tensors.

        If fit_scale=True, compute self.mut_scale from this dataset.
        If fit_scale=False, reuse the existing self.mut_scale.
        """

        # Non-dimensional data
        xstar, ystar, rhostar, ustar, vstar, pstar, mutstar = \
            self.get_nondimensional_data(xdata, ydata, rhodata, udata, vdata, 
                                         pdata, mutdata)
        
        # Turbulent viscosity scaling
        if self.eq == "RANS":
            if mutstar is None:
                raise ValueError("For equation='RANS', mut must be provided.")

            if fit_scale:
                self.mut_scale = float(np.percentile(mutstar, 95.0))

                if self.mut_scale <= 0.0:
                    self.mut_scale = 1.0

            elif not hasattr(self, "mut_scale"):
                raise ValueError(
                    "mut_scale has not been defined. "
                    "Call this method first with fit_scale=True using the training data."
                )
            muthat = mutstar / self.mut_scale
        else:
            muthat = None

        # Coordinates
        Xdata = np.concatenate([xstar, ystar], axis=1)
        Xdata = torch.tensor(Xdata, dtype=torch.float32, device=self.device)
        x = Xdata[:, 0:1]
        y = Xdata[:, 1:2]
        # Physical variables
        rho = torch.tensor(rhostar, dtype=torch.float32, device=self.device)
        u = torch.tensor(ustar, dtype=torch.float32, device=self.device)
        v = torch.tensor(vstar, dtype=torch.float32, device=self.device)
        p = torch.tensor(pstar, dtype=torch.float32, device=self.device)
        if self.eq == "RANS":
            mut = torch.tensor(muthat, dtype=torch.float32, device=self.device)
        else:
            mut = None

        return Xdata, x, y, rho, u, v, p, mut
    
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

        if checkpoint.get("optimizer_adam_state_dict") is not None 
            and hasattr(self, "optimizer_adam"):
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
            self.lb = checkpoint["lb"].to(self.device) \
                if torch.is_tensor(checkpoint["lb"]) else checkpoint["lb"]
        if "ub" in checkpoint:
            self.ub = checkpoint["ub"].to(self.device) \
                if torch.is_tensor(checkpoint["ub"]) else checkpoint["ub"]

        print("---------------------------------------")
        print(f"Model loaded from: {filepath}")

    def evaluate_data(self, xdata, ydata, rhodata,
                      udata, vdata, pdata, mutdata=None):
        """
        Evaluate prediction errors on an external dataset.

        This method can be used for validation or test data.
        It does not update the neural network weights.
        """

        self.eval()

        # Non-dimensionalize data using the same scaling as training
        xstar, ystar, rhostar, ustar, vstar, pstar, mutstar = \
            self.get_nondimensional_data(xdata, ydata, rhodata, udata,
                                         vdata, pdata, mutdata)

        # Coordinates
        X = np.column_stack((xstar, ystar))

        X = torch.tensor(X, dtype=torch.float32, device=self.device)

        rho_true = torch.tensor(rhostar, dtype=torch.float32,
                                device=self.device).reshape(-1, 1)

        u_true = torch.tensor(ustar, dtype=torch.float32,
                              device=self.device,).reshape(-1, 1)

        v_true = torch.tensor(vstar, dtype=torch.float32,
                              device=self.device).reshape(-1, 1)

        p_true = torch.tensor(pstar, dtype=torch.float32,
                              device=self.device,).reshape(-1, 1)

        if self.eq == "RANS":
            if mutstar is None:
                raise ValueError("For RANS evaluation, mutdata must be provided.")

            mut_true = torch.tensor(mutstar, dtype=torch.float32, 
                                    device=self.device).reshape(-1, 1)

        with torch.no_grad():
            x_t = X[:, 0:1]
            y_t = X[:, 1:2]

            if self.eq == "Euler":
                rho_pred, u_pred, v_pred, p_pred = self.net_fields(x_t, y_t)

            elif self.eq == "RANS":
                rho_pred, u_pred, v_pred, p_pred, muthat_pred = \
                    self.net_fields(x_t, y_t)

                # Recover mutstar
                mut_pred = self.mut_scale * muthat_pred

        metrics = {}

        metrics["rho"] = compute_metrics(rho_pred, rho_true)
        metrics["u"]   = compute_metrics(u_pred,   u_true)
        metrics["v"]   = compute_metrics(v_pred,   v_true)
        metrics["p"]   = compute_metrics(p_pred,   p_true)

        if self.eq == "RANS":
            metrics["mut"] = compute_metrics(mut_pred, mut_true)

        return metrics
    
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
