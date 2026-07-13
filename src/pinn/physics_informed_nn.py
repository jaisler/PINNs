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
        self.device_str = params['run'].get("device", None)
        if self.device_str is None:
            if torch.cuda.is_available():
                self.device_str = "cuda"
            else:
                self.device_str = "cpu"
            self.device = torch.device(self.device_str)
        else:
            if "cuda" in self.device_str and not torch.cuda.is_available():
                print("---------------------------------------")
                print("CUDA requested but not available. Falling back to CPU")
                self.device = torch.device("cpu")
                self.device_str = "cpu"
            else:
                self.device = torch.device(self.device_str)

        # Register network as a submodule
        self.network = network
        # Move the full PINN model, including the network, 
        # to the selected device
        self.to(self.device)

        # Model
        self.model = params['run']['model']
        # Equation
        self.eq = params['run']['equation']
        # Network architecture
        self.net_arch = params['network']['architecture']

        # Check network
        if self.net_arch not in ('mlp', 'gnn'):
            raise ValueError(f"Unknown network architecture type: {self.net_arch}")

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
        if self.net_arch == 'mlp':        
            if network.layers[-1] != expected_out:
                raise ValueError(
                    f"For equation='{self.eq}', last layer must be {expected_out}, "
                    f"but got {network.layers[-1]}")

        # IO loss function
        self.io_loss = int(params['loss'].get('print_frequency', 999999))
        if self.io_loss <= 0:
            raise ValueError(f"print_frequency must be greater than zero")

        # Loss weights
        loss_weights = params['loss'].get("weights", {})
        # Data
        self.w_rho = float(loss_weights['data'].get("rho", 1.0))
        self.w_u   = float(loss_weights['data'].get("u", 1.0))
        self.w_v   = float(loss_weights['data'].get("v", 1.0))
        self.w_p   = float(loss_weights['data'].get("p", 1.0))
        # Residual
        self.w_f1  = float(loss_weights['residual'].get("f1", 1.0))
        self.w_f2  = float(loss_weights['residual'].get("f2", 1.0))
        self.w_f3  = float(loss_weights['residual'].get("f3", 1.0))
        self.w_f4  = float(loss_weights['residual'].get("f4", 1.0))
        # Equation
        if self.eq == "Euler":
            self.w_mut = 0.0
        elif self.eq == "RANS":
            self.w_mut = float(loss_weights['data'].get("mut", 1.0))

        # Initialisation: 
        # Collocation points
        Xf = None
        # Loss histories
        self.ldata = []
        self.lres = []
        self.lval = []
        self.loss = []
        # Training history
        self.n_epoch = 0
        self.enable_data_dropout = False
        # Turbulent viscosity
        self.mut = None 

        # Physical parameters
        phys_cfg = params['physics']
        # Heat capacity ratio
        self.gamma = float(phys_cfg['gas']['gamma'])
        # Reference parameters (Euler and RANS)
        self.Lref    = float(phys_cfg['reference']['Lref'])
        self.rhoref = float(phys_cfg['reference']['rho'])
        self.Uref   = float(phys_cfg['reference']['U_0'])
        self.pref   = self.rhoref * self.Uref * self.Uref

        if self.eq == 'RANS':
            # Universal gas constant
            self.R = float(phys_cfg['gas']['R'])
            # Prandtl number
            self.Pr = float(phys_cfg['turbulence']['Pr'])
            # Turbulent Prandtl number 
            self.Prt = float(phys_cfg['turbulence']['Prt'])
            # Molecular dynamic viscosity
            self.muref = float(phys_cfg['reference']['mu'])
            # Temperature
            self.Tref = self.Uref**2 / self.R
            # Reynolds number
            self.Re = self.rhoref * self.Uref * self.Lref / self.muref
            # Starred quatities (Sutherland's)
            self.T0star = float(phys_cfg['sutherland']['T0']) / self.Tref
            self.Sstar  = float(phys_cfg['sutherland']['S']) / self.Tref
            self.mu0star = float(phys_cfg['sutherland']['mu0']) / self.muref

        # Training data
        _, self.x, self.y, self.rho, self.u, self.v, self.p, self.mut = \
            self.prepare_torch_supervised_data(xdata, ydata, rhodata, udata, 
                                               vdata, pdata, mutdata, True)

        # Validation
        # These variables are always required for validation.
        validation_values = [
            xval,
            yval,
            rhoval,
            uval,
            vval,
            pval,
        ]

        # Turbulent viscosity validation data are additionally
        # required for the RANS equations.
        if self.eq == "RANS":
            validation_values.append(mutval)

        self.has_validation = all(
            value is not None
            for value in validation_values
        )        

        if self.has_validation:
            (
                _, self.xval, self.yval, 
                self.rhoval, self.uval, self.vval, 
                self.pval, self.mutval
            ) = self.prepare_torch_supervised_data(xval, yval, rhoval, uval, 
                                                   vval, pval, mutval, False)
        else:
            self.xval = None
            self.yval = None
            self.rhoval = None
            self.uval = None
            self.vval = None
            self.pval = None
            self.mutval = None

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

        # Coordinates used by the PINN/GNN
        # Data coordinates
        self.X_data = torch.cat([self.x, self.y], dim=1)
        self.n_data = self.X_data.shape[0]

        # Collocation coordinates
        if self.model == "pinn" and self.Xf is not None:
            self.X_res = self.Xf
            self.n_res = self.X_res.shape[0]
        else:
            self.X_res = None
            self.n_res = 0

        # All training coordinates: data + collocation
        if self.X_res is not None:
            self.X_all = torch.cat([self.X_data, self.X_res], dim=0)
        else:
            self.X_all = self.X_data

        # Input bounds for normalization
        self.lb = self.X_all.detach().min(dim=0).values
        self.ub = self.X_all.detach().max(dim=0).values

        # GNN graph construction
        if self.net_arch == "gnn":
            # Graph data
            # supervised data nodes followed by collocation points
            self.X_graph_train = self.X_all
            self.n_data_graph = self.n_data
            
            # Bounds
            self.data_slice = slice(0, self.n_data)
            self.res_slice = slice(self.n_data, self.n_data + self.n_res)

            # Important: build graph using the same coordinates seen by the GNN.
            # Since self.forward() normalizes before calling self.network,
            # the fixed graph should also be built from normalized coordinates.
            X_graph_train_norm = self.normalize_input(self.X_graph_train.detach())

            self.edge_index_train, self.edge_attr_train = \
                self.network.build_graph(X_graph_train_norm)

            # Validatin graph
            if self.has_validation:

                # X_graph_data_val
                self.X_graph_val = torch.cat([self.xval, self.yval], dim=1)
                # Normalization
                X_graph_val_norm = self.normalize_input(self.X_graph_val.detach())

                self.edge_index_val, self.edge_attr_val = \
                    self.network.build_graph(X_graph_val_norm)

            else:
                # Validation graph
                self.X_graph_val = None
                self.edge_index_val = None
                self.edge_attr_val = None

        else:
            # Training graph
            self.X_graph_train = None
            self.data_slice = None
            self.res_slice = None
            self.edge_index_train = None
            self.edge_attr_train = None
            # Validation graph
            self.X_graph_val = None
            self.edge_index_val = None
            self.edge_attr_val = None


		# Optimizers
        # Adam
        adam_cfg = params['optimizer']['adam']
        self.use_adam = adam_cfg.get('enabled', False)
        if self.use_adam:
            self.n_adam_iter = int(adam_cfg.get('iterations', 50000))
            if self.n_adam_iter <= 0:
                raise ValueError("n_adam_iter must be greater than zero")
            learning_rate_adam = float(adam_cfg.get('learning_rate', 5e-4))
            if learning_rate_adam <= 0:
                raise ValueError("learning rate must be greater than zero")        
            scheduler_size = int(adam_cfg['scheduler'].get('step_size', 10000))
            scheduler_gamma = float(adam_cfg['scheduler'].get('gamma', 0.5))
             
            # Optimizer
            self.optimizer_adam = torch.optim.Adam(network.parameters(), 
                lr=learning_rate_adam)

            # Scheduler
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer_adam, step_size=scheduler_size, gamma=scheduler_gamma)
            
        else:
            self.n_adam_iter = 0

        # LBFGS
        lbfgs_cfg = params['optimizer']['lbfgs']
        self.use_lbfgs = lbfgs_cfg.get('enabled', False)
        if self.use_lbfgs:
            self.n_lbfgs_iter = int(lbfgs_cfg.get('iterations', 1000))
            if self.n_lbfgs_iter <= 0:
                raise ValueError("n_lbfgs_iter must be greater than zero")
            max_iter_lbfgs = int(lbfgs_cfg.get('max_iter_per_step', 20))
            if max_iter_lbfgs <= 0:
                raise ValueError("max_iter_lbfgs must be greater than zero")
            learning_rate_lbfgs = float(lbfgs_cfg.get('learning_rate', 1.0))
            if learning_rate_lbfgs <= 0:
                raise ValueError("L-BFGS learning rate must be greater than zero")

            # Optimizer
            self.optimizer_lbfgs = torch.optim.LBFGS(
                network.parameters(),
                lr=learning_rate_lbfgs,
                max_iter=max_iter_lbfgs,
                history_size=50,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-7,
                tolerance_change=1e-9)
        else:
            self.n_lbfgs_iter = 0

        # Load model
        if params['run']['checkpoint']['load_model']:
            self.load_model(params['paths']['model'], 
                            params['files']['model_name'])

        # Verbose
        self.verbose = params['run'].get("verbose", False)
        # PhysicsInformedNN setup
        if self.verbose:
            print("---------------------------------------")
            print("Physics Informed Neural Network initialized")
            print(f"  Device                         : {self.device}")
            print(f"  Learning formulation           : {self.model}")
            print(f"  Equation                       : {self.eq}")
            print(f"  Network architecture           : {self.net_arch}")
            if self.net_arch == 'mlp':
                print(f"    MLP                          : {network.layers}")
                print(f"    Activation function          : {network.activation}")
            elif self.net_arch == 'gnn':
                print(f"    Latent feature dimension     : {network.latent_dim}")  
                print(f"    Activation function          : {network.activation}")  
                print(f"    Graph neighbors per node     : {network.neighbors}")
                message_cfg = params['network']['gnn']['attributes']
                print(f"    Node boundary marker         : {message_cfg['node']['boundary_marker']}")
                print(f"    Edge distance feature        : {message_cfg['edge']['distance']}")
                processor_cfg = params['network']['gnn']['processor']
                print(f"    Message-passing layers       : {processor_cfg['message_layers']}")
                print(f"    Message aggregation          : {processor_cfg['aggregation']}")
                print(f"    Residual update              : {processor_cfg['residual']}")
            print(f"  Use Adam                       : {self.use_adam}") 
            if self.use_adam:
                print(f"    Number of Adam iterations    : {self.n_adam_iter}")
                print(f"    Adam learning rate           : {learning_rate_adam}")
                print(f"    Scheduler size               : {scheduler_size}")
                print(f"    Learning Reduction rate      : {scheduler_gamma}")
            print(f"  Use L-BFGS                     : {self.use_lbfgs}")
            if self.use_lbfgs:
                print(f"    Number of L-BFGS iterations  : {self.n_lbfgs_iter}")
                print(f"    L-BFGS learning rate         : {learning_rate_lbfgs}")
                print(f"    Max iterrations for L-BFGS   : {max_iter_lbfgs}")
            if self.net_arch == 'mlp':
                if network.dropout_p > 0.0:
                    print(f"  Dropout:")
                    print(f"    Probability                  : {network.dropout_p}")
                    print(f"    Hidden layer indices         : {network.dropout_indices}")
            print(f"  Loss weights:")
            print(f"    rho                          : {self.w_rho}")
            print(f"    u                            : {self.w_u}")
            print(f"    v                            : {self.w_v}")
            print(f"    p                            : {self.w_p}")
            if self.eq == 'RANS':
                print(f"    mut                          : {self.w_mut}")
            print(f"    f1                           : {self.w_f1}")
            print(f"    f2                           : {self.w_f2}")
            print(f"    f3                           : {self.w_f3}")
            print(f"    f4                           : {self.w_f4}")
            print(f"  Model I/O:")
            print(f"    Load                         : {params['run']['checkpoint']['load_model']}")
            print(f"    Save                         : {params['run']['checkpoint']['save_model']}")
            
    def forward(self, X, use_dropout=False, edge_index=None, edge_attr=None):
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

        edge_index : torch.Tensor or None, optional
            GNN graph connectivity

        edge_attr : torch.Tensor or None, optional
            GNN edge attributes

        Returns
        -------
        torch.Tensor
            Raw network output.
        """
        
        # Normalised input coordinates
        X_norm = self.normalize_input(X)

        # Note that, this is self.network.forward(...)
        if self.net_arch == 'mlp':
            return self.network(X_norm, use_dropout=use_dropout)

        if self.net_arch == 'gnn':
            if edge_index is None:
                raise ValueError("GNN forward evaluation requires edge_index.")

            if edge_attr is None:
                raise ValueError("GNN forward evaluation requires edge_attr.")

            return self.network(X_norm, edge_index, edge_attr, use_dropout=False)

        raise ValueError(f"Unknown network architecture: {self.net_arch}")

    def normalize_input(self, X):
        """
        Input normalisation between [-1,1]

        Parameters
        ----------
        X : torch.Tensor
            Input coordinates.
        """

        coordinate_range = self.ub - self.lb

        if torch.any(coordinate_range <= 0.0):
            raise ValueError(
                "Each input coordinate must have a nonzero range. "
                f"lb={self.lb}, ub={self.ub}")

        eps = 1e-12
        return 2.0 * (X - self.lb) / (self.ub - self.lb + eps) - 1.0

    def output_to_fields(self, out):
        """
        Convert raw network output into physical flow variables.

        Note that, this method can be used to:
        rho_data, u_data, v_data, p_data = self.output_to_fields(out_data)
        rho_f, u_f, v_f, p_f = self.output_to_fields(out_res)
        for GNN.
        """

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


    def net_fields(self, x, y, use_dropout=False, role=None):
        """
        Predict physical flow variables.

        Parameters
        ----------
        x, y : torch.Tensor
            Coordinates.

        use_dropout : bool, optional
            Whether dropout is enabled.

        role : {"data", "residual", "validation", "query"} or None
            Specifies which graph the GNN should evaluate.

            This argument is ignored by the MLP.
        """
        
        X = torch.cat([x, y], dim=1)

        # MLP
        if self.net_arch == "mlp":
            out = self.forward(X, use_dropout)
            return self.output_to_fields(out)

        # GNN
        elif self.net_arch == "gnn":

            # GNN supervised training-data evaluation
            if role == "data":
                out_graph = self.forward(self.X_graph_train, use_dropout=False,
                                         edge_index=self.edge_index_train,
                                         edge_attr=self.edge_attr_train)
                out = out_graph[self.data_slice]
                return self.output_to_fields(out)

            # GNN PDE residual evaluation
            if role == "residual":
                if self.X_res is None:
                    raise ValueError("GNN residual evaluation requires collocation " \
                                     "points.")
                
                if X.shape[0] != self.n_res:
                    raise ValueError("The number of residual coordinates passed " \
                                     "to net_fields does not match the number of " \
                                     "collocation nodes. " \
                                    f"Expected {self.n_res}, received {X.shape[0]}.")

                # X contains the collocation coordinates with
                # requires_grad=True.
                #
                # The supervised-data coordinates remain fixed, but they are
                # included in the GNN forward pass so the complete training
                # graph is used for message passing.                
                X_graph = torch.cat([self.X_data.detach(), X], dim=0)

                # Normalize without detach so the dependence on the residual
                # coordinates is preserved. Note that, this normalisation is
                # because I will generate a new edge_attr_res, which is
                # differentiable.
                X_graph_norm = self.normalize_input(X_graph)

                # Keep the same fixed graph connectivity, but reconstruct the
                # geometric edge attributes from the differentiable coordinates.
                edge_attr_res = self.network.build_edge_attr(
                    X_graph_norm,
                    self.edge_index_train,
                )

                # Note that the graph was already created with a normalised data,
                # although X_graph is going to be normalised in forward.
                out_graph = self.forward(X_graph, use_dropout=False, 
                                        edge_index=self.edge_index_train, 
                                        edge_attr=edge_attr_res)
                
                # Only collocation node predictions are used by the PDE loss.
                out = out_graph[self.res_slice]
                return self.output_to_fields(out)
        
            if role == "validation":        
                if self.X_graph_val is None:
                    raise ValueError("Validation graph has not been initialized.")
                
                out_graph = self.forward(self.X_graph_val, use_dropout=False, 
                                         edge_index=self.edge_index_val,
                                         edge_attr=self.edge_attr_val)
                
                return self.output_to_fields(out_graph)


            # GNN arbitrary query or prediction
            if role == "query":
                X_query_norm = self.normalize_input(X.detach())

                # Create a graph for prediction
                edge_index_query, edge_attr_query = \
                    self.network.build_graph(X_query_norm)

                out_graph = self.forward(X, use_dropout=False,
                                         edge_index=edge_index_query,
                                         edge_attr=edge_attr_query)

                return self.output_to_fields(out_graph)
     
            raise ValueError("GNN net_fields requires role='data', 'residual', " \
                             "'validation', or 'query'.")

        else:  
            raise ValueError(f"Unknown network architecture: {self.net_arch}")
        
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
        Predict dimensional flow variables at given physical coordinates.

        The input coordinates are non-dimensionalized using `self.Lref`, converted
        to tensors on `self.device`, and evaluated with the network in inference
        mode.

        Parameters
        ----------
        x, y : array_like
            Physical x- and y-coordinates of the query points. Accepted shapes are
            `(N,)` or `(N, 1)`.

        Returns
        -------
        tuple of numpy.ndarray
            For `eq == 'Euler'`: `(rho, u, v, p)`.
            For `eq == 'RANS'`: `(rho, u, v, p, mut)`.

            All returned quantities are dimensional and have shape `(N, 1)`.
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
            rho, u, v, p = self.net_fields(x_t, y_t, role="query")

            rhod, ud, vd, pd = self.get_dimensional_data(rho, u, v, p)
            return (
                rhod.cpu().numpy(), 
                ud.cpu().numpy(), 
                vd.cpu().numpy(), 
                pd.cpu().numpy()
            )

        elif self.eq == 'RANS':
            rho, u, v, p, muthat = self.net_fields(x_t, y_t, role="query")
            
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
                    "mut_scale has not been defined. " \
                    "Call this method first with fit_scale=True using the training" \
                    " data."
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

        if checkpoint.get("optimizer_adam_state_dict") is not None \
            and hasattr(self, "optimizer_adam"):
            self.optimizer_adam.load_state_dict(checkpoint["optimizer_adam_state_dict"])

        if (checkpoint.get("optimizer_lbfgs_state_dict") is not None 
            and hasattr(self, "optimizer_lbfgs") and self.optimizer_lbfgs is not None):
            self.optimizer_lbfgs.load_state_dict(checkpoint["optimizer_lbfgs_state_dict"])

        if (checkpoint.get("scheduler_state_dict") is not None 
            and hasattr(self, "scheduler") and self.scheduler is not None):
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
        Evaluate the trained model on the held-out test dataset.

        Computes model predictions and error metrics without updating the
        network parameters.
            
        Parameters
        ----------
        x, y : torch.Tensor
            Test-point coordinates.

        rho, u, v, p : torch.Tensor
            Reference test values for density, velocity, and pressure.

        mut : torch.Tensor, optional
            Reference eddy viscosity for RANS cases.

        Returns
        -------
        predictions : dict
            Predicted physical fields.

        metrics : dict
            Error metrics for the test dataset.
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
                rho_pred, u_pred, v_pred, p_pred = \
                    self.net_fields(x_t, y_t, use_dropout=False, role="query")

            elif self.eq == "RANS":
                rho_pred, u_pred, v_pred, p_pred, muthat_pred = \
                    self.net_fields(x_t, y_t, use_dropout=False, role="query")

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
