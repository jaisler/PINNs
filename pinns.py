import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(1234)
np.random.seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, rho, u, v, p, params):

        device = torch.device(params.get("device", "cpu"))
        self.device = device

        X = np.concatenate([x, y], 1)

        self.lb = torch.tensor(X.min(0), dtype=torch.float32, device=device)  # (2,)
        self.ub = torch.tensor(X.max(0), dtype=torch.float32, device=device)  # (2,)

        # Spatial coordinates
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.x = self.X[:,0:1]
        self.y = self.X[:,1:2]

        # Variables
        self.u = torch.tensor(u, dtype=torch.float32, device=device)
        self.v = torch.tensor(v, dtype=torch.float32, device=device)
        self.rho = torch.tensor(rho, dtype=torch.float32, device=device)
        self.p = torch.tensor(p, dtype=torch.float32, device=device)

        # Layers
        self.layers = params["layers"]
        # Heat capacity ratio
        self.gamma = float(params["gamma"])

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(self.layers)

        # Optional LBFGS
        self.optimizer = torch.optim.LBFGS(
            self.trainable_parameters(),
            max_iter=params.get("lbfgs_maxiter", 50000),
            history_size=params.get("lbfgs_history", 50),
            line_search_fn="strong_wolfe",
            tolerance_grad=params.get("lbfgs_tol_grad", 1e-12),
            tolerance_change=params.get("lbfgs_tol_change", 1e-12),)

        # Optimizers
        self.optimizer_Adam = torch.optim.Adam(self.trainable_parameters(), 
            lr=params.get("lr", 1e-3))

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
        Returns only (rho,u,v,p) from the network.
        No derivatives.
        Used by predict(...) and net_steady_euler(...)
        """
        X = torch.cat([x, y], dim=1)
        out = self.neural_net(X, self.weights, self.biases)

        rho = out[:,0:1]
        u   = out[:,1:2]
        v   = out[:,2:3]
        p   = out[:,3:4]

        return rho, u, v, p

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

        # Get forward pass
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

    def loss_fn(self, x, y, rho_t, u_t, v_t, p_t):
        """
        Loss function for PINNs regarding the steady euler equation
        """
        rho_pred, u_pred, v_pred, p_pred, \
            f1_res, f2_res, f3_res, f4_res \
            = self.net_steady_euler(x, y)
        
        loss = (torch.mean((rho_t - rho_pred) ** 2) +
            torch.mean((u_t   - u_pred)   ** 2) +
            torch.mean((v_t   - v_pred)   ** 2) +
            torch.mean((p_t   - p_pred)   ** 2) +
            torch.mean(f1_res ** 2) +
            torch.mean(f2_res ** 2) +
            torch.mean(f3_res ** 2) +
            torch.mean(f4_res ** 2))
        
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

            loss = self.loss_fn(self.x, self.y, self.rho, self.u, self.v, self.p)
            loss.backward()
            self.optimizer_Adam.step()

            if it % print_every == 0:
                self.callback(it, loss.item())

        # LBFGS refinement (optional)
        if use_lbfgs:
            def closure():
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.x, self.y, self.rho, self.u, self.v,
                    self.p)
                loss.backward()
                return loss

            loss = self.optimizer.step(closure)
            print(f"LBFGS final loss: {float(loss):.3e}")

    @torch.no_grad()
    def predict(self, x, y):
        """
        Predict flow fields (rho, u, v, p) at given spatial locations using a
        forward pass of the trained neural network.

        This method performs *inference only* (no training, no gradient tracking,
        no PDE residual evaluation). It is typically used after (or during) training
        to query the network solution at arbitrary points.

        Mathematically, the PINN represents an approximation of the unknown fields:
            (rho, u, v, p)(x, y) ≈ (ρ̂, û, v̂, p̂)_θ(x, y)

        where θ are the learned network parameters. This routine evaluates the
        mapping (x, y) -> (ρ̂, û, v̂, p̂) by calling `self.net_fields(x_t, y_t)`.

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

        Expected Attributes (class members)
        -----------------------------------
        self.device : torch.device
            Device where the model and tensors live (e.g., CPU or CUDA GPU).
        self.net_fields : callable
            Function that takes torch tensors (x_t, y_t) with shape (N, 1)
            and returns torch tensors (rho, u, v, p) each of shape (N, 1).

        Notes
        -----
        - This function does NOT compute PDE residuals or derivatives such as
        ∂u/∂x, ∂u/∂y, etc. For residual evaluation you must use a function
        that runs with autograd enabled (i.e., without `@torch.no_grad()`).
        - Inputs are converted to float32. If your model was trained in float64,
        you may want to change dtype accordingly.
        - Outputs are moved to CPU (`.cpu()`) before converting to NumPy, so this
        works regardless of whether the model runs on CPU or GPU.
        """

        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim == 1: x = x[:, None]
        if y.ndim == 1: y = y[:, None]

        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)

        rho, u, v, p = self.net_fields(x_t, y_t)

        return (rho.cpu().numpy(),
                u.cpu().numpy(),
                v.cpu().numpy(),
                p.cpu().numpy())

    def callback(self, it, loss_value):
        print(f"It: {it}, Loss: {loss_value:.3e}")