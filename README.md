# Physics-Informed Neural Networks for Supersonic Compressible Flows

This repository contains a PyTorch implementation of Physics-Informed Neural Networks (PINNs) for reconstructing compressible flow fields from sparse data. The project focuses on steady compressible Euler and RANS-type formulations, with applications to high-speed internal flows such as scramjet-like configurations.

The neural network predicts flow variables such as density, velocity components, pressure, and, for RANS cases, turbulent viscosity. The model can be trained either as a purely supervised neural network or as a PINN by combining data loss with the residuals of the governing equations.

## Features

- PyTorch-based PINN implementation
- Support for steady compressible Euler equations
- Support for RANS-style formulation with effective viscosity
- Supervised and physics-informed training modes
- Sparse data reconstruction from CFD samples
- Collocation-point residual minimization
- Input normalization and non-dimensionalization
- Positive constraints for density, pressure, and turbulent viscosity
- Adam and L-BFGS optimization
- Validation and test error metrics
- Sampling utilities for CFD data and geometry-based points
- Modular structure for losses, residuals, networks, and utilities

## Project Structure

```text
PINNs/
├── main.py
├── README.md
├── LICENSE
└── src/
    ├── pinn/
    │   ├── physics_informed_nn.py
    │   ├── losses.py
    │   └── residuals.py
    ├── networks/
    │   ├── base.py
    │   └── mlp.py
    └── utils/
        ├── metrics.py
        ├── print_loss.py
        └── plotting.py
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
