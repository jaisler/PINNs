# Physics-Informed Graph Neural Networks for Compressible Flows

This repository contains a PyTorch implementation of Physics-Informed Graph Neural Networks (PIGNNs) for reconstructing compressible flow fields from sparse data. The project focuses on steady compressible Euler and RANS-type formulations, with applications to high-speed internal flows such as scramjet-like configurations.

The neural network predicts flow variables such as density, velocity components, pressure, and, for RANS cases, turbulent viscosity. The model can be trained either as a purely supervised neural network or a supervised graph neural network and also be combined with a PINN by using data loss with the residuals of the governing equations.

## Features

- PyTorch-based implementation
- Support for steady compressible Euler equations
- Support for RANS-style formulation with effective viscosity
- Supervised neural netwok and graph neural network and physics-informed training modes
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
├── LICENSE
├── main.py
├── README.md
├── configs/
│   └── configuration.yaml/
└── src/
    ├── networks/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── gnn.py
    │   ├── message_passing.py
    │   └── mlp.py
    ├── pinn/
    │   ├── __init__.py
    │   ├── losses.py
    │   ├── physics_informed_nn.py
    │   └── residuals.py
    └── utils/
        ├── __init__.py
        ├── metrics.py
        ├── ploy.py
        └── print_loss.py
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
