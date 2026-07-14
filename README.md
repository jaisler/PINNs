# Physics-Informed Graph Neural Networks for Compressible Flows

This repository contains a PyTorch implementation of physics-informed neural networks and graph neural networks for reconstructing compressible flow fields from sparse CFD data.

The project focuses on steady compressible Euler and RANS-type formulations, with applications to high-speed internal flows such as scramjet-like configurations. The model can be trained using sparse supervised data and, when enabled, physics-informed residual losses from the governing equations.

The neural network predicts flow variables such as density, velocity components, pressure, and, for RANS-type cases, turbulent viscosity.

## Features

- PyTorch-based implementation
- Support for steady compressible Euler equations and RANS-type formulations with effective viscosity
- Support for MLP and GNN architectures
- Supervised and physics-informed training modes
- Sparse data reconstruction from CFD samples
- Collocation-point residual minimization
- Input normalization and non-dimensionalization
- Positive constraints for density, pressure, and turbulent viscosity
- Adam and L-BFGS optimization
- Validation and test error metrics
- Sampling utilities for CFD data and geometry-based points
- Full-mesh prediction on CFD meshes
- VTK export of predicted flow fields and error fields
- Scaled absolute error post-processing
- PyVista-based plotting of simulation, prediction, and error fields
- Modular structure for networks, losses, residuals, sampling, metrics, plotting, and post-processing

## Project Structure

```text
PINNs/
├── LICENSE
├── main.py
├── README.md
├── configs/
│   └── configuration.yaml/
└── src/
    ├── config/
    │   ├── __init__.py
    │   └── io.py
    ├── networks/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── factory.py
    │   ├── gnn.py
    │   ├── message_passing.py
    │   └── mlp.py
    ├── pinn/
    │   ├── __init__.py
    │   ├── losses.py
    │   ├── physics_informed_nn.py
    │   └── residuals.py
    ├── postprocessing/
    │   ├── __init__.py
    │   ├── flowfield.py
    │   └── workflow.py
    ├── sampling/
    │   ├── __init__.py
    │   ├── data.py
    │   ├── sampling.py
    │   └── splitting.py
    └── utils/
        ├── __init__.py
        ├── metrics.py
        ├── ploy.py
        └── print_loss.py
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
