# PIRFlow: Physics-Informed Reconstruction of Flow Field

## Overview
This repository provides a PyTorch framework for the physics-informed reconstruction of compressible flow fields from sparse and heterogeneous observations. It combines Physics-Informed Neural Networks (PINNs) with Graph Neural Network (GNN) architectures, enabling the governing Euler or Reynolds-averaged Navier–Stokes (RANS) equations to be imposed directly on irregular computational meshes.

The primary objective is to solve an inverse problem for scramjet flows: reconstruct the complete flow field—including density, velocity, pressure, and relevant turbulence quantities—from limited measurements such as schlieren-derived density gradients, particle image velocimetry (PIV) data within the cavity, and discrete wall-pressure measurements. The framework aims to recover both the global shock-wave system and the local cavity dynamics, including the shear layer and recirculation region.

The Euler formulation provides a controlled numerical benchmark for evaluating the reconstruction methodology, while the RANS formulation addresses a more experimentally representative problem. Ultimately, the project investigates how physical constraints and complementary measurement types can enable accurate and robust flow-field reconstruction when only sparse, incomplete, or noisy data are available.

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
    ├── __init__.py
    ├── runner.py
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
    │   ├── factory.py
    │   ├── losses.py
    │   ├── physics_informed_nn.py
    │   ├── residuals.py
    │   └── training.py
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
