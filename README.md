# PIRFlow

## Overview

This repository provides a PyTorch framework for reconstructing compressible flow fields from sparse and heterogeneous observations. It combines physics-informed learning with Graph Neural Network (GNN) architectures, incorporating governing equations such as the Euler and Reynolds-averaged Navier–Stokes (RANS) equations directly into the learning process on irregular computational meshes.

The framework addresses inverse problems in which complete flow fields—including density, velocity, pressure, and turbulence quantities—are inferred from limited measurements or numerical data. Its objective is to recover complex global and local flow features, such as shock waves, boundary layers, shear layers, and recirculation regions, while remaining applicable to a broad range of compressible-flow configurations.

## Features

- PyTorch framework supporting MLP and GNN architectures
- Steady compressible Euler and RANS formulations
- Supervised and physics-informed training with sparse CFD data and collocation points
- Non-dimensionalization, input normalization, and positivity constraints
- Adam and L-BFGS optimization with validation and test metrics
- Flexible sampling utilities for flow data and computational geometries
- Full-mesh prediction, VTK export, and PyVista visualization of predicted fields and errors
- Modular design for networks, physical residuals, losses, sampling, evaluation, and post-processing

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
