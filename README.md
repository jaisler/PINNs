# PIRFlow

## Overview

This repository provides a PyTorch framework for reconstructing compressible flow fields from sparse, heterogeneous observations. It combines physics-informed learning with Graph Neural Networks (GNNs), embedding governing equations, including the Euler and Reynolds-averaged Navier–Stokes (RANS) equations, directly into the learning process on irregular computational meshes.

The framework targets inverse problems in which complete flow fields—such as density, velocity, pressure, and turbulence quantities—are inferred from limited experimental or numerical data. It is designed to recover both global and local flow structures, including shock waves, boundary layers, shear layers, and recirculation regions, while remaining adaptable to a broad range of compressible-flow configurations.

## Features

- PyTorch framework supporting MLP and GNN architectures
- Steady compressible Euler and RANS formulations
- Supervised and physics-informed training with sparse CFD data and collocation points
- Non-dimensionalization, input normalization, and positivity constraints
- Adam and L-BFGS optimization with validation and test metrics
- Flexible sampling utilities for flow data and computational geometries
- Full-mesh prediction, VTK export, and PyVista visualization of predicted fields and errors
- Modular design for networks, physical residuals, losses, sampling, evaluation, and post-processing

## Documentation

See the [PIRFlow documentation](docs/README.md) for the user guide,
configuration reference, architecture overview, module reference, and current
identification-module status. Code contribution standards are documented in
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
