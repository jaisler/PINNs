# Contributing to PIRFlow

Thank you for contributing to PIRFlow. Contributions that improve the
scientific correctness, reliability, documentation, or usability of the
project are welcome.

## Before You Start

For substantial features, changes to the governing equations, or changes to
public interfaces, open an issue or discussion first. Describe the problem,
the proposed approach, and any scientific assumptions. Small bug fixes and
documentation corrections can be submitted directly.

Do not commit CFD datasets, generated samples, trained models, result files,
credentials, or machine-specific paths. The `samples/`, `model/`, and
`results/` directories are intentionally ignored by Git, since these 
repositories are reserved for the code output.

## Development Setup

PIRFlow is a Python project. Create an isolated environment and install the
pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy or edit `configs/configuration.yaml` for your local CFD flowfield, mesh,
sample, model, and result paths. Keep personal paths out of commits.

## Making Changes

- Keep each contribution focused on one problem.
- Preserve the existing module boundaries: configuration, sampling, networks,
  PINN physics and training, identification, post-processing, and utilities.
- Follow PEP 8 and use clear names for physical quantities, normalized values,
  tensor dimensions, and coordinate systems.
- Add docstrings to public classes and functions. PIRFlow generally uses
  NumPy-style sections such as `Parameters`, `Returns`, and `Raises`.
- Add type hints when they improve clarity without obscuring tensor or array
  behavior.
- Prefer `pathlib.Path` for new filesystem code.
- Raise specific exceptions with actionable messages when configuration,
  shapes, fields, or physical assumptions are invalid.
- Avoid unrelated formatting or refactoring in the same change.

## Scientific and Machine-Learning Changes

Changes to physical residuals, nondimensionalization, loss terms, sampling, or
network behavior require extra care. A contribution should document:

- the equations, closure assumptions, and sign conventions being used;
- dimensional and nondimensional units;
- expected NumPy array and PyTorch tensor shapes;
- effects on Euler and RANS configurations;
- effects on MLP and GNN architectures;
- whether automatic differentiation remains connected to the collocation
  coordinates; and
- numerical safeguards used for positivity, division, scaling, or stability.

Where applicable, cite the paper, textbook, or reference implementation that
supports the formulation. Do not change reference scales or physical defaults
without explaining the effect on existing configurations and checkpoints.

## Configuration Changes

When adding a configuration option:

1. Add it to `configs/configuration.yaml` with a useful default and comment.
2. Validate unsupported or inconsistent values near the point of use.
3. Use `.get(...)` only when a genuine backward-compatible default exists.
4. Update the README or relevant docstrings when users need to know about it.

## Validation

PIRFlow does not yet have a complete automated test suite. Before submitting a
change, perform the checks relevant to your work:

```bash
python -m compileall -q main.py src
python main.py
```

Also verify the affected combinations where practical:

- supervised and physics-informed models;
- MLP and GNN architectures;
- Euler and RANS equations;
- CPU and CUDA devices, if the change is device-sensitive;
- fresh sampling and loading previously saved samples; and
- training, checkpoint loading, evaluation, and post-processing.

For bug fixes, include a minimal regression test when possible. New tests
should be deterministic, use small synthetic inputs, and avoid requiring large
CFD files or a GPU. If full execution requires unavailable data or hardware,
state exactly what was and was not validated in the pull request.

## Commits and Pull Requests

Write concise, imperative commit messages, for example:

```text
Fix GNN residual edge gradients
Add pressure-tap observation loader
Document RANS viscosity scaling
```

A pull request should include:

- a clear description of the problem and solution;
- the motivation and scientific rationale, when relevant;
- configuration or compatibility implications;
- validation commands and results;
- plots or metrics for changes that affect numerical results; and
- links to related issues or references.

Keep generated artifacts out of the pull request unless they are small and
necessary to explain or test the change. Reviewers may ask for additional
validation when a contribution changes numerical behavior.

## Reporting Bugs

A useful bug report includes the PIRFlow revision, operating system, Python and
dependency versions, relevant configuration, a minimal reproduction command,
the complete error message, and the expected behavior. Remove private paths or
data before sharing logs.

## License

By contributing, you agree that your contribution will be licensed under the
MIT License used by PIRFlow.
