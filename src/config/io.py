# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any

import yaml

def load_config(config_path='configs/configuration.yaml'):
    """Load a PIRFlow YAML configuration.

    Parameters
    ----------
    config_path : str
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration.
    """

    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    if params is None:
        raise ValueError(
            f"Configuration file is empty: {config_path}"
        )

    return params

def create_output_directories(params):
    """Create the configured output directories.
    
    Parameters
    ----------
    params : dict
        PIRFlow configuration.

    Returns
    -------
    None
        Missing output directories are created.
    """

    required_paths = ('results', 'samples', 'model', 'observations')

    try:
        paths = params["paths"]
    except KeyError as error:
        raise KeyError(
            "Missing required configuration section: params['paths']"
        ) from error

    for name in required_paths:
        if name not in paths:
            raise KeyError(
                f"Missing required path in configuration: params['paths']['{name}']"
            )

        Path(paths[name]).mkdir(parents=True, exist_ok=True)