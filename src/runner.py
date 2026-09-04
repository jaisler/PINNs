# SPDX-License-Identifier: MIT
from src.config import create_output_directories, load_config
from src.networks import build_network
from src.observation import ObservationData, prepare_observation_data
from src.pinn import build_pinn_model, evaluate_data, train_model
from src.postprocessing import run_flowfield_postprocessing
from src.sampling import (
    get_collocation_points,
    get_data_points,
    prepare_data,
)
from src.utils import (
    plot_observation_data,
    plot_prepared_observation_data,
    plot_prepared_sampling_data,
    plot_sampling_data,
)


def run() -> None:
    """Execute the configured PIRFlow reconstruction workflow.

    Returns
    -------
    None
        Results are written to the configured output directories.
    """

    # Configuration file
    params = load_config()

    # Create output directories
    create_output_directories(params)

    # Sample or read collocation points if pinn model
    # Required for both forward and inverse problems.
    collocation_pnts = get_collocation_points(params)

    # Sample or read data points if forward problem
    data_pnts = get_data_points(params)

    # Plot original sampling points
    plot_sampling_data(data_pnts, collocation_pnts, params)

    # Problem definition
    problem = params["run"].get("problem", "forward").lower()

    # Load observation data
    if problem == "inverse":
        # Load and organize observation data for the inverse problem
        observation_loader = ObservationData(params)

        # Raw observation dataset
        observations = observation_loader.load_observation_data()

        # Split observation dataset
        prepared_observations = prepare_observation_data(
            observations,
            params,
        )

        # Plot all observation points
        plot_observation_data(observations, params)

        # Plot observation datasets: training, validation, test
        plot_prepared_observation_data(prepared_observations, params)

    if problem == "forward":
        # Prepare training, validation, test and collocation datasets
        datasets = prepare_data(
            data_pnts["X"],
            data_pnts["U"],
            data_pnts["rho"],
            data_pnts["p"],
            data_pnts["mut"],
            # Collocation points must be used the inverse problem as well
            collocation_pnts["Xf"],
            params,
        )

        # Plot prepared datasets from sampling
        plot_prepared_sampling_data(datasets, params)

        # Build neural network
        network = build_network(params)

        # Build model
        # The model is built with the datasets, but for the inverse problem,
        # we need to pass the observation data instead of datasets.
        model = build_pinn_model(network, datasets, params)

        # Train model
        train_model(model, params)

        # Evaluate test dataset
        evaluate_data(model, datasets)

        # Postprocess flowfield
        if params["run"]["routines"].get("postprocessing", False):
            run_flowfield_postprocessing(model, params)
