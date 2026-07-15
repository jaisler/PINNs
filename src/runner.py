# SPDX-License-Identifier: MIT
from src.config import create_output_directories, load_config
from src.sampling import get_data_points, get_collocation_points, prepare_data
from src.networks import build_network
from src.pinn import build_pinn_model, train_model, evaluate_data
from src.postprocessing import run_flowfield_postprocessing
from src.utils import plot_prepared_data, plot_sampling_data

def run() -> None:
    
    # Configuration file
    params = load_config()
    
    # Create output directories
    create_output_directories(params)

    # Sample or read data points
    data_pnts = get_data_points(params)

    # Sample or read collocation points
    collocation_pnts = get_collocation_points(params)
    
    # Plot original sampling points
    plot_sampling_data(data_pnts, collocation_pnts, params)

    # Stop after sampling when inference is disabled
    if not params["run"]["routines"]["inference"]:
        return

    # Collocation coordinates
    if collocation_pnts is not None:
        Xf = collocation_pnts["Xf"]
    else:
        Xf = None
        
    # Prepare training, validation, test and collocation datasets
    data = prepare_data(
        data_pnts["X"],
        data_pnts["U"], 
        data_pnts["rho"], 
        data_pnts["p"], 
        data_pnts["mut"], 
        Xf, 
        params
    )

    # Plot prepared datasets
    plot_prepared_data(data, params)

    # Build neural network
    network = build_network(params)

    # Build model
    model = build_pinn_model(network, data, params)

    # Train model
    train_model(model, params)
 
    # Evaluate test dataset
    evaluate_data(model, data)
    
    # Postprocess flowfield 
    if params["run"]["routines"].get("postprocessing", False):
        run_flowfield_postprocessing(model, params)        