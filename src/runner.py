# SPDX-License-Identifier: MIT
import numpy as np
import time

from src.config import create_output_directories, load_config
from src.sampling import get_data_points, get_collocation_points, prepare_data
from src.networks import build_network
from src.pinn import build_pinn_model, training_is_enabled
from src.utils import print_metrics_table
from src.postprocessing import run_flowfield_postprocessing
import src.utils.plot as pl

def run() -> None:
    
    # Configuration file
    params = load_config()
    
    # Create output directories
    create_output_directories(params)

    # Sample or read data
    data_pnts = get_data_points(params)
    collocation_pnts = get_collocation_points(params)
    
    # Plot original sampling points
    pl.plot_sampling_data(data_pnts, collocation_pnts, params)

    # Stop after sampling when inference is disabled
    if not params["run"]["routines"]["inference"]:
        return

    # Collocation coordinates
    if collocation_pnts is not None:
        Xf = collocation_pnts["Xf"]
    else:
        Xf = None
        
    # Prepare training, validation, test and collocation datasets
    data = prepare_data(data_pnts["X"], data_pnts["U"], data_pnts["rho"], 
                        data_pnts["p"], data_pnts["mut"], Xf, 
                        params)

    # Plot prepared datasets
    pl.plot_prepared_data(data, params)

    # Build neural network: MLP or GNN
    network = build_network(params)

    # Build PINN model
    model = build_pinn_model(network, data, params)

    # Check if training is enabled
    do_training = training_is_enabled(params)

    # Train
    if do_training:

        start_time = time.time()                
        model.fit()
        elapsed = time.time() - start_time
        print("---------------------------------------")                
        print('Training time: %.4f' % (elapsed))

        # Save model
        if params['run']['checkpoint']['save_model']:
            model.save_model(params['paths']['model'], 
                             params['files']['model_name'])

        # Get losses
        l_data = model.get_data_loss()
        l_res = model.get_residual_loss()
        l_total = model.get_total_loss()
        l_val = model.get_validation_data_loss()
        n_epoch = model.get_n_epoch()
        
        # Plot losses
        pl.plot_losses(l_data, l_res, l_total, n_epoch, params)
        # Plot validation loss
        pl.plot_validation_loss(l_data, l_val, n_epoch, params)

    else:
        print("---------------------------------------")
        print("Skipping training. Adam and LBFGS are disabled or both have "
                "zero iterations.")
 
    # Evaluate data
    # evaluate_data(mode, data)
    if (data["xtest"] is not None and data["ytest"] is not None and 
        data["xtest"].shape[0] > 0):
        test_metrics = model.evaluate_data(
            data["xtest"], data["ytest"], data["rhotest"], data["utest"], 
            data["vtest"], data["ptest"], data["muttest"]
        )

        # Print metrics of the test dataset
        print_metrics_table(test_metrics, title="Test dataset metrics")

    else:
        print("---------------------------------------")
        print("Skipping test evaluation.")
        print("No test data were created. "
              "This usually means N_test_data = 0 after the "
              "train/validation/test split.")

    # Postprocess flowfield 
    if params["run"]["routines"].get("postprocessing", False):
        run_flowfield_postprocessing(model, params)        