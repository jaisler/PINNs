# SPDX-License-Identifier: MIT
import numpy as np
import time

from src.config import create_output_directories, load_config
from src.sampling import SamplingData, prepare_data
from src.pinn import PhysicsInformedNN
from src.networks import build_network
from src.utils import print_metrics_table
from src.postprocessing import run_flowfield_postprocessing
import src.utils.plot as pl

def main() -> None:
    
    # Configuration file
    params = load_config()
    # Create output directories
    create_output_directories(params)
    
    # Collocation dataset initialisation
    Xf = None
    
    # Sampling points - Data points
    if (params['run']['routines']['sampling']):
        flag = False # Data points
        # Create data set
        sample_data = SamplingData(params, flag)
        sample_data.sample() 
        # Write data to file
        sample_data.write_data_to_npz()
        
        # Get sampling ponits and fields. Data points
        X = sample_data.get_x() # N x 3
        U = sample_data.get_u() # N x 3
        rho = sample_data.get_rho() # N
        p = sample_data.get_p() # N
        # Note that if Euler equations are used it return an array
        # of zeros
        mut = sample_data.get_mut()

        # Get domain points
        pts_in_data = sample_data.get_pts_in()
        pts_bc_data = sample_data.get_pts_bc()
        pts_grad_data = sample_data.get_pts_grad()

        # Collocation points (PDE residuals)
        if params['run']['model'] == 'pinn':  
            flag = True # collocation points
            sample_coll = SamplingData(params, flag)
            sample_coll.sample() 
            # Write data to file
            sample_coll.write_data_to_npz()

            Xf = sample_coll.get_x()  

            # Get domain points
            pts_in_coll = sample_coll.get_pts_in()
            pts_bc_coll = sample_coll.get_pts_bc()
            pts_grad_coll = sample_coll.get_pts_grad()

    else:
        flag = False        
        read_data = SamplingData(params, flag)    
        X, pts_in_data, pts_bc_data, pts_grad_data, U, rho, p, mut = \
            read_data.read_data_from_npz()

        if params['run']['model'] == 'pinn':  
            flag = True
            read_coll = SamplingData(params, flag)    
            Xf, pts_in_coll, pts_bc_coll, pts_grad_coll, _, _, _, _ = \
                read_coll.read_data_from_npz()

    # Plot sampling points (data)
    pl.plot_sampling_points(X, pts_in_data, pts_bc_data, 
                            pts_grad_data, params, False)

    if params['run']['model'] == 'pinn':  
        # Plot sampling points (collocation)
        pl.plot_sampling_points(Xf, pts_in_coll, pts_bc_coll, 
                                pts_grad_coll, params, True)

    if(params['run']['routines']['inference']):

        # Prepare training, validation, test and collocation data
        data = prepare_data(X, U, rho, p, mut, Xf, params)

        # Plot prepared datasets
        pl.plot_prepared_data(data, params)

        # Build neural network: MLP or GNN
        network = build_network(params)

        # Note that model is a object of the class
        model = PhysicsInformedNN(
            network, # MLP or GNN 
            data["xtrain"], data["ytrain"], # training data
            data["rhotrain"], data["utrain"], data["vtrain"], 
            data["ptrain"], # training data
            data["xftrain"], data["yftrain"], # collocation data
            params, # general parameters
            data["muttrain"], # RANS eq.
            data["xval"], data["yval"], data["rhoval"], data["uval"], 
            data["vval"], data["pval"], data["mutval"] # validation data
        )

        # Optimizer settings
        adam_enabled = params["optimizer"]["adam"].get("enabled", False)
        lbfgs_enabled = params["optimizer"]["lbfgs"].get("enabled", False)

        adam_iterations = int(params["optimizer"]["adam"].get("iterations", 0))
        lbfgs_iterations = int(params["optimizer"]["lbfgs"].get("iterations", 0))

        do_training = (
            adam_enabled and adam_iterations > 0
        ) or (
            lbfgs_enabled and lbfgs_iterations > 0
        )

        # Train
        if do_training:

            start_time = time.time()                
            model.fit()
            elapsed = time.time() - start_time
            print("---------------------------------------")                
            print('Training time: %.4f' % (elapsed))

            # Save model
            if params['run']['checkpoint']['save_model']:
                model.save_model(params['paths']['model'], params['files']['model_name'])

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


        if params["run"]["routines"].get("postprocessing", False):
            run_flowfield_postprocessing(model, params)        

if __name__ == "__main__":
    main()
