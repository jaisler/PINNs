# SPDX-License-Identifier: MIT
import numpy as np
import time
from pathlib import Path

from src.config import create_output_directories, load_config
from src.sampling import SamplingData, get_data_split_indeces
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
    
    # Collocation points initialisation
    Xf = None
    xf = None
    yf = None
    xftrain = None
    yftrain = None

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
        # Number of points inside the geometry. This is not the same
        # number of the points provided in the configuration file.
        # Data points
        N = X.shape[0]

        # Rearrange Data 
        x = X[:,0]   # N 
        y = X[:,1]   # N
        rho = rho[:] # N
        u = U[:,0]   # N
        v = U[:,1]   # N
        p = p[:]     # N
        mut = mut[:] # N: eddy viscosity

        # Get data split indeces 
        (idx_train, idx_val, idx_test, N_train_data, N_val_data, 
            N_test_data) = get_data_split_indeces(N, params)

        # Training data
        xtrain = x[idx_train, None]
        ytrain = y[idx_train, None]
        rhotrain = rho[idx_train, None]
        utrain = u[idx_train, None]
        vtrain = v[idx_train, None]
        ptrain = p[idx_train, None]
        muttrain = mut[idx_train, None]

        # Validation data
        xval = None
        yval = None
        rhoval = None
        uval = None
        vval = None
        pval = None
        mutval = None

        if N_val_data > 0:
            xval = x[idx_val, None]
            yval = y[idx_val, None]
            rhoval = rho[idx_val, None]
            uval = u[idx_val, None]
            vval = v[idx_val, None]
            pval = p[idx_val, None]
            mutval = mut[idx_val, None]

        # Test data
        xtest = None
        ytest = None
        rhotest = None
        utest = None
        vtest = None
        ptest = None
        muttest = None

        if N_test_data > 0:
            xtest = x[idx_test, None]
            ytest = y[idx_test, None]
            rhotest = rho[idx_test, None]
            utest = u[idx_test, None]
            vtest = v[idx_test, None]
            ptest = p[idx_test, None]
            muttest = mut[idx_test, None]

        Ncoll = 0
        if Xf is not None:  
            # Collocation points
            Ncoll = Xf.shape[0]
            xf = Xf[:,0]   # N 
            yf = Xf[:,1]   # N
            # For training data
            # File for loading collocation ponts
            idxc_file = Path(params['paths']['samples']) / "idx_train_coll.npy"

            if idxc_file.exists() and not params['run']['routines']['sampling']:
                idxc = np.load(idxc_file)

                if idxc.shape[0] != Ncoll:
                    raise ValueError(
                        "Loaded collocation indices have a different size from Xf. "
                        f"Expected {Ncoll}, got {idxc.shape[0]}."
                    )

                if idxc.size > 0 and np.max(idxc) >= Ncoll:
                    raise ValueError(
                        "Loaded collocation indices are not compatible with Xf."
                    )

                if idxc.size > 0 and np.any(idxc < 0):
                    raise ValueError(
                        "Loaded collocation indices contain negative values."
                    )

            else:
                rng = np.random.default_rng(params.get("seed", 1234))
                idxc = rng.choice(Ncoll, Ncoll, replace=False)
                np.save(idxc_file, idxc)

            xftrain = xf[idxc, None]
            yftrain = yf[idxc, None]

        # Print dataset information
        print("---------------------------------------")
        print("Dataset information")
        print(f"  Training data points           : {N_train_data}")
        print(f"  Validation data points         : {N_val_data}")
        print(f"  Test data points               : {N_test_data}")
        if xftrain is not None:
            print(f"  Training collocation points    : {Ncoll}")
            
        # Plot training dataset points
        pl.plot_dataset(xtrain, ytrain, params, dataset='training')
        # Plot validation dataset points
        if xval is not None:
            pl.plot_dataset(xval, yval, params, dataset='validation')
        # Plot test dataset points
        if xtest is not None:
            pl.plot_dataset(xtest, ytest, params, dataset='test')
        # Plot all traning points
        pl.plot_target_points(xtrain, ytrain, xftrain, yftrain, params, True)
        # Plot all points
        pl.plot_target_points(x, y, xf, yf, params)

        # Build neural network: MLP or GNN
        network = build_network(params)

        # Note that model is a object of the class
        model = PhysicsInformedNN(
            network, # MLP or GNN 
            xtrain, ytrain, # training data
            rhotrain, utrain, vtrain, ptrain, # training data
            xftrain, yftrain, # collocation data
            params, # general parameters
            muttrain, # RANS eq.
            xval, yval, rhoval, uval, vval, pval, mutval # validation data
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
        if xtest is not None and ytest is not None and xtest.shape[0] > 0:
            test_metrics = model.evaluate_data(
                xtest, ytest, rhotest, utest, vtest, ptest, muttest
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
