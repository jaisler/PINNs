# SPDX-License-Identifier: MIT
import os
import yaml
import numpy as np
import time
import pyvista as pv
from pathlib import Path
import torch
import torch.nn as nn

from src.sampling.sampling import SamplingData
from src.networks import MLP, GNN
from src.pinn import PhysicsInformedNN
from src.utils import print_metrics_table
import src.utils.plot as pl

def main():

    # Configuration file
    with open(r'configs/configuration.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        params = yaml.load(file, Loader=yaml.FullLoader)

    # Check if folder exists: results
    if not os.path.isdir(params['paths']['results']):
        os.makedirs(params['paths']['results'], exist_ok=True)

    # Check if folder exists: data
    if not os.path.isdir(params['paths']['data']):
        os.makedirs(params['paths']['data'], exist_ok=True)

    # Check if folder exists: model
    if not os.path.isdir(params['paths']['model']):
        os.makedirs(params['paths']['model'], exist_ok=True)

    # Load CFD mesh/flowfield once
    # It is used for:
    #   1. plotting the original simulation fields,
    #   2. evaluating the PINN prediction on the CFD mesh points.
    flowfield = None
    if params['run']['routines']['inference']:
        flowfield = pv.read(os.path.join(params['paths']['flow'], 
                                         params['files']['flowfield']))

    # Plot CFD/simulation fields using unified PyVista style
    pl.plot_simulation_flow(flowfield, params)
    
    # Initialisation
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
        # number of the points provided in the configureation file.
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
                
        # Data points
        # Train / validation / test split
        N_train_data = min(params['dataset']['n_train_data'], N)

        N_val_data = min(
            params['dataset'].get('n_validation_data', 0),
            N - N_train_data
        )

        N_test_data = min(
            params['dataset'].get('n_test_data', N - N_train_data - N_val_data),
            N - N_train_data - N_val_data
        )

        # Path for the dataset split
        idx_file = Path(params['paths']['data']) / "idx_split_data.npz"
        # Get data from file
        if idx_file.exists() and not params['run']['routines']['sampling']:
            split = np.load(idx_file)

            idx_train = split["idx_train"]
            idx_val = split["idx_val"]
            idx_test = split["idx_test"]

            if np.max(idx_train) >= N or np.max(idx_val) >= N or np.max(idx_test) >= N:
                raise ValueError(
                    "Loaded split indices are not compatible with the current data."
                )

        else:
            rng = np.random.default_rng(params.get("seed", 1234))
            # Shuffle index
            idx_all = rng.permutation(N)
            # Training data
            idx_train = idx_all[:N_train_data]
            # Validation data
            idx_val = idx_all[
                N_train_data:N_train_data + N_val_data
            ]
            # Test data
            idx_test = idx_all[
                N_train_data + N_val_data:
                N_train_data + N_val_data + N_test_data
            ]

            # Save index for loading datasets
            np.savez(idx_file, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

        # Training data
        xtrain = x[idx_train, None]
        ytrain = y[idx_train, None]
        rhotrain = rho[idx_train, None]
        utrain = u[idx_train, None]
        vtrain = v[idx_train, None]
        ptrain = p[idx_train, None]
        muttrain = mut[idx_train, None]

        # Validation data
        xval = x[idx_val, None]
        yval = y[idx_val, None]
        rhoval = rho[idx_val, None]
        uval = u[idx_val, None]
        vval = v[idx_val, None]
        pval = p[idx_val, None]
        mutval = mut[idx_val, None]

        # Test data
        xtest = x[idx_test, None]
        ytest = y[idx_test, None]
        rhotest = rho[idx_test, None]
        utest = u[idx_test, None]
        vtest = v[idx_test, None]
        ptest = p[idx_test, None]
        muttest = mut[idx_test, None]

        if Xf is not None:  
            # Collocation points
            Ncoll = Xf.shape[0]
            xf = Xf[:,0]   # N 
            yf = Xf[:,1]   # N
            # For training data
            N_train_coll = min(params['dataset']['n_train_collocation'], Ncoll)
            # File for loading collocation ponts
            idxc_file = Path(params['paths']['data']) / "idx_train_coll.npy"

            if idxc_file.exists() and not params['run']['routines']['sampling']:
                idxc = np.load(idxc_file)

                if idxc.shape[0] != N_train_coll:
                    raise ValueError(
                        "Loaded collocation indices have a different size from "
                        f"n_train_collocation. Expected {N_train_coll}, got {idxc.shape[0]}."
                    )

                if np.max(idxc) >= Ncoll:
                    raise ValueError(
                        "Loaded collocation indices are not compatible with Xf."
                    )

            else:
                rng = np.random.default_rng(params.get("seed", 1234))
                idxc = rng.choice(Ncoll, N_train_coll, replace=False)
                np.save(idxc_file, idxc)

            xftrain = xf[idxc, None]
            yftrain = yf[idxc, None]        
        
        # Plot traning ponts
        pl.plot_target_points(xtrain, ytrain, xftrain, yftrain, params, True)
        # Plot all points
        pl.plot_target_points(x, y, xf, yf, params)
        
        # Network architecture
        if params['network']['architecture'] == 'mlp':
            mlp_cfg = params['network']['mlp'] 
            network = MLP(
                layers=mlp_cfg['layers'],
                activation=mlp_cfg['activation'],
                dropout_p=mlp_cfg['dropout'].get("probability", 0.0),
                dropout_indices=mlp_cfg['dropout'].get("hidden_layer_indices", [])
            )
        #elif params['network']['architecture'] == 'gnn':
            gnn_cfg = params['network']['gnn'] 
            
            node_input_dim = edge_input_dim = params['geometry']['dimension']
            if gnn_cfg['features']['node']['boundary_marker']:
                node_input_dim += 1
            if gnn_cfg['features']['edge']['distance']:
                edge_input_dim += 1 

            if params['run']['equation'] == 'Euler':
                output_dim = 4
            elif params['run']['equation'] == 'RANS':
                output_dim = 5

            #network = GNN(
            GNN(
                node_input_dim=node_input_dim,
                edge_input_dim=edge_input_dim,
                output_dim=output_dim,
                latent_dim=gnn_cfg['latent_dim'],
                activation=gnn_cfg['activation'],
                neighbors=gnn_cfg['graph']['neighbors'],
                message_layers=gnn_cfg['processor']['message_layers'],
                aggregation=gnn_cfg['processor']['aggregation'],
                residual=gnn_cfg['processor']['residual'],
                use_boundary_marker=gnn_cfg['features']['node']['boundary_marker'],
                use_edge_distance=gnn_cfg['features']['edge']['distance'],
            )
        else:
            raise ValueError(
                f"Unknown nektwork architecture: {params['network']['architecture']}."
            )

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

        # Train
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

        # Prediction on the CFD mesh
        # This gives high-resolution prediction plots because the PINN is
        # evaluated at every point of the original CFD mesh.
        xmesh = flowfield.points[:, 0]
        ymesh = flowfield.points[:, 1]

        if params['run']['equation'] == 'Euler':
            rho_pred_mesh, u_pred_mesh, v_pred_mesh, p_pred_mesh = \
                  model.predict(xmesh, ymesh)
            pred_list = [rho_pred_mesh, p_pred_mesh, u_pred_mesh, v_pred_mesh]
        elif params['run']['equation'] == 'RANS':
            rho_pred_mesh, u_pred_mesh, v_pred_mesh, p_pred_mesh, mut_pred_mesh = \
                model.predict(xmesh, ymesh)
            pred_list = [rho_pred_mesh, p_pred_mesh, u_pred_mesh, v_pred_mesh, 
                         mut_pred_mesh]

        pl.plot_predicted_flow_pyvista(flowfield, pred_list, params)

        # Evaluate data        
        test_metrics = model.evaluate_data(xtest, ytest, rhotest, utest,
                                           vtest, ptest, muttest)
        # Print metrics of the test dataset
        print_metrics_table(test_metrics)

if __name__ == "__main__":
    main()
