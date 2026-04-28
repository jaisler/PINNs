import os
import yaml
import numpy as np
import time
import pyvista as pv
from scipy.interpolate import griddata

import pinns
import sampling as smp
import plot as pl

def main():

    # Configuration file
    with open(r'configuration.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        params = yaml.load(file, Loader=yaml.FullLoader)

    # Check if folder exists: results
    if not os.path.isdir(params['pathRes']):
        os.makedirs(params['pathRes'], exist_ok=True)

    # Check if folder exists: data
    if not os.path.isdir(params['pathData']):
        os.makedirs(params['pathData'], exist_ok=True)

    # Check if folder exists: model
    if not os.path.isdir(params['pathModel']):
        os.makedirs(params['pathModel'], exist_ok=True)

    # Plot flow fields to be analysed
    if (params['routine']['plotflow']):
        flowfield = pv.read(os.path.join(params['pathFlow'], params['flowfield']))
        pl.plot_flow_field(flowfield, params)

    # Initialisation
    Xf = None
    xf = None
    yf = None
    xftrain = None
    yftrain = None

    # Sampling points - Data points
    if (params['routine']['sampling']):
        flag = False # Data points
        # Create data set
        objSampleData = smp.SamplingData(params)
        objSampleData.sample(flag) 
        # Write data to file
        objSampleData.write_data_to_npz()
        
        # Get sampling ponits and fields. Data points
        X = objSampleData.get_x() # N x 3
        U = objSampleData.get_u() # N x 3
        rho = objSampleData.get_rho() # N
        p = objSampleData.get_p() # N
        # Note that if Euler equations are used it return an array
        # of zeros
        mut = objSampleData.get_mut()

        # Get domain points
        pts_in_data = objSampleData.get_pts_in()
        pts_bc_data = objSampleData.get_pts_bc()
        pts_grad_data = objSampleData.get_pts_grad()

        # Plot sampling points (data)
        pl.plot_sampling_points(X, pts_in_data, pts_bc_data, 
                                pts_grad_data, params, flag)

        # Collocation points (PDE residuals)
        if params['model'] == 'pinn':  
            flag = True # collocation points
            objSampleColl = smp.SamplingData(params)
            objSampleColl.sample(flag) 
            # Write data to file
            objSampleColl.write_data_to_npz()

            Xf = objSampleColl.get_x()  

            # Get domain points
            pts_in_coll = objSampleColl.get_pts_in()
            pts_bc_coll = objSampleColl.get_pts_bc()
            pts_grad_coll = objSampleColl.get_pts_grad()

            # Plot sampling points (collocation)
            pl.plot_sampling_points(Xf, pts_in_coll, pts_bc_coll, 
                                    pts_grad_coll, params, flag)
    else:
        flag = False 


    if(params['routine']['inference']):
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
        
        # Training Data - noiseless data
        N_train_data = min(params['N_train_data'], N)    
        idx = np.random.choice(N, N_train_data, replace=False)
        xtrain = x[idx,None]
        ytrain = y[idx,None]
        rhotrain = rho[idx,None]
        utrain = u[idx,None]
        vtrain = v[idx,None]
        ptrain = p[idx,None]
        muttrain = mut[idx,None]

        if Xf is not None:  
            # Collocation points
            Ncoll = Xf.shape[0]
            xf = Xf[:,0]   # N 
            yf = Xf[:,1]   # N
            # Training data
            N_train_coll = min(params['N_train_coll'], Ncoll)    
            idxc = np.random.choice(Ncoll, N_train_coll, replace=False)
            xftrain = xf[idxc,None]
            yftrain = yf[idxc,None]
        
        # Plot traning ponts
        pl.plot_target_points(xtrain, ytrain, xftrain, yftrain, params, True)
        # Plot all points
        pl.plot_target_points(x, y, xf, yf, params)
        
        # Training - note that model is a object of the class
        # Note that model is a object of the class
        model = pinns.PhysicsInformedNN(xtrain, ytrain, rhotrain, utrain, 
            vtrain, ptrain, xftrain, yftrain, params, muttrain)

        # Train
        start_time = time.time()                
        model.fit()
        elapsed = time.time() - start_time
        print("---------------------------------------")                
        print('Training time: %.4f' % (elapsed))

        # Save model
        if params['save_model']:
            model.save_model(params['pathModel'], params['model_name'])

        # Get losses
        ldata = model.get_data_loss()
        lres = model.get_residual_loss()
        ltotal = model.get_total_loss()
        nepoch = model.get_n_epoch()
        # Plot losses
        pl.plot_losses(ldata, lres, ltotal, nepoch, params)

        # Prediction for plotting (data, collocation)
        if Xf is not None:
            Xall = np.concatenate([X, Xf], axis=0)
        else:
            Xall = X.copy()
        xall = Xall[:,0]
        yall = Xall[:,1]
        rhoall_pred, uall_pred, vall_pred, pall_pred = model.predict(xall, yall) 

        # Plot Prediction
        pred_list = [rhoall_pred, pall_pred, uall_pred, vall_pred]
        for ifield, pred in enumerate(pred_list):
            pl.plot_predicted_flow(xall, yall, pred, ifield, params)

        # Prediction for the data points (error calculation)
        rho_pred, u_pred, v_pred, p_pred = model.predict(x, y) 

        # compute relative L2 errors if you have ground truth at these points
        def rel_l2(pred, true):
            pred = np.asarray(pred).reshape(-1)
            true = np.asarray(true).reshape(-1)
            return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)

        err_rho = rel_l2(rho_pred, rho)
        err_u = rel_l2(u_pred, u)
        err_v = rel_l2(v_pred, v)
        err_p = rel_l2(p_pred, p)

        print("---------------------------------------")
        print("Relative L2 errors:")
        print(f"  rho: {err_rho:.3e}")
        print(f"  u  : {err_u:.3e}")
        print(f"  v  : {err_v:.3e}")
        print(f"  p  : {err_p:.3e}")

if __name__ == "__main__":
    main()
