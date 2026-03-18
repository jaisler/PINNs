import os
import yaml
import pandas as pd
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

    # Plot flow fields to be analysed
    if (params['routine']['plotflow']):
        flowfield = pv.read(os.path.join(params['pathFlow'], params['flowfield']))
        pl.PlotFlowField(flowfield, params)

    # Sampling points - Data points
    if (params['routine']['sampling']):
        # Create data set
        objSampleData = smp.SamplingData(params, False) 
        objSampleData.WriteDataToCSV(params)
        #objSampleData.PlotSamplingPointsToPDF(params)

        # Get sampling ponits and fields. Data points
        X = objSampleData.GetX() # N x 3
        U = objSampleData.GetU() # N x 3
        rho = objSampleData.GetRHO() # N
        p = objSampleData.GetP() # N
        # Note that if Euler equations are used it return an array
        # of zeros
        mut = objSampleData.GetMut()

        # Collocation points (PDE residuals)
        if params['routine']['pinn']:  
            objSampleColl = smp.SamplingData(params, True) 
            Xf = objSampleColl.GetX()

    else:
        # Read data set
        df = pd.read_csv(os.path.join(params['pathData'], 
            params['sampling']['fdata'] + '.csv'))
        X = df[['x', 'y', 'z']].to_numpy(dtype=float)
        U = df[['u', 'v', 'w']].to_numpy(dtype=float)
        rho = df['rho'].to_numpy(dtype=float)
        p = df['p'].to_numpy(dtype=float)
        mut = df['mut'].to_numpy(dtype=float) 

        # TODO: Implement writing collocation points in a file
        # TODO: Implement plotting collocation points

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

        # TODO: I need to add exception when collocation points are not provided.
        if params['routine']['pinn']:    
            # Collocation points
            Ncoll = Xf.shape[0]
            xf = Xf[:,0]   # N 
            yf = Xf[:,1]   # N
            # Training data
            N_train_coll = min(params['N_train_coll'], Ncoll)    
            idxc = np.random.choice(Ncoll, N_train_coll, replace=False)
            xftrain = xf[idxc,None]
            yftrain = yf[idxc,None]
        
        # Plot target points
        #pl.PlotTargetPoints(xtrain, ytrain, xftrain, yftrain, params)

        # Training - note that model is a object of the class
        # Note that model is a object of the class
        model = pinns.PhysicsInformedNN(xtrain, ytrain, rhotrain, utrain, 
            vtrain, ptrain, xftrain, yftrain, params, muttrain)

        # Train
        start_time = time.time()                
        model.fit(params['N_AdamIter'])
        elapsed = time.time() - start_time                
        print('Training time: %.4f' % (elapsed))
        
        # Prediction for plotting (data, collocation)
        Xall = np.concatenate([X, Xf], axis=0)
        xall = Xall[:,0]
        yall = Xall[:,1]
        rhoall_pred, uall_pred, vall_pred, pall_pred = model.predict(xall, yall) 
        # Plot inference
        pl.PlotPredictedFlow(xall, yall, rhoall_pred, params)

        # Predition for the data points (error calculation)
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

        print("Relative L2 errors:")
        print(f"  rho: {err_rho:.3e}")
        print(f"  u  : {err_u:.3e}")
        print(f"  v  : {err_v:.3e}")
        print(f"  p  : {err_p:.3e}")

if __name__ == "__main__":
    main()
