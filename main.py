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
    flowfield = pv.read(os.path.join(params['pathFlow'], params['flowfield']))
    pl.PlotFlowField(flowfield, params)

    # Sampling points
    if (params['routine']['sampling']):
        # Create data set
        objSample = smp.SamplingData(params) 
        objSample.WriteDataToCSV(params)
        objSample.PlotSamplingPointsToPDF(params)

        # Get sampling ponits and fields
        X = objSample.GetX() # N x 3
        U = objSample.GetU() # N x 3
        rho = objSample.GetRHO() # N
        p = objSample.GetP() # N
        # Note that if Euler equations are used it return an array
        # of zeros
        mut = objSample.GetMut()

    else:
        # Read data set
        df = pd.read_csv(os.path.join(params['pathData'], 
            params['sampling']['fdata'] + '.csv'))
        X = df[['x', 'y', 'z']].to_numpy(dtype=float)
        U = df[['u', 'v', 'w']].to_numpy(dtype=float)
        rho = df['rho'].to_numpy(dtype=float)
        p = df['p'].to_numpy(dtype=float)
        mut = df['mut'].to_numpy(dtype=float) 

    if(params['routine']['inference']):
        # Number of points inside the geometry. This is not the same
        # number of the points provided in the configureation file.
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
        N_train = min(params['N_train'], N)    
        idx = np.random.choice(N, N_train, replace=False)
        xtrain = x[idx, None]
        ytrain = y[idx, None]
        rhotrain = rho[idx, None]
        utrain = u[idx, None]
        vtrain = v[idx, None]
        ptrain = p[idx, None]
        # Additional parameter from CFD
        mut = mut[idx, None]

        # Plot target points
        pl.PlotTargetPoints(xtrain, ytrain, params)

        # Training - note that model is a object of the class
        # Note that model is a object of the class
        model = pinns.PhysicsInformedNN(xtrain, ytrain, rhotrain, utrain, 
            vtrain, ptrain, params, mut) 
        # Train
        start_time = time.time()                
        model.train(params['N_AdamIter'])
        elapsed = time.time() - start_time                
        print('Training time: %.4f' % (elapsed))
        
        # Prediction
        rho_pred, u_pred, v_pred, p_pred  = model.predict(x, y) 
        
        # Plot inference
        pl.PlotPredictedFlow(x, y, rho_pred, params)

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
