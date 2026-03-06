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
        Xstar = objSample.GetXstar() # N x 3
        Ustar = objSample.GetUstar() # N x 3
        rhostar = objSample.GetRHOstar() # N
        pstar = objSample.GetPstar() # N

    else:
        # Read data set
        df = pd.read_csv(os.path.join(params['pathData'], 
            params['sampling']['fdata'] + '.csv'))
        Xstar = df[['xstar', 'ystar', 'zstar']].to_numpy(dtype=float)
        Ustar = df[['ustar', 'vstar', 'wstar']].to_numpy(dtype=float)
        rhostar = df['rhostar'].to_numpy(dtype=float)
        pstar = df['pstar'].to_numpy(dtype=float)

    if(params['routine']['inference']):
        # Number of points inside the geometry. This is not the same
        # number of the points provided in the configureation file.
        N = Xstar.shape[0]

        # Rearrange Data 
        x = Xstar[:,0]   # N 
        y = Xstar[:,1]   # N
        rho = rhostar[:] # N
        u = Ustar[:,0]   # N
        v = Ustar[:,1]   # N
        p = pstar[:]     # N
        
        # Training Data - noiseless data
        N_train = min(params['N_train'], N)    
        idx = np.random.choice(N, N_train, replace=False)
        xtrain = x[idx,None]
        ytrain = y[idx,None]
        rhotrain = rho[idx,None]
        utrain = u[idx,None]
        vtrain = v[idx,None]
        ptrain = p[idx,None]

        # Dimensional values
        xtd = xtrain * params['Lref']
        ytd = ytrain * params['Lref']
        # Plot target points
        pl.PlotTargetPoints(xtd, ytd, params)

        # Training - note that model is a object of the class
        # Note that model is a object of the class
        model = pinns.PhysicsInformedNN(xtrain, ytrain, rhotrain, utrain, 
            vtrain, ptrain, params) 
        # Train
        start_time = time.time()                
        model.train(params['N_AdamIter'])
        elapsed = time.time() - start_time                
        print('Training time: %.4f' % (elapsed))
        # Prediction
        rho_pred, u_pred, v_pred, p_pred  = model.predict(x, y) 

        # Dimensional values
        xd = x * params['Lref']
        yd = y * params['Lref']
        rhod = rho_pred * params['rho']
        # Plotting - postprocessing      
        pl.PlotPredictedFlow(xd, yd, rhod, params)

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
