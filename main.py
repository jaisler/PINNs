import os
import yaml
import sampling as smp
import pandas as pd
import numpy as np
import pinns

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
        df = []
        df.append(pd.read_csv(params['pathData'] + '/' 
            + params['datafilename'] + '.csv', delimiter=','))
        Xstar = df[['xstar', 'ystar', 'zstar']] # N x 3
        Ustar = df[['ustar', 'vstar', 'wstar']] # N x 3
        rhostar = df['rhostar']                 # N
        pstar = df['pstar']                     # N

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
    idx = np.random.choice(N, params['N_train'], replace=False)
    xtrain = x[idx, None]
    ytrain = y[idx, None]
    rhotrain = rho[idx, None]
    utrain = u[idx, None]
    vtrain = v[idx, None]
    ptrain = p[idx, None]

    # Training - note that model is a object of the class
    model = pinns.PhysicsInformedNN(xtrain, ytrain, rhotrain, utrain, vtrain, 
        ptrain, params) 

if __name__ == "__main__":
    main()
