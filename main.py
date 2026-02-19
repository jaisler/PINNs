import os
import yaml
import sampling as smp
import pandas as pd
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
        pstar = objSample.GetPstar() # N

    else:
        # Read data set
        df = []
        df.append(pd.read_csv(params['pathData'] + '/' 
            + params['datafilename'] + '.csv', delimiter=','))
        Xstar = df[['xstar', 'ystar', 'zstar']] # N x 3
        Ustar = df[['ustar', 'vstar', 'wstar']] # N x 3
        pstar = df['pstar']                     # N

    # Number of points inside the geometry. This is not the same
    # number of the points provided in the configureation file.
    N = Xstar.shape[0]

    if (params['routine']['training']):
        print('passou')

    # PINNs object
    #objPINNs = pinns.PhysicsInformedNN(params) 

if __name__ == "__main__":
    main()
