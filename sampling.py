import numpy as np
import pyvista as pv
import plot as pl

class SamplingData:
    # Initialize the class
    def __init__(self, params):

        self.X = [] 
        self.U = []
        self.p = []

        # Load your solution
        # .vtk, .pvtu, .vtm, ...
        mesh = pv.read(params['pathFlow']+'/'+params['flowfield'])   

        # Choose fields you want
        #print(mesh.array_names)  

        # Sample points
        N = params['sampling']['nspoin']
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

        if (params['sampling']['Type'] == 'random'):
            pts = np.column_stack([
                np.random.uniform(xmin, xmax, N),
                np.random.uniform(ymin, ymax, N),
                np.random.uniform(zmin, zmax, N),
            ])
        
        elif (params['sampling']['Type'] == 'lhs'):
            try:
                from scipy.stats import qmc
                sampler = qmc.LatinHypercube(d=3)   # use d=2 for 2D
                u01 = sampler.random(n=N)           # (N,3) in [0,1)
            except ImportError:
                # "stratified" sampling (still better than pure random)
                d = 3
                u01 = np.empty((N, d))
                for j in range(d):
                    perm = np.random.permutation(N)
                    # one stratum per point
                    u01[:, j] = (perm + np.random.rand(N)) / N  

            # Map to physical domain bounds
            pts = np.empty_like(u01)
            pts[:, 0] = xmin + (xmax - xmin) * u01[:, 0]
            pts[:, 1] = ymin + (ymax - ymin) * u01[:, 1]
            pts[:, 2] = zmin + (zmax - zmin) * u01[:, 2]

        # Interpolate solution at points
        point_cloud = pv.PolyData(pts)
        # Interpolates point/cell data onto pts
        sampled = point_cloud.sample(mesh)  

        # sampled.point_data now contains interpolated arrays at your points
        #print(sampled.point_data.keys())

        # Extract arrays
        # Note that the arrays are normalised
        X = sampled.points        # (N,3) or (N,2)
        U = sampled["Velocity"]   # (N,3) or (N,2) if vector
        rho = sampled["Density"]  # (N,) 
        p = sampled["Pressure"]   # (N,) if scalar

        # Get points inside the geometry
        if "vtkValidPointMask" in sampled.point_data:
            mask = sampled["vtkValidPointMask"].astype(bool)
        else:
            raise RuntimeError("vtkValidPointMask not found")

        # Remove invalid points and normalised it
        self.Xstar = X[mask] / params['Lref']
        self.rhostar = rho[mask] / params['rho']
        self.Ustar = U[mask] / params['U_0']
        self.pstar = p[mask] / (params['rho'] 
            * params['U_0'] * params['U_0']) 

    def GetXstar(self):       
        return self.Xstar

    def GetRHOstar(self):
        return self.pstar

    def GetUstar(self):       
        return self.Ustar

    def GetPstar(self):
        return self.pstar

    def WriteDataToCSV(self, params):
        out = np.column_stack([self.Xstar, self.rhostar, self.Ustar[:,0], 
            self.Ustar[:,1], self.Ustar[:,2], self.pstar])
        np.savetxt(
            params['pathData']+'/'+params['sampling']['datafilename']+'.csv', 
            out, delimiter=",", 
            header="xstar,ystar,zstar,rhostar,ustar,vstar,wstar,pstar", 
            comments="")
        
    def PlotSamplingPointsToPDF(self, params):
        pl.plot_sampling_points(self.Xstar, params)

