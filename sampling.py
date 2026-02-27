import numpy as np
import pyvista as pv
import plot as pl

class SamplingData:
    # Initialize the class
    def __init__(self, params):

        # Dimension
        self.dims = params['dims']
        # N points
        # --------- this should not be a member of the class -----------
        # --------- because I also have N_pool -------------
        self.N = params['sampling']['nspoin']

        # Load your solution
        # .vtk, .pvtu, .vtm, ...
        mesh = pv.read(params['pathFlow']+'/'+params['flowfield'])   

        # Choose fields you want
        #print(mesh.array_names)  

        # Sample points
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        # Get base sampler function 
        base_sampler = self.GetBaseSampler(params['sampling']['Type'])
        # Call chosen sampler 
        pts = base_sampler(xmin, xmax, ymin, ymax)  
        # The flow is 2D but VTK expects 3D points, lift to z=zmin (or 0)
        if self.dims == 2:
            pts = np.column_stack([pts, np.full((pts.shape[0],), zmin)])

        # Add extra points in regions detected by a sensor
        # Extra points based on gradient |grad(rho)|
        N_extra = params['sampling'].get('nspoin_extra', 0)
        if N_extra > 0:
            extra_pts = self.SampleBasedOnGrad(
                mesh=mesh,
                N_extra=N_extra,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
                base_sampler=base_sampler,
                var_name=params['sampling'].get('GradTypeVar', 'Density'),
                pool_factor=params['sampling'].get('pool_factor', 8),
                alpha=params['sampling'].get('alpha_grad_rho', 1.5),
                seed=params['sampling'].get('seed', 1234))
            #pts = np.vstack([pts, extra_pts])
        
        # Extra points on the boundary condition
        self.SampleBoundaryCondition

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

    def GetBaseSampler(self, sampling_type: str):
        if sampling_type == "random":
            return self.SampleRandomPoints
        elif sampling_type == "lhs":
            return self.SampleLatinHypercube
        else:
            raise ValueError("sampling Type must be 'random' or 'lhs'")

    def SampleRandomPoints(self, xmin, xmax, ymin, ymax):
        pts = np.column_stack([
            np.random.uniform(xmin, xmax, self.N),
            np.random.uniform(ymin, ymax, self.N),
            ])
        return pts 
    
    def SampleLatinHypercube(self, xmin, xmax, ymin, ymax):
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=self.dims)   # use d=2 for 2D
            u01 = sampler.random(n=self.N)              # (N,2) in [0,1)
        except ImportError:
            u01 = np.empty((self.N, self.dims))
            for j in range(self.dims):
                perm = np.random.permutation(self.N)
                # one stratum per point
                u01[:,j] = (perm + np.random.rand(self.N)) / self.N  

        # Map to physical domain bounds
        pts = np.empty_like(u01)
        pts[:,0] = xmin + (xmax - xmin) * u01[:,0]
        pts[:,1] = ymin + (ymax - ymin) * u01[:,1]
        
        return pts
    
    def SampleBoundaryCondition():

        return 0
    
    def SampleBasedOnGrad(self, mesh, N_extra,
        xmin, xmax, ymin, ymax, zmin, zmax, base_sampler,
        var_name="Density",
        pool_factor=8,
        alpha=1.5,
        eps=1e-12,
        seed=1234):

        rng = np.random.default_rng(seed)

        # Compute grad(var) on the mesh
        if var_name not in mesh.array_names:
            raise ValueError(f"'{var_name}' not found in mesh arrays: \
                {mesh.array_names}")

        mesh_g = mesh.compute_derivative(scalars=var_name, gradient=True)
        if "gradient" not in mesh_g.point_data:
            raise RuntimeError("Gradient not found after compute_derivative().")

        # Get the gradient vector 
        grad_vec = mesh_g.point_data["gradient"]
        # Calculate the norm
        mesh_g.point_data["grad_rho_mag"] = np.linalg.norm(grad_vec, axis=1)

        # Try to think about this code yourself.


        return mesh_g
    
    def GetXstar(self):       
        return self.Xstar

    def GetRHOstar(self):
        return self.rhostar

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

