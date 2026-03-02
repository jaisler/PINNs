import numpy as np
import pyvista as pv
import gmsh
import re
from pathlib import Path
import plot as pl


class SamplingData:
    # Initialize the class
    def __init__(self, params):

        # Dimension
        self.dims = params['dims']

        # Load your solution
        # .vtk, .pvtu, .vtm, ...
        mesh = pv.read(params['pathFlow']+'/'+params['flowfield'])   

        # Choose fields you want
        #print(mesh.array_names)  

        # Sample points
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        # Get base sampler function 
        base_sampler = self.GetBaseSampler(params['sampling']['type'])

        if params['sampling']['nspoin'] > 0:
            # Call chosen sampler 
            pts = base_sampler(params['sampling']['nspoin'], xmin, xmax, ymin, ymax)  
            # The flow is 2D but VTK expects 3D points, lift to z=zmin (or 0)
            if self.dims == 2:
                pts = np.column_stack([pts, np.full((pts.shape[0],), zmin)])
        else:
            raise ValueError("Number of sample points must be provided ")

        # Add extra points in regions detected by a sensor
        # Extra points based on gradient |grad(rho)|
        if params['sampling']['nspoin_grad'] > 0:
            pts_grad = self.SampleBasedOnGrad(
                mesh=mesh, npoin_grad=params['sampling']['nspoin_grad'],
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
                base_sampler=base_sampler,
                var_name=params['sampling'].get('grad_type_var', 'Density'),
                pool_factor=params['sampling'].get('pool_factor', 8),
                alpha=params['sampling'].get('alpha_grad_rho', 1.5),
                seed=params['sampling'].get('seed', 1234))
            pts = np.vstack([pts, pts_grad])

        # Points on the boundary condition
        for phys_name in params['sampling']['bc']:
            pts_bc = self.SampleBoundaryCondition(phys_name, params)
            #pts = np.vstack([pts, pts_bc])

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
            raise ValueError("sampling type must be 'random' or 'lhs'")

    def SampleRandomPoints(self, npoin, xmin, xmax, ymin, ymax):
        pts = np.column_stack([
            np.random.uniform(xmin, xmax, npoin),
            np.random.uniform(ymin, ymax, npoin),
            ])
        return pts 
    
    def SampleLatinHypercube(self, npoin, xmin, xmax, ymin, ymax):
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=self.dims)   # use d=2 for 2D
            u01 = sampler.random(n=npoin)              # (N,2) in [0,1)
        except ImportError:
            u01 = np.empty((npoin, self.dims))
            for j in range(self.dims):
                perm = np.random.permutation(npoin)
                # one stratum per point
                u01[:,j] = (perm + np.random.rand(npoin)) / npoin  

        # Map to physical domain bounds
        pts = np.empty_like(u01)
        pts[:,0] = xmin + (xmax - xmin) * u01[:,0]
        pts[:,1] = ymin + (ymax - ymin) * u01[:,1]
        
        return pts
    
    def SampleBoundaryCondition(self, phys_name, params):
        """
        Sample boundary points from a Physical Group in a .geo file.
        phys_name: physical group name in the .geo, e.g. "inlet", "outlet", 
        "wall"
        Returns: 
        """
        rng = np.random.default_rng(params['sampling']['seed'])

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.open(params['pathMesh']+'/'+params['mesh'])
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(self.dims)

            # Find physical group tag by name
            # Gmsh stores physical groups by (dim, tag)
            phys_groups = gmsh.model.getPhysicalGroups()
            matches = []

            for d, tag in phys_groups:
                name = gmsh.model.getPhysicalName(d, tag)
                if name == phys_name:
                    matches.append((d, tag))
        
            if not matches:
                raise ValueError(f"Physical group '{phys_name}' not found.")

        finally:
            gmsh.finalize()                

    def SampleBasedOnGrad(self, mesh, npoin_grad,
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

        # Candidate pool
        npoin_pool = int(pool_factor * npoin_grad)
        cand = base_sampler(npoin_pool, xmin, xmax, ymin, ymax)   

        if self.dims == 2:
            cand = np.column_stack([cand, np.full((cand.shape[0],), zmin)])

        # keep only candidates that lie in some cell
        cell_ids = mesh_g.find_containing_cell(cand)
        cand = cand[cell_ids >= 0]
        if cand.shape[0] == 0:
            return np.empty((0, 3))

        # Evaluate grad magnitude at candidates 
        sampled = pv.PolyData(cand).sample(mesh_g)

        if "vtkValidPointMask" in sampled.point_data:
            mask = sampled["vtkValidPointMask"].astype(bool)
            pts_in = sampled.points[mask]
            grad = sampled["grad_rho_mag"][mask]
        else:
            pts_in = sampled.points
            grad = sampled["grad_rho_mag"]

        if pts_in.shape[0] == 0:
            return np.empty((0, 3))

        # accept–reject with p ∝ (g+eps)^alpha (if density)
        w = (grad + eps) ** alpha
        wmax = np.max(w)
        if (not np.isfinite(wmax)) or wmax <= 0:
            return np.empty((0, 3))

        p = w / wmax
        keep = rng.random(p.shape[0]) < p
        pts_grad = pts_in[keep]

        # Ensure exactly N_extra points (top-up by highest weights)
        if pts_grad.shape[0] < npoin_grad:
            order = np.argsort(w)[::-1]
            need = npoin_grad - pts_grad.shape[0]
            pts_grad = np.vstack([pts_grad, pts_in[order[:need]]])
        else:
            pts_grad = pts_grad[:npoin_grad]

        return pts_grad
    
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
            params['pathData']+'/'+params['sampling']['data_filename']+'.csv', 
            out, delimiter=",", 
            header="xstar,ystar,zstar,rhostar,ustar,vstar,wstar,pstar", 
            comments="")

    def PlotSamplingPointsToPDF(self, params):
        pl.PlotSamplingPoints(self.Xstar, params)

