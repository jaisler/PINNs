import numpy as np
import pyvista as pv
import gmsh
import re
from pathlib import Path
import plot as pl

class SamplingData:
    # Initialize the class
    def __init__(self, params, collpts=False):

        # Dimension
        self.dims = params['dims']
        self.pts_in = np.empty((0, 3), dtype=float)
        self.pts_bc = np.empty((0, 3), dtype=float)
        self.pts_grad = np.empty((0, 3), dtype=float)
        self.pts = np.empty((0, 3), dtype=float)

        if collpts:
            # Collocation points
            npinner = params['sampling']['nspoin_coll']
            npgrad = params['sampling']['nspoin_coll_grad']
            npbc = params['sampling']['nspoin_coll_bc']
        else:
            # Data points
            npinner = params['sampling']['nspoin']
            npgrad = params['sampling']['nspoin_grad']
            npbc = params['sampling']['nspoin_bc']

        # Load your solution
        # .vtk, .pvtu, .vtm, ...
        mesh = pv.read(params['pathFlow']+'/'+params['flowfield'])   

        # Choose fields you want
        #print(mesh.array_names)  

        # Sample points
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        # Get base sampler function 
        base_sampler = self.get_base_sampler(params['sampling']['type'])

        if npinner > 0:
            # Call chosen sampler 
            pts_in = base_sampler(npinner, xmin, xmax, ymin, ymax)  
            # The flow is 2D but VTK expects 3D points, lift to z=zmin (or 0)
            if self.dims == 2:
                pts_in = np.column_stack([pts_in, np.full((pts_in.shape[0],), zmin)])
        else:
            raise ValueError("Number of sample points must be provided ")
        self.pts_in = np.vstack([self.pts_in, pts_in])
        # Apply mask at the inner points
        sampled = pv.PolyData(self.pts_in).sample(mesh)
        mask = sampled["vtkValidPointMask"].astype(bool)
        self.pts_in = self.pts_in[mask]
        # All points
        self.pts = np.vstack([self.pts, pts_in])
        
        # Add extra points in regions detected by a sensor
        # Extra points based on gradient |grad(rho)|
        if npgrad > 0:
            pts_grad = self.sample_based_on_grad(
                mesh=mesh, npoin_grad=npgrad,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
                base_sampler=base_sampler,
                var_name=params['sampling'].get('grad_type_var', 'Density'),
                pool_factor=params['sampling'].get('pool_factor', 8),
                alpha=params['sampling'].get('alpha_grad_rho', 1.5),
                seed=params['sampling'].get('seed', 1234))
            self.pts_grad = np.vstack([self.pts_grad, pts_grad])
            self.pts = np.vstack([self.pts, pts_grad])

        # Points on the boundary condition
        bc_names = params['sampling']['bc']
        bc_poin = npbc #params['sampling']['nspoin_bc']
        for phys_name, n_bc in zip(bc_names, bc_poin):
            if n_bc > 0:
                pts_bc = self.sample_boundary_condition(phys_name, n_bc, params)                
                pts_bc = self.nudge_bc_points(pts_bc, phys_name, xmin, xmax, ymin, ymax)
                self.pts_bc = np.vstack([self.pts_bc, pts_bc])
                self.pts = np.vstack([self.pts, pts_bc])

        # Interpolate solution at all points
        point_cloud = pv.PolyData(self.pts)
        # Interpolates point/cell data onto pts
        sampled = point_cloud.sample(mesh)  
        # Apply the mask in all points
        mask = sampled["vtkValidPointMask"].astype(bool)

        # sampled.point_data now contains interpolated arrays at your points
        #print(sampled.point_data.keys())

        # Extract arrays
        # Note that the arrays are normalised
        self.pts = self.pts[mask]
        self.X = sampled.points[mask]        # (N,3) or (N,2)

        if collpts:
            self.U = sampled.points[mask] * 0.0  
            self.rho = sampled.points[mask] * 0.0  
            self.p = sampled.points[mask] * 0.0
            self.mut = sampled.points[mask] * 0.0
            
        else:
            self.U = sampled["Velocity"][mask]   # (N,3) or (N,2) if vector
            self.rho = sampled["Density"][mask]  # (N,) 
            self.p = sampled["Pressure"][mask]   # (N,) if scalar

            # Depending on the equations the eddy viscosity returns
            # zero or the value from the CFD
            if (params['equation'] == 'RANS'):
                mut = sampled["Eddy_Viscosity"]
                #self.mut = mut[mask] / params["mu"]
                self.mut = mut[mask]
            elif (params['equation'] == 'Euler'):
                # Otherwise return zero
                self.mut = np.zeros((self.X.shape[0], 1), dtype=float)

    def get_base_sampler(self, sampling_type: str):
        if sampling_type == "random":
            return self.sample_random_points
        elif sampling_type == "lhs":
            return self.sample_latin_hypercube
        else:
            raise ValueError("sampling type must be 'random' or 'lhs'")

    def sample_random_points(self, npoin, xmin, xmax, ymin, ymax):
        pts = np.column_stack([
            np.random.uniform(xmin, xmax, npoin),
            np.random.uniform(ymin, ymax, npoin),
            ])
        return pts 
    
    def sample_latin_hypercube(self, npoin, xmin, xmax, ymin, ymax):
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
    
    def sample_boundary_condition(self, phys_name, npoin_bc, params):
        """
        Sample boundary points from a Physical Group in a .geo file.
        phys_name: physical group name in the .geo, e.g. "inlet", "outlet", 
        "wall"
        Returns: 
        """
        rng = np.random.default_rng(params['sampling']['seed'])
        if npoin_bc <= 0:
            raise ValueError("params['sampling']['nspoin_bc'] must be > 0")

        gmsh.initialize()
        # Turn off terminal output from Gmsh
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.open(params['pathMesh']+'/'+params['mesh'])
            gmsh.model.geo.synchronize()
            #gmsh.model.mesh.generate(self.dims)
            # For 1D mesh generation before 2D
            gmsh.model.mesh.generate(self.dims-1)   # mesh curves (boundary)
            gmsh.model.mesh.generate(self.dims)   # mesh surface

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

            # if self.dims == 2 (2D domain), boundary is dim=1 (curves)
            # if self.dims == 3 (3D domain), boundary is dim=2 (surfaces)
            bc_dim = self.dims - 1
            d_use, tag_use = matches[0]  # There is only one per call
            for d, tag in matches:
                if d == bc_dim:
                    d_use, tag_use = d, tag

           # Robust node extraction: physical group -> entities -> nodes
            ent_tags = gmsh.model.getEntitiesForPhysicalGroup(d_use, tag_use)
            if len(ent_tags) == 0:
                raise RuntimeError(f"Physical group '{phys_name}' has no entities \
                                   (dim={d_use}, tag={tag_use}).")

            all_pts_bc = []
            for e in ent_tags:
                nodes = gmsh.model.mesh.getNodes(d_use, e)
                node_coords = nodes[1]  # works for both 2-return and 3-return variants
                if len(node_coords) > 0:
                    # Make the array (N,3)
                    all_pts_bc.append(np.asarray(node_coords, dtype=float).reshape(-1, 3))

            if not all_pts_bc:
                raise RuntimeError(
                    f"No nodes found for physical group '{phys_name}' on entities {ent_tags} "
                    f"(dim={d_use}, tag={tag_use})."
                )

            # Put the list of arrays in a single array, since all_pyts is the 
            # composition of many lists related to the possible entities that 
            # compose the boundary condition, e.g., many lines or surfaces.
            pts_bc = np.vstack(all_pts_bc)

            # Remove duplicates (curves share endpoints)
            # axis=0 means “consider each row [x,y,z] as one item
            pts_bc = np.unique(pts_bc, axis=0)

            # Sample with replacement if you ask for more than available.
            # It also reduce the number of points if it is too high accordanly
            # with the value provided (npoin_bc)
            idx = rng.integers(0, pts_bc.shape[0], size=npoin_bc)
            return pts_bc[idx]

        finally:
            gmsh.finalize()                

    def sample_based_on_grad(self, mesh, npoin_grad,
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
        mesh_g.point_data["grad_mag"] = np.linalg.norm(grad_vec, axis=1)

        # Candidate pool
        npoin_pool = int(pool_factor * npoin_grad)
        cand = base_sampler(npoin_pool, xmin, xmax, ymin, ymax)   

        if self.dims == 2:
            cand = np.column_stack([cand, np.full((cand.shape[0],), zmin)])

        # Keep only candidates that lie in some cell
        cell_ids = mesh_g.find_containing_cell(cand)
        cand = cand[cell_ids >= 0]
        if cand.shape[0] == 0:
            return np.empty((0, 3))

        # Evaluate grad magnitude at candidates 
        sampled = pv.PolyData(cand).sample(mesh_g)

        if "vtkValidPointMask" in sampled.point_data:
            mask = sampled["vtkValidPointMask"].astype(bool)
            pts_in = sampled.points[mask]
            grad_mag = sampled["grad_mag"][mask]
        else:
            pts_in = sampled.points
            grad_mag = sampled["grad_mag"]
        
        # Check if pts_in is zero, so no point was interpolated, all points
        # were outside of the geometry.
        if pts_in.shape[0] == 0:
            return np.empty((0, 3))

        # accept–reject with p = (g+eps)^alpha
        w = (grad_mag + eps) ** alpha
        wmax = np.max(w) 
        # Check for Nan/inf or negativa values
        if (not np.isfinite(wmax)) or wmax <= 0:
            return np.empty((0, 3))

        # Convert weights to acceptance probabilities in [0,1]
        p = w / wmax # probability
        # Note that if q < p, we keep the points because p
        # has a high probability value.
        keep = rng.random(p.shape[0]) < p
        pts_grad = pts_in[keep]

        # Ensure exactly npoin_grad points (top-up by highest weights)
        if pts_grad.shape[0] < npoin_grad:
            order = np.argsort(w)[::-1]
            need = npoin_grad - pts_grad.shape[0]
            pts_grad = np.vstack([pts_grad, pts_in[order[:need]]])
        else:
            pts_grad = pts_grad[:npoin_grad]

        return pts_grad
        
    def nudge_bc_points(self, pts_bc, name, xmin, xmax, ymin, ymax):
        pts = pts_bc.copy()

        # scale-aware eps (tiny fraction of domain size)
        epsx = 1e-7 * (xmax - xmin)
        epsy = 1e-7 * (ymax - ymin)

        n = name.lower()
        if n == "inlet":      # x = xmin
            pts[:, 0] += epsx
        elif n == "outlet":   # x = xmax
            pts[:, 0] -= epsx
        elif n == "bottom":   # y = ymin
            pts[:, 1] += epsy
        elif n == "top":      # y = ymax
            pts[:, 1] -= epsy

        return pts
    
    def get_x(self):       
        return self.X

    def get_rho(self):
        return self.rho

    def get_u(self):       
        return self.U

    def get_p(self):
        return self.p
    
    def get_mut(self):
        return self.mut

    def write_data_to_csv(self, params, collpts=False):
        if collpts:
            out = self.X 
            np.savetxt(
                params['pathData']+'/'+params['sampling']['fcoll']+'.csv', 
                out, delimiter=",", 
                header="xf,yf,zf", 
                comments="")
        else:     
            out = np.column_stack([self.X, self.rho, self.U[:,0], 
                self.U[:,1], self.U[:,2], self.p,self.mut]) 
            np.savetxt(
                params['pathData']+'/'+params['sampling']['fdata']+'.csv', 
                out, delimiter=",", 
                header="x,y,z,rho,u,v,w,p,mut", 
                comments="")

    def plot_sampling_points(self, params, collpts=False):
        pl.plot_sampling_points(self.X, self.pts_in, self.pts_bc, \
                              self.pts_grad, params, collpts)

