import numpy as np
import pyvista as pv
import plot as pl

def sampling_points(params):
        
    # Load your solution
    # .vtk, .pvtu, .vtm, ...
    mesh = pv.read(params['pathFlow']+'/'+params['flowfield'])   

    # Choose fields you want
    print(mesh.array_names)  

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
            # fallback: "stratified" sampling (still better than pure random)
            d = 3
            u01 = np.empty((N, d))
            for j in range(d):
                perm = np.random.permutation(N)
                u01[:, j] = (perm + np.random.rand(N)) / N  # one stratum per point

        # Map to physical domain bounds
        pts = np.empty_like(u01)
        pts[:, 0] = xmin + (xmax - xmin) * u01[:, 0]
        pts[:, 1] = ymin + (ymax - ymin) * u01[:, 1]
        pts[:, 2] = zmin + (zmax - zmin) * u01[:, 2]

    # Interpolate solution at points
    point_cloud = pv.PolyData(pts)
    sampled = point_cloud.sample(mesh)  # interpolates point/cell data onto pts

    # sampled.point_data now contains interpolated arrays at your points
    print(sampled.point_data.keys())

    #  Extract arrays
    X = sampled.points                       # (N,3)
    rho = sampled["Density"]                 # (N,) if scalar
    U = sampled["Velocity"]                  # (N,3) if vector
    T = sampled["Temperature"]               # (N,) if scalar
    p = sampled["Pressure"]                  # (N,) if scalar

    # Get points outside the geometry
    if "vtkValidPointMask" in sampled.point_data:
        mask = sampled["vtkValidPointMask"].astype(bool)
    else:
        raise RuntimeError("vtkValidPointMask not found")

    # Remove invalid points
    X = X[mask]
    rho = rho[mask]
    U = U[mask]
    T = T[mask]
    p = p[mask]

    # Export to .csv 
    out = np.column_stack([X, rho, U[:,0], U[:,1], U[:,2], T, p])
    np.savetxt(params['pathRes']+'/'+params['sampling']['datafilename']+'.csv', 
               out, delimiter=",", header="x,y,z,rho,u,v,w,T,p", comments="")
    
    pl.plot_sampling_points(X, params)