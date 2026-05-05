import os
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.tri as mtri
from matplotlib import rc, cm
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
rc('text', usetex=True)

def plot_sampling_points(pall, pin, pbc, pgrad, params, collpts=False):

    fig, ax = plt.subplots(1, 1, num=1, figsize=(12, 4), sharey=True)
    plt.rc('legend', **{'fontsize': 14})

    p0, = ax.plot(pin[:,0], pin[:,1], 'o', color='r', markersize=2)
    p1, = ax.plot(pbc[:,0], pbc[:,1], 'o', color='b', markersize=2)
    p2, = ax.plot(pgrad[:,0], pgrad[:,1], 'o', color='m', markersize=2)
    #p3, = ax.plot(pall[:,0], pall[:,1], 'o', color='k', markersize=2)

    ax.tick_params(direction="in", which='both')
    ax.grid(color='0.5', linestyle=':', linewidth=0.5, which='both')
    #ax.set_xlim(0.0, 0.0)
    #ax.set_ylim(0.0, 0.0)
    ax.tick_params(labelsize=18)
    ax.set_xlabel(r'$x$ $[m]$', fontsize=18)
    ax.set_ylabel(r'$y$ $[m]$', fontsize=18)

    ax.set_title("Collocation points", fontsize=18) if collpts else \
        ax.set_title("Data points", fontsize=18)

    ax.legend([p0,p1,p2], [r'Inner',
                           r'Boundary',
                           r'$\left|\nabla \rho\right|^{\alpha}$'], loc='best')
    
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.97)
    if collpts:
        fig.savefig(params['pathRes']+'/'+params['sampling']['plcoll']+'.pdf')
    else:
        fig.savefig(params['pathRes']+'/'+params['sampling']['pldata']+'.pdf')

    plt.close(fig)

def plot_target_points(x, y, xf, yf, params, training=False):

    fig, ax = plt.subplots(1, 1, num=1, figsize=(12, 4), sharey=True)
    plt.rc('legend', **{'fontsize': 14})

    if xf is not None or yf is not None:
        p0, = ax.plot(xf, yf, 'o', color='darkorange', markersize=2)
    p1, = ax.plot(x, y, 'o', color='g', markersize=2)
    
    ax.tick_params(direction="in", which='both')
    ax.grid(color='0.5', linestyle=':', linewidth=0.5, which='both')
    #ax.set_xlim(0.0, 0.0)
    #ax.set_ylim(0.0, 0.0)
    ax.tick_params(labelsize=18)
    ax.set_xlabel(r'$x$ $[m]$', fontsize=18)
    ax.set_ylabel(r'$y$ $[m]$', fontsize=18)

    if xf is not None or yf is not None:
        ax.legend([p0,p1], [r'Residual',r'Data'], loc='best')
    else:
        ax.legend([p1], [r'Data'], loc='best')

    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.97)
    if training:
        ax.set_title("Data and Collocation training points", fontsize=18) 
        fig.savefig(params['pathRes']+'/'+params['sampling']['pltrain']+'.pdf')
    else:
        ax.set_title("Data and Collocation points", fontsize=18) 
        fig.savefig(params['pathRes']+'/'+params['sampling']['plall']+'.pdf')
    plt.close(fig)

def plot_losses(l_data, l_res, l_total, n_epoch, params):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.rc('legend', **{'fontsize': 14})

    # Epochs calculation
    epochs = np.arange(0, n_epoch, 1)

    p0, = ax.semilogy(epochs, l_data, '-', color='b', linewidth=2)
    if params['model'] == 'pinn': 
        p1, = ax.semilogy(epochs, l_res, '-', color='r', linewidth=2)
        p2, = ax.semilogy(epochs, l_total, '-', color='k', linewidth=2)
        ax.legend([p0,p1,p2], [r'Data loss',r'Residual loss',r'Total loss'], loc='best')
    else:
        ax.legend([p0], [r'Data loss'], loc='best')

    ax.tick_params(direction="in", which='both')
    fig.subplots_adjust(left=0.127, right=0.97, bottom=0.117, top=0.97)
    ax.grid(color='0.5', linestyle=':', linewidth=0.5, which='both')
    #ax.set_xlim(0.0, 0.0)
    ax.set_ylim(0.0001, 2.0)
    ax.tick_params(labelsize=18)
    ax.set_xlabel(r'$Epochs$', fontsize=18)
    ax.set_ylabel(r'$Losses$', fontsize=18)
    fig.savefig(params['pathRes']+'/training_losses.pdf')
    plt.show()
    plt.close()

def plot_validation_loss(l_train, l_val, n_epoch, params):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.rc('legend', **{'fontsize': 14})

    # Epochs calculation
    epochs = np.arange(0, n_epoch, 1)

    p0, = ax.semilogy(epochs, l_train, '-', color='b', linewidth=2)
    p1, = ax.semilogy(epochs, l_val, '-', color='r', linewidth=2)
    ax.legend([p0,p1], [r'Training data loss',r'Validation data loss'], loc='best')

    ax.tick_params(direction="in", which='both')
    fig.subplots_adjust(left=0.127, right=0.97, bottom=0.117, top=0.97)
    ax.grid(color='0.5', linestyle=':', linewidth=0.5, which='both')
    #ax.set_xlim(0.0, 0.0)
    ax.set_ylim(0.0001, 2.0)
    ax.tick_params(labelsize=18)
    ax.set_xlabel(r'$Epochs$', fontsize=18)
    ax.set_ylabel(r'$Losses$', fontsize=18)
    fig.savefig(params['pathRes']+'/validation_loss.pdf')
    plt.show()
    plt.close()

def plot_field(x, y, values, ifield, params, suffix="", perc=95, mult=1.8):
    """
    Generic plotting routine for both simulation and predicted data.

    Parameters
    ----------
    x, y : array_like
        Coordinates.
    values : array_like
        Scalar values to plot.
    ifield : int
        Index of the field in params['plotflow'].
    params : dict
        Plot settings.
    suffix : str
        Extra text for file name, e.g. 'sim' or 'pred'.
    perc, mult : float
        Parameters used to mask large triangles.
    """

    pf = params["plotflow"]

    field  = pf["fields"][ifield]
    comp   = pf["comp"][ifield]
    cmap   = pf["cmap"][ifield]
    clabel = pf["latex"][ifield]

    scale = np.asarray(pf["scale"][ifield], dtype=float)
    vmin, vmax = float(scale[0]), float(scale[1])

    nlevels = pf.get("nlevels", 501)
    levels = np.linspace(vmin, vmax, nlevels)

    if "ticks" in pf:
        ticks = np.asarray(pf["ticks"][ifield], dtype=float)
    else:
        ticks = scale

    # Triangulation
    tri = mtri.Triangulation(x, y)
    tris = tri.triangles

    xtri = x[tris]
    ytri = y[tris]

    e0 = np.hypot(xtri[:, 1] - xtri[:, 0], ytri[:, 1] - ytri[:, 0])
    e1 = np.hypot(xtri[:, 2] - xtri[:, 1], ytri[:, 2] - ytri[:, 1])
    e2 = np.hypot(xtri[:, 0] - xtri[:, 2], ytri[:, 0] - ytri[:, 2])
    emax = np.maximum.reduce([e0, e1, e2])

    thr = np.percentile(emax, perc) * mult
    tri.set_mask(emax > thr)

    fig, ax = plt.subplots(figsize=(12, 4))

    values = np.asarray(values).ravel()

    # Optional: center colormap at zero for signed variables
    if vmin < 0.0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = None

    cs = ax.tricontourf(
        tri,
        values,
        levels=levels,
        cmap=cmap,
        vmin=vmin if norm is None else None,
        vmax=vmax if norm is None else None,
        norm=norm,
        extend="both"
    )

    cbar = fig.colorbar(
        cs,
        ax=ax,
        shrink=0.42,
        fraction=0.1,
        pad=0.02,
        aspect=10,
        ticks=ticks
    )
    cbar.set_label(clabel, fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    ax.tick_params(labelsize=18)
    ax.set_xlabel(r"$x$ $[m]$", fontsize=18)
    ax.set_ylabel(r"$y$ $[m]$", fontsize=18)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    out_dir = params.get("pathRes", ".")
    os.makedirs(out_dir, exist_ok=True)

    fname = f"{field.lower()}_{comp}_{suffix}.pdf" if suffix else f"{field.lower()}_{comp}.pdf"
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")

    plt.show()
    plt.close(fig)

def plot_field_pyvista(mesh, ifield, params, values=None, suffix=""):
    """
    Unified PyVista plotting function for both simulation and prediction.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh to plot on.
    ifield : int
        Index of the field in params["plotflow"].
    params : dict
        Configuration dictionary.
    values : array_like or None
        If None, plot the field already stored in the mesh.
        If given, attach these values to a copy of the mesh and plot them.
        Values can be:
          - 1D array of shape (n_points,)
          - 2D array of shape (n_points, ncomp)
    suffix : str
        Extra suffix for file naming, e.g. "sim" or "pred".
    """

    pf = params["plotflow"]

    field = pf["fields"][ifield]
    comp = pf["comp"][ifield]
    cmap = pf["cmap"][ifield]
    clabel = pf["latex"][ifield]

    scale = np.asarray(pf["scale"][ifield], dtype=float)
    vmin = float(np.min(scale))
    vmax = float(np.max(scale))
    clim = (vmin, vmax)

    zoom = pf.get("zoom", 1.02)
    show_edges = pf.get("show_edges", False)
    edge_color = pf.get("edge_color", "black")
    line_width = pf.get("line_width", 0.2)

    out_dir = params.get("pathRes", ".")
    os.makedirs(out_dir, exist_ok=True)

    mesh_plot = mesh.copy()

    # Decide which scalar field to plot
    if values is None:
        # Plot an existing field from the mesh
        if field in mesh_plot.cell_data and field not in mesh_plot.point_data:
            mesh_plot = mesh_plot.cell_data_to_point_data()

        if field not in mesh_plot.array_names:
            raise ValueError(
                f"Field '{field}' was not found in the mesh."
            )

        scalar_name = field
        arr = np.asarray(mesh_plot[scalar_name])

        # If array is vector/tensor-like, use component
        if arr.ndim > 1 and arr.shape[1] > 1:
            use_component = comp
        else:
            use_component = None

    else:
        # Plot provided values by attaching them to the mesh copy
        values = np.asarray(values)

        scalar_name = f"{field}_{comp}_{suffix}" if suffix else f"{field}_{comp}"

        if values.ndim == 1:
            if values.shape[0] != mesh_plot.n_points:
                raise ValueError(
                    f"Values length ({values.shape[0]}) does not match "
                    f"number of mesh points ({mesh_plot.n_points})."
                )
            mesh_plot.point_data[scalar_name] = values.ravel()
            use_component = None

        elif values.ndim == 2:
            if values.shape[0] != mesh_plot.n_points:
                raise ValueError(
                    f"Values shape {values.shape} is incompatible with "
                    f"number of mesh points ({mesh_plot.n_points})."
                )
            mesh_plot.point_data[scalar_name] = values
            use_component = comp if values.shape[1] > 1 else None

        else:
            raise ValueError(
                "values must be a 1D or 2D array."
            )

    # Plot
    pl = pv.Plotter(off_screen=True)

    add_mesh_kwargs = dict(
        scalars=scalar_name,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=False,
        show_edges=show_edges,
        edge_color=edge_color,
        line_width=line_width,
    )

    if use_component is not None:
        add_mesh_kwargs["component"] = use_component

    pl.add_mesh(mesh_plot, **add_mesh_kwargs)

    pl.view_xy()
    pl.camera.zoom(zoom)

    axes_actor = pl.show_bounds(
        mesh=mesh_plot,
        xtitle=pf.get("xtitle", "x [m]"),
        ytitle=pf.get("ytitle", "y [m]"),
        ztitle="",
        location="outer",
        grid=False,
        ticks="outside",
        font_size=16,
        n_xlabels=5,
        n_ylabels=4,
        show_yaxis=False,
        show_zaxis=False,
        use_2d=True,
        bold=False,
        font_family="times",
    )

    pl.add_scalar_bar(
        title=f"{clabel}",
        position_x=0.87,
        position_y=0.438,
        height=0.163,
        width=0.08,
        vertical=True,
        font_family="times",
        n_labels=2,
        fmt="%.2f"
    )

    pl.show(auto_close=False)

    fname = f"{field.lower()}_{comp}"
    if suffix:
        fname += f"_{suffix}"
    fname += ".pdf"

    pl.save_graphic(os.path.join(out_dir, fname))
    pl.close()

def plot_simulation_flow(mesh, params):
    """
    Plot all simulation fields defined in params["plotflow"].
    """
    nfields = len(params["plotflow"]["fields"])
    for ifield in range(nfields):
        plot_field_pyvista(mesh, ifield, params, values=None, suffix="sim")


def plot_predicted_flow_pyvista(mesh, pred_list, params):
    """
    Plot predicted fields on the provided mesh.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh on which the prediction is defined.
    pred_list : list
        List of predicted arrays in the same order as plotflow fields.
    params : dict
        Configuration dictionary.
    """
    for ifield, values in enumerate(pred_list):
        plot_field_pyvista(mesh, ifield, params, values=values, suffix="pred")