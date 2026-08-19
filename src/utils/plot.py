# SPDX-License-Identifier: MIT
import os
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.tri as mtri
from matplotlib import rc, cm
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
rc('text', usetex=False)

def plot_prepared_data(data, params):
    """Plot prepared data splits and collocation points.

    Parameters
    ----------
    data : dict
        Prepared dataset mapping.
    params : dict
        PIRFlow configuration.

    Returns
    -------
    None
        PDF files are written to the result directory.
    """

    # Plot training dataset points
    plot_dataset(
        data["training"]["xtrain"],
        data["training"]["ytrain"],
        params,
        dataset='training',
    )

    # Plot validation dataset points
    if data["validation"]["xval"] is not None:
        plot_dataset(
            data["validation"]["xval"],
            data["validation"]["yval"],
            params,
            dataset='validation',
        )

    # Plot test dataset points
    if data["test"]["xtest"] is not None:
        plot_dataset(
            data["test"]["xtest"],
            data["test"]["ytest"],
            params,
            dataset='test',
        )

    # Plot training data and training collocation points
    plot_target_points(
        data["training"]["xtrain"],
        data["training"]["ytrain"],
        data["collocation"]["xftrain"],
        data["collocation"]["yftrain"],
        params,
        True,
    )

    # Plot all data and all collocation points
    plot_target_points(
        data["all"]["x"],
        data["all"]["y"],
        data["collocation"]["xf"],
        data["collocation"]["yf"],
        params,
    )

def plot_dataset(x, y, params, dataset):
    """Plot one prepared dataset split.

    Parameters
    ----------
    x, y : array_like
        Point coordinates.
    params : dict
        PIRFlow configuration.
    dataset : {"training", "validation", "test"}
        Split name and output filename selector.

    Returns
    -------
    None
        A PDF file is written to the result directory.
    """

    fig, ax = plt.subplots(1, 1, num=1, figsize=(12, 4), sharey=True)
    plt.rc('legend', **{'fontsize': 14})

    p0, = ax.plot(x, y, 'o', color='k', markersize=2)
    ax.tick_params(direction="in", which='both')
    ax.grid(color='0.5', linestyle=':', linewidth=0.5, which='both')
    ax.tick_params(labelsize=18)
    ax.set_xlabel(r'$x$ $[m]$', fontsize=18)
    ax.set_ylabel(r'$y$ $[m]$', fontsize=18)
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.97)
    
    if dataset == 'training':
        ax.legend([p0], [r'Training data'], loc='lower left')
        fig.savefig(params['paths']['results']+'/training_dataset_points.pdf')
    elif dataset == 'validation':
        ax.legend([p0], [r'Validation data'], loc='lower left')
        fig.savefig(params['paths']['results']+'/validation_dataset_points.pdf')
    elif dataset == 'test':
        ax.legend([p0], [r'Test data'], loc='lower left')
        fig.savefig(params['paths']['results']+'/test_dataset_points.pdf')
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    plt.close(fig)

def plot_target_points(x, y, xf, yf, params, training=False):
    """Plot observation and collocation coordinates together.

    Parameters
    ----------
    x, y : array_like
        Observation coordinates.
    xf, yf : array_like or None
        Collocation coordinates.
    params : dict
        PIRFlow configuration.
    training : bool, optional
        Whether to use the training-only output filename.

    Returns
    -------
    None
        A PDF file is written to the result directory.
    """

    fig, ax = plt.subplots(1, 1, num=1, figsize=(12, 4), sharey=True)
    plt.rc('legend', **{'fontsize': 14})

    if xf is not None or yf is not None:
        p0, = ax.plot(xf, yf, 'o', color='darkorange', markersize=2)
    p1, = ax.plot(x, y, 'o', color='g', markersize=2)
    
    ax.tick_params(direction="in", which='both')
    ax.grid(color='0.5', linestyle=':', linewidth=0.5, which='both')
    ax.tick_params(labelsize=18)
    ax.set_xlabel(r'$x$ $[m]$', fontsize=18)
    ax.set_ylabel(r'$y$ $[m]$', fontsize=18)
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.97)

    if xf is not None or yf is not None:
        ax.legend([p0,p1], [r'Residual',r'Data'], loc='lower left')
    else:
        ax.legend([p1], [r'Data'], loc='best')

    if training:
        fig.savefig(params['paths']['results']+'/'+
                    'training_dataset.pdf')
    else:
        fig.savefig(params['paths']['results']+'/'+
                    'all_dataset.pdf')

    plt.close(fig)

def plot_sampling_data(data_points, collocation_points, params):
    """Plot observation and collocation sampling groups.

    Parameters
    ----------
    data_points : dict
        Observation coordinates and point groups.
    collocation_points : dict
        Collocation coordinates and point groups.
    params : dict
        PIRFlow configuration.

    Returns
    -------
    None
        PDF files are written to the result directory.
    """

    if data_points["X"] is not None:
        plot_sampling_points(
            data_points["X"],
            data_points["pts_in"],
            data_points["pts_bc"],
            data_points["pts_grad"],
            params,
            False,
        )

    if collocation_points["Xf"] is not None:
        plot_sampling_points(
            collocation_points["Xf"],
            collocation_points["pts_in"],
            collocation_points["pts_bc"],
            collocation_points["pts_grad"],
            params,
            True,
        )

def plot_sampling_points(pall, pin, pbc, pgrad, params, collpts=False):
    """Plot interior, boundary, and gradient-focused samples.

    Parameters
    ----------
    pall : array_like
        All sampled points; reserved for future plotting.
    pin : array_like
        Interior points.
    pbc : array_like
        Boundary points.
    pgrad : array_like
        Gradient-based points.
    params : dict
        PIRFlow configuration.
    collpts : bool, optional
        Whether to use the collocation output filename.

    Returns
    -------
    None
        A PDF file is written to the result directory.
    """

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
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.97)

    ax.legend([p0,p1,p2], [r'Inner',
                           r'Boundary',
                           r'$\left|\nabla \rho\right|^{\alpha}$'], 
                           loc='lower left')
    
    if collpts:
        fig.savefig(params['paths']['results']+'/'+
                    'collocation_dataset_points.pdf')
    else:
        fig.savefig(params['paths']['results']+'/'+
                    'data_dataset_points.pdf')

    plt.close(fig)

def plot_history_training(model, params):
    """Plot training and validation loss histories.

    Parameters
    ----------
    model : PhysicsInformedNN
        Trained model containing loss histories.
    params : dict
        PIRFlow configuration.

    Returns
    -------
    None
        PDF files are written to the result directory.
    """

    # Get losses
    l_data = model.get_data_loss()
    l_res = model.get_residual_loss()
    l_total = model.get_total_loss()
    l_val = model.get_validation_data_loss()
    n_epoch = model.get_n_epoch()
    
    # Plot losses
    plot_losses(l_data, l_res, l_total, n_epoch, params)
    # Plot validation loss
    plot_validation_loss(l_data, l_val, n_epoch, params)

def plot_losses(l_data, l_res, l_total, n_epoch, params):
    """Plot data, residual, and total training losses.

    Parameters
    ----------
    l_data : array_like
        Data-loss history.
    l_res : array_like
        Residual-loss history.
    l_total : array_like
        Total-loss history.
    n_epoch : int
        Number of recorded optimizer steps.
    params : dict
        PIRFlow configuration.

    Returns
    -------
    None
        A PDF file is written to the result directory.
    """

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.rc('legend', **{'fontsize': 14})

    # Epochs calculation
    epochs = np.arange(0, n_epoch, 1)

    p0, = ax.semilogy(epochs, l_data, '-', color='b', linewidth=2)
    if params['run']['model'] == 'pinn': 
        p1, = ax.semilogy(epochs, l_res, '-', color='r', linewidth=2)
        p2, = ax.semilogy(epochs, l_total, '-', color='k', linewidth=2)
        ax.legend([p0,p1,p2], [r'Data loss',r'Residual loss',r'Total loss'], loc='best')
    else:
        ax.legend([p0], [r'Data loss'], loc='best')

    ax.tick_params(direction="in", which='both')
    fig.subplots_adjust(left=0.146, right=0.97, bottom=0.124, top=0.97)
    ax.grid(color='0.5', linestyle=':', linewidth=0.5, which='both')
    ax.set_ylim(0.0001, 200.0)
    ax.tick_params(labelsize=18)
    ax.set_xlabel(r'$Epochs$', fontsize=18)
    ax.set_ylabel(r'$Losses$', fontsize=18)
    fig.savefig(params['paths']['results']+'/training_losses.pdf')
    plt.close()

def plot_validation_loss(l_train, l_val, n_epoch, params):
    """Plot training-data and validation losses.

    Parameters
    ----------
    l_train : array_like
        Training-loss history.
    l_val : array_like
        Validation-loss history.
    n_epoch : int
        Number of recorded optimizer steps.
    params : dict
        PIRFlow configuration.

    Returns
    -------
    None
        A PDF file is written to the result directory.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.rc('legend', **{'fontsize': 14})

    # Epochs calculation
    epochs = np.arange(0, n_epoch, 1)

    p0, = ax.semilogy(epochs, l_train, '-', color='b', linewidth=2)
    p1, = ax.semilogy(epochs, l_val, '-', color='r', linewidth=2)
    ax.legend([p0,p1], [r'Training data loss',r'Validation data loss'], loc='best')

    ax.tick_params(direction="in", which='both')
    fig.subplots_adjust(left=0.146, right=0.97, bottom=0.124, top=0.97)
    ax.grid(color='0.5', linestyle=':', linewidth=0.5, which='both')
    ax.set_ylim(0.0001, 200.0)
    ax.tick_params(labelsize=18)
    ax.set_xlabel(r'$Epochs$', fontsize=18)
    ax.set_ylabel(r'$Losses$', fontsize=18)
    fig.savefig(params['paths']['results']+'/validation_loss.pdf')
    plt.close()

def plot_field_pyvista(
    mesh,
    ifield,
    params,
    values=None,
    suffix="",
    clabel=None,
    clim=None,
    cmap=None,
):
    """Render one simulation, prediction, or error field with PyVista.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh used for rendering.
    ifield : int
        Field index in the post-processing configuration.
    params : dict
        PIRFlow configuration.
    values : array_like or None
        Optional values to attach instead of using an existing mesh field.
    suffix : str
        Output filename suffix.
    clabel : str or None
        Optional colorbar label override.
    clim : tuple or None
        Optional color-limit override.
    cmap : str or None
        Optional colormap override.

    Returns
    -------
    None
        An off-screen PDF rendering is written to disk.
    """

    pf = params["post_processing"]

    field = pf["fields"][ifield]
    comp = pf["components"][ifield]

    if cmap is None:
        cmap = pf["colormaps"][ifield]

    if clabel is None:
        clabel = pf["latex"][ifield]

    if clim is None:
        scale = np.asarray(pf["scales"][ifield], dtype=float)
        vmin = float(np.min(scale))
        vmax = float(np.max(scale))
        clim = (vmin, vmax)

    zoom = 1.02
    show_edges = False
    edge_color = "black"
    line_width = 0.2

    out_dir = params["paths"]["results"]
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
            raise ValueError("values must be a 1D or 2D array.")

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

    pl.show_bounds(
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
        title=clabel,
        position_x=0.87,
        position_y=0.438,
        height=0.163,
        width=0.08,
        vertical=True,
        font_family="times",
        n_labels=2,
        fmt="%.2f",
    )

    pl.show(auto_close=False)

    fname = f"{field.lower()}_{comp}"

    if suffix:
        fname += f"_{suffix}"

    fname += ".pdf"

    pl.save_graphic(os.path.join(out_dir, fname))
    pl.close()

def plot_simulation_flow_pyvista(mesh, params):
    """Plot all configured reference simulation fields.

    Parameters
    ----------
    mesh : pyvista.DataSet
        CFD mesh containing reference fields.
    params : dict
        PIRFlow configuration.

    Returns
    -------
    None
        PDF renderings are written to disk.
    """

    nfields = len(params["post_processing"]["fields"])
    for ifield in range(nfields):
        plot_field_pyvista(
            mesh,
            ifield,
            params,
            values=None,
            suffix="sim",
        )

def plot_predicted_flow_pyvista(mesh, predicted_fields, params):
    """Plot all configured predicted fields.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh on which predictions are defined.
    predicted_fields : dict
        Predicted arrays keyed by flow variable.
    params : dict
        PIRFlow configuration.

    Returns
    -------
    None
        PDF renderings are written to disk.
    """

    variables = get_flow_variables(params)
    nfields = len(params["post_processing"]["fields"])

    if nfields > len(variables):
        raise ValueError(
            f"params['post_processing']['fields'] has {nfields} fields, "
            f"but equation {params['run']['equation']} only defines "
            f"{len(variables)} variables."
        )

    for ifield in range(nfields):
        variable = variables[ifield]

        if variable not in predicted_fields:
            raise KeyError(
                f"Predicted field '{variable}' was not found. "
                f"Available predicted fields are: {list(predicted_fields.keys())}"
            )

        values = predicted_fields[variable]

        plot_field_pyvista(
            mesh,
            ifield,
            params,
            values=values,
            suffix="pred",
        )

def plot_error_flow_pyvista(
    mesh,
    error_fields,
    params,
    error_type="abs_error_01",
):
    """Plot all configured pointwise error fields.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh on which errors are defined.
    error_fields : dict
        Error arrays keyed by variable and error type.
    params : dict
        PIRFlow configuration.
    error_type : str, optional
        Error representation to plot; currently ``"abs_error_01"``.

    Returns
    -------
    None
        PDF renderings are written to disk.
    """

    valid_error_types = [
        #"error",
        #"abs_error",
        #"rel_error",
        "abs_error_01",
    ]

    if error_type not in valid_error_types:
        raise ValueError(
            f"Unknown error_type '{error_type}'. "
            f"Use one of: {valid_error_types}."
        )

    variables = get_flow_variables(params)
    nfields = len(params["post_processing"]["fields"])

    if nfields > len(variables):
        raise ValueError(
            f"params['post_processing']['fields'] has {nfields} fields, "
            f"but equation {params['run']['equation']} only defines "
            f"{len(variables)} variables."
        )

    # Labels are defined explicitly because the plotted error fields
    # are scaled and therefore do not use the original field labels.
    error_labels = {
        "rho": r"$\rho_{\mathrm{error}}$",
        "p": r"$p_{\mathrm{error}}$",
        "u": r"$u_{\mathrm{error}}$",
        "v": r"$v_{\mathrm{error}}$",
        "mut": r"$\widehat{\mu}_{t,\mathrm{error}}$",
    }

    for ifield in range(nfields):
        # Variable names
        variable = variables[ifield]
        # Also created in compute_error_fields
        error_name = f"{variable}_{error_type}" 

        if error_name not in error_fields:
            raise KeyError(
                f"Error field '{error_name}' was not found. "
                f"Available error fields are: "
                f"{list(error_fields.keys())}"
            )

        if variable not in error_labels:
            raise KeyError(
                f"No error label has been defined for variable "
                f"'{variable}'."
            )

        values = error_fields[error_name]
        clabel = error_labels[variable]

        if error_type == "abs_error_01":
            clim = (0.0, 1.0) # graph scale
        else:
            clim = None

        plot_field_pyvista(
            mesh,
            ifield,
            params,
            values=values,
            suffix=error_name,
            clabel=clabel,
            clim=clim,
            cmap=params["post_processing"].get(
                "error_colormap",
                "bwr",
            ),
        )
        
def get_flow_variables(params):
    """Return flow-variable names in post-processing order.

    Parameters
    ----------
    params : dict
        PIRFlow configuration.

    Returns
    -------
    list of str
        Euler or RANS variable names.
    """

    equation = params["run"]["equation"]

    if equation == "euler":
        return ["rho", "p", "u", "v"]

    if equation == "rans":
        return ["rho", "p", "u", "v", "mut"]

    raise ValueError(f"Unknown equation: {equation}.")
