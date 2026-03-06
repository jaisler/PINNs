import os
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.tri as mtri
from matplotlib import rc, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
rc('text', usetex=True)

def PlotSamplingPoints(pall, pin, pbc, pgrad, params):

    fig, ax = plt.subplots(1, 1, num=1, figsize=(12, 4), sharey=True)
    plt.rc('legend', **{'fontsize': 14})

    p0, = ax.plot(pin[:,0], pin[:,1], 'o', color='r', markersize=3)
    p1, = ax.plot(pbc[:,0], pbc[:,1], 'o', color='b', markersize=3)
    p2, = ax.plot(pgrad[:,0], pgrad[:,1], 'o', color='m', markersize=3)
    #p3, = ax.plot(pall[:,0], pall[:,1], 'o', color='k', markersize=2)

    ax.tick_params(direction="in", which='both')
    ax.grid(color='0.5', linestyle=':', linewidth=0.5, which='both')
    #ax.set_xlim(0.0, 0.0)
    #ax.set_ylim(0.0, 0.0)
    ax.tick_params(labelsize=18)
    ax.set_xlabel(r'$x$ $[m]$', fontsize=18)
    ax.set_ylabel(r'$y$ $[m]$', fontsize=18)

    ax.legend([p0,p1,p2], [r'Inner',
                           r'Boundary',
                           r'$\left|\nabla \rho\right|^{\alpha}$'], loc='best')
    
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.97)
    fig.savefig(params['pathRes']+'/'+params['sampling']['fpoints']+'.pdf')
    plt.show()
    plt.close(fig)

def PlotTargetPoints(x, y, params):

    fig, ax = plt.subplots(1, 1, num=1, figsize=(12, 4), sharey=True)
    plt.rc('legend', **{'fontsize': 14})

    p0, = ax.plot(x, y, 'o', color='g', markersize=3)

    ax.tick_params(direction="in", which='both')
    ax.grid(color='0.5', linestyle=':', linewidth=0.5, which='both')
    #ax.set_xlim(0.0, 0.0)
    #ax.set_ylim(0.0, 0.0)
    ax.tick_params(labelsize=18)
    ax.set_xlabel(r'$x$ $[m]$', fontsize=18)
    ax.set_ylabel(r'$y$ $[m]$', fontsize=18)

    ax.legend([p0], [r'Training points'], loc='best')
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.97)
    fig.savefig(params['pathRes']+'/'+params['sampling']['ftargets']+'.pdf')
    plt.show()
    plt.close(fig)

def PlotPredictedFlow(x, y, u_pred, params, perc=90, mult=1.5, debug=True):
    
    tri = mtri.Triangulation(x, y)

    tris = tri.triangles
    
    xtri = x[tris]
    ytri = y[tris]

    e0 = np.hypot(xtri[:,1]-xtri[:,0], ytri[:,1]-ytri[:,0])
    e1 = np.hypot(xtri[:,2]-xtri[:,1], ytri[:,2]-ytri[:,1])
    e2 = np.hypot(xtri[:,0]-xtri[:,2], ytri[:,0]-ytri[:,2])
    emax = np.maximum.reduce([e0, e1, e2])

    thr = np.percentile(emax, perc) * mult
    tri.set_mask(emax > thr)

    fig, ax = plt.subplots(1, 1, num=2, figsize=(12, 4), sharey=True)
    
    # Scale
    vmin, vmax = 0.0, 1.3
    levels = np.linspace(vmin, vmax, 51) 
    cs = ax.tricontourf(tri, u_pred.ravel(), cmap="bwr",
                        levels=levels, vmin=vmin, vmax=vmax, extend="both")

    cbar = fig.colorbar(
        cs, ax=ax,
        shrink=0.42,       # length of the bar (0-1)
        fraction=0.1,    # thickness (relative to axes)
        pad=0.02,         # gap between plot and colorbar
        aspect=10,         # also affects thickness vs length
        ticks=[0.0, 0.5, 1.0, 1.3]
    )
    cbar.set_label(r"$\rho$ $[kg/m^{3}]$", fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    ax.tick_params(labelsize=18)
    ax.set_xlabel(r'$x$ $[m]$', fontsize=18)
    ax.set_ylabel(r'$y$ $[m]$', fontsize=18)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(params['pathRes'] + '/predict_flow.pdf')
    plt.show()
    plt.close(fig)

def PlotFlowField(flowfield, params):
    fields = params["plotflow"]["fields"]
    latex = params["plotflow"]["latex"]
    cmap = params["plotflow"]["cmap"]
    comp = params["plotflow"]["comp"]
    scale = params["plotflow"]["scale"]

    if isinstance(fields, str):
        fields = [fields]
    if isinstance(latex, str):
        latex = [latex]
    if isinstance(cmap, str):
        cmap = [cmap]

    out_dir = params.get("pathRes", ".")
    os.makedirs(out_dir, exist_ok=True)

    zoom = params["plotflow"].get("zoom", 1.02)

    for f, l, c, k, clim in zip(fields, latex, cmap, comp, scale):
        # normalize / safe order
        vmin, vmax = float(clim[0]), float(clim[1])
        clim = (min(vmin, vmax), max(vmin, vmax))

        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(flowfield, scalars=f, cmap=c, component=k, 
                    clim=clim, show_scalar_bar=False)
        pl.view_xy()
        pl.camera.zoom(zoom)
        pl.add_scalar_bar(title = f"{l}",
                          position_x=0.87, position_y=0.438,
                          height=0.163, width=0.08, vertical=True)
                          #fmt="%.2f")
        
        pl.show(auto_close=False)  # init render
        pl.save_graphic(os.path.join(out_dir, f"{f}"+str(k)+".pdf"))
        pl.close()