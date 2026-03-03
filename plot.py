import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import rc, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
rc('text', usetex=True)

def PlotSamplingPoints(X, params):

    fig, ax = plt.subplots(1, 1, num=1, figsize=(12, 4), sharey=True)
    plt.rc('legend', **{'fontsize': 16})

    p0, = ax.plot(X[:,0], X[:,1], 'o', color='r', markersize=3)

    ax.tick_params(direction="in", which='both')
    ax.grid(color='0.5', linestyle=':', linewidth=0.5, which='both')
    #ax.set_xlim(0.0, 0.0)
    #ax.set_ylim(0.0, 0.0)
    ax.tick_params(labelsize=18)
    ax.set_xlabel(r'$x (mm)$', fontsize=18)
    ax.set_ylabel(r'$y (mm)$', fontsize=18)

    ax.legend([p0], [r'samples'], loc='best')

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.97)
    fig.savefig(params['pathRes']+'/'+params['sampling']['graph_filename']+'.pdf')
    plt.show()
    plt.close(fig)
    
def PlotPredictedFlow(x, y, u_pred, params):
    
    tri = mtri.Triangulation(x, y)

    tris = tri.triangles
    xtri = x[tris]
    ytri = y[tris]

    e0 = np.hypot(xtri[:,1]-xtri[:,0], ytri[:,1]-ytri[:,0])
    e1 = np.hypot(xtri[:,2]-xtri[:,1], ytri[:,2]-ytri[:,1])
    e2 = np.hypot(xtri[:,0]-xtri[:,2], ytri[:,0]-ytri[:,2])
    emax = np.maximum.reduce([e0, e1, e2])

    thr = np.percentile(emax, 90) * 1.5
    tri.set_mask(emax > thr)

    fig, ax = plt.subplots(1, 1, num=2, figsize=(12, 4), sharey=True)
    cs = ax.tricontourf(tri, u_pred.ravel(), levels=50, cmap="coolwarm")

    cbar = fig.colorbar(
        cs, ax=ax,
        shrink=1.0,      # length of the bar (0-1)
        fraction=0.05,    # thickness (relative to axes)
        pad=0.02,         # gap between plot and colorbar
        aspect=30         # also affects thickness vs length
    )
    cbar.set_label(r"$\rho$", fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    ax.tick_params(labelsize=18)
    ax.set_xlabel(r'$x (mm)$', fontsize=18)
    ax.set_ylabel(r'$y (mm)$', fontsize=18)

    #ax.set_aspect("equal", adjustable="box")
    ax.set_title(r"$\rho$ (triangulation with masked long-edge triangles)")
    fig.tight_layout()

    fig.savefig(params['pathRes'] + '/flowfield.pdf')
    plt.show()
    plt.close(fig)
    