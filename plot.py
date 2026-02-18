import matplotlib.pyplot as plt
from matplotlib import rc, cm
rc('text', usetex=True)

def plot_sampling_points(X, params):

    fig, ax = plt.subplots(1, 1, num=1, figsize=(12, 4), sharey=True)
    plt.rc('legend', **{'fontsize': 16})

    p0, = ax.plot(X[:,0], X[:,1], 'o', color='r', markersize=3)

    ax.tick_params(direction="in", which='both')
    ax.grid(color='0.5', linestyle=':', linewidth=0.5, which='both')
    #ax.set_xlim(0.0, 0.0)
    #ax.set_ylim(0.0, 0.0)
    ax.tick_params(labelsize=18)

    ax.set_xlabel(r'$x (m)$', fontsize=18)
    ax.set_ylabel(r'$y (m)$', fontsize=18)

    ax.legend([p0], [r'samples'], loc='best')

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.97)
    fig.savefig(params['pathRes']+'/'+params['sampling']['graphfilename']+'.pdf')
    #plt.show()