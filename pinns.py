import tensorflow as tf
import numpy as np

np.random.seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, u, v, rho, p, params): 
         
        X = np.column_stack([x, y])
        
        self.lb = X.min(0)
        self.ub = X.max(0)

        # Spatial coordinates
        self.X = X
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        # Velocity components
        self.u = u
        self.v = v
        # Density
        self.rho = rho
        # Pressure
        self.p = p
        # Layers
        self.layers = params['layers']
        # Heat capacity ratio
        self.gamma = params['gamma']
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(self.layers) 
                
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], 
                dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
    
    def xavier_init(self, size):
        # If the activation variance collapses layer-by-layer, 
        # then the network output becomes almost constant, 
        # so that gradients vanish and training stalls.
        # If the activation variance grows too much, then 
        # activations become huge (or saturate), so that gradients 
        # explode or vanish and training becomes unstable or very slow.
        # That is why initialisation picks a specific variance for the
        # weights.
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim,
            out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
 
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y   
    
def net_steady_euler(self, x, y):
    gamma = self.gamma  # e.g. 1.4

    # NN outputs: rho, u, v, p
    out = self.neural_net(tf.concat([x, y], 1), self.weights, self.biases)
    rho = out[:, 0:1]
    u   = out[:, 1:2]
    v   = out[:, 2:3]
    p   = out[:, 3:4]

    # Conservative variables / flux ingredients
    rhou = rho * u
    rhov = rho * v

    # rhoE from ideal-gas closure
    rhoE = p/(gamma - 1.0) + 0.5 * rho * (u*u + v*v)

    # Fluxes in x
    F1 = rhou
    F2 = rho*u*u + p
    F3 = rho*u*v
    F4 = u * (rhoE + p)

    # Fluxes in y
    G1 = rhov
    G2 = rho*v*u
    G3 = rho*v*v + p
    G4 = v * (rhoE + p)

    # Divergences (steady PDE residuals)
    f1 = tf.gradients(F1, x)[0] + tf.gradients(G1, y)[0]   # continuity
    f2 = tf.gradients(F2, x)[0] + tf.gradients(G2, y)[0]   # x-momentum
    f3 = tf.gradients(F3, x)[0] + tf.gradients(G3, y)[0]   # y-momentum
    f4 = tf.gradients(F4, x)[0] + tf.gradients(G4, y)[0]   # energy

    return rho, u, v, p, f1, f2, f3, f4

def net_compressible_rans(self, x, y):
    # 1. Note that the derivatives should be adapt to 
    # be able to calculate the RANS equation.
    # 2. How can I sample the geometry based on the density gradient?

    return x, y