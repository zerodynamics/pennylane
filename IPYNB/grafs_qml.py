import pudb as dbg

from math import pi
from scipy.linalg import hadamard
from scipy.signal.windows import dpss
from matplotlib import pyplot as plt

# pennyland/autgrad numpy
import pennylane.numpy as np
import pennylane as qml

from pennylane.ops.qubit import Rot,RotXY
from pennylane.init import strong_ent_layers_uniform
from pennylane.templates.layers import StronglyEntanglingLayersRotXY
from pennylane.templates.layers import StronglyEntanglingLayers

#the og
import numpy as onp

#@qml.qnode(dev)
def cc(params):
    #RotXY(*params,wires=[0])
    StronglyEntanglingLayersRotXY(params,range(4))
    return qml.expval(qml.PauliX(0))

obs = [
  qml.PauliX(1),
  qml.PauliZ(1),
  qml.PauliX(0) @ qml.PauliX(1),
  qml.PauliY(0) @ qml.PauliY(1),
  qml.PauliZ(0) @ qml.PauliZ(1)
]

# num layers must be power of 2 for walsh basis
num_layers = 2**4
num_wires  = 5

# walsh basis functions (columns)
# V is square ... take one less column for dimension counting/sanity
V = hadamard(num_layers)[:,:-1]
num_basis = V.shape[1]

# initial coefficients
alpha_init = onp.random.rand(num_basis,num_wires,2)

## MODIFIED default.qubit device to add custom operation "RotXY"
dev = qml.device("default.qubit", wires=num_wires, analytic=True)

# Map our ansatz over our list of observables,
qnodes = qml.map(StronglyEntanglingLayersRotXY, obs, device=dev)

# should have shape (num_layers, num_wires, 2)
init_params = np.random.rand(num_layers,num_wires,2)

# choose a circuit from qnodes list
circuit = qnodes[4]

# create tape
circuit(init_params)

################################################################
# GRAFS functions

def grafs_circuit(alpha):
    # transform global (alpha) --> local (theta)
    theta = onp.tensordot(V,alpha,([1],[0]))
    # evaluate the circuit
    return circuit(theta)

def grafs_grad(alpha):
    # alpha --> theta
    theta = onp.tensordot(V,alpha,([1],[0]))
    # compute gradient wrt theta with pennylane/autograd
    cc = qml.grad(circuit)(theta)[0]
    # complete the chain-rule to get gradient wrt alpha
    return onp.tensordot(V,cc,([0],[0]))

def grafs_step(alpha,lr):
    # gradient descent in function space
    return alpha - lr*grafs_grad(alpha)

##############################################################################


steps = 20
learning_rate = 0.01

# vanilla gradient descent
opt   = qml.GradientDescentOptimizer(learning_rate)

# natural gradient optimizer
opt_ng = qml.QNGOptimizer(learning_rate)

# alpha --> theta
theta = np.tensordot(V,alpha_init,([1],[0]))

# for natural gradient
theta_ng = theta

gd_cost = []
qng_cost = []
grafs_cost = []

# initial cost
gd_cost.append(circuit(theta))
grafs_cost.append(grafs_circuit(alpha_init))
print('%f : %f' %(gd_cost[-1],grafs_cost[-1]))

alpha = alpha_init

for _ in range(steps):

    # take a theta step
    theta = opt.step(circuit, theta)
    gd_cost.append(circuit(theta))

    # take an alpha step
    alpha = grafs_step(alpha,learning_rate)
    grafs_cost.append(grafs_circuit(alpha))

    # natural gradient ... singular
    #theta_ng = opt_ng.step(circuit, theta_ng)
    #qng_cost.append(circuit(theta_ng))

    print('%f : %f' %(gd_cost[-1],grafs_cost[-1]))

dbg.set_trace()
plt.style.use("seaborn")
plt.plot(gd_cost, "b", label="gradient descent")
plt.plot(grafs_cost, "g", label="GRAFS")

plt.ylabel("Cost function value")
plt.xlabel("Optimization steps")
plt.legend()
plt.show()











#
