import numpy as np
import torch
from torch.autograd import Variable
from operations import gradient

# kinetic energy
def KE(v):
    return 0.5 * v.dot(v)


# Hamiltonian
def H(x, v, E):
    return E(x) + KE(v)


# n-step leapfrog
def leapfrog(x_0, v_0, E, eps, n):
    v = v_0 - 0.5 * eps * gradient(E, x_0)
    x = x_0 + eps * v

    for i in range(n):
        v = v - eps * gradient(E, x)
        x = x + eps * v

    v = v - 0.5 * eps * gradient(E, x)

    return x, v


# get one sample
def sample(x_0, E, eps, n):
    v_0 = Variable(torch.randn(x_0.size()))
    x, v = leapfrog(x_0, v_0, E, eps, n)
    accept = torch.min(Variable(torch.Tensor(1.0)), torch.exp(H(x_0, v_0, E) - H(x, v, E)))
    if accept.data.numpy() > np.random.rand():
        return x
    else:
        return x_0


# run hmc:
def hmc(E, eps, n, num_samples):
