import numpy as np
import torch
from torch.autograd import Variable
from operations import gradient

from matplotlib import pyplot as plt


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
    accept = torch.min(Variable(torch.Tensor([1.0])), torch.exp(H(x_0, v_0, E) - H(x, v, E)))
    if accept.data.numpy() > np.random.rand():
        return x, True
    else:
        return x_0, False


# run hmc:
def hmc(num_samples, burn_in, x_0, E, eps, n):
    result = []
    x = Variable(torch.FloatTensor(x_0))
    for i in range(burn_in):
        x, _ = sample(x, E, eps, n)
    for i in range(num_samples):
        x, _ = sample(x, E, eps, n)
        result.append(x.data.numpy())
    return np.vstack(result)


# adaptive hmc:
def adaptive_hmc(num_samples, burn_in, x_0, E, eps, n=10,
                 accept_rate_residual=0.9, accept_rate_target=0.9,
                 eps_up=1.02, eps_down=0.98, eps_min=1e-3, eps_max=0.25):

    def update_accept_rate(accept_rate, accept):
        if accept_rate is None:
            return float(accept)
        else:
            return accept_rate_residual * accept_rate + (1 - accept_rate_residual) * float(accept)

    def update_eps(eps, accept_rate):
        result = eps * (eps_up if accept_rate > accept_rate_target else eps_down)
        return min(max(result, eps_min), eps_max)

    result = []
    accept_rate = None

    accept_rate_history = []
    eps_history = []

    x = Variable(torch.FloatTensor(x_0))
    for i in range(burn_in):
        x, accept = sample(x, E, eps, n)
        accept_rate = update_accept_rate(accept_rate, accept)
        eps = update_eps(eps, accept_rate)
        accept_rate_history.append(accept_rate)
        eps_history.append(eps)
    for i in range(num_samples):
        x, accept = sample(x, E, eps, n)
        accept_rate = update_accept_rate(accept_rate, accept)
        eps = update_eps(eps, accept_rate)
        result.append(x.data.numpy())
        accept_rate_history.append(accept_rate)
        eps_history.append(eps)
    return np.vstack(result), accept_rate_history, eps_history


def test_hmc():
    dim = 5
    mu = np.random.rand(dim, 1) * 10
    cov = np.random.rand(dim, dim)
    cov = (cov + cov.transpose()) / 2.0
    cov[np.arange(dim), np.arange(dim)] = 1.0
    cov_inv = np.linalg.inv(cov)

    mu = Variable(torch.Tensor(mu))
    cov_inv = Variable(torch.Tensor(cov_inv))

    def E(x):
        x = x.unsqueeze(dim=1)
        return 0.5 * (x - mu).t().mm(cov_inv).mm(x - mu).squeeze()

    x_0 = np.random.randn(dim)
    print("x_0", x_0)

    # samples, accept_rate, eps = hmc(num_samples=1000, burn_in=1000, x_0=x_0, E=E, eps=0.2, n=10), None, None
    samples, accept_rate, eps = adaptive_hmc(num_samples=2000, burn_in=2000, x_0=x_0, E=E, eps=0.01, n=5,
                                             accept_rate_target=0.85,
                                             eps_min=0.008, eps_max=0.5)

    print('****** TARGET VALUES ******')
    print('target mean:', mu.squeeze())
    print('target cov:\n', cov)

    print('****** EMPIRICAL MEAN/COV USING HMC ******')
    print('empirical mean: ', samples.mean(axis=0))
    print('empirical_cov:\n', np.cov(samples.T))

    print('****** HMC INTERNALS ******')
    print('final stepsize', eps[-1])
    print('final acceptance_rate', accept_rate[-1])

    plt.plot(accept_rate)
    plt.plot(eps)
    plt.show()


if __name__ == "__main__":
    test_hmc()
