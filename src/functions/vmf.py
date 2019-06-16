from chainer import Function
from chainer import cuda
from scipy.special import ive, gammaln
import numpy as np
import math

from src.distribution.von_mises_fisher import VonMisesFisher


class SphericalKLDivergence(Function):

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)

        mu, kappa = inputs
        dim = mu.shape[-1]

        k = cuda.to_cpu(kappa).astype('float64')

        r = k * ive(dim / 2, k) / (ive(dim / 2 - 1, k) + 1e-150)
        c = (dim / 2 - 1) * np.log(k) - (dim / 2) * np.log(2 * math.pi) - np.log(ive(dim / 2 - 1, k))
        s = (dim / 2) * np.log(math.pi) + np.log(2) - gammaln(dim / 2)

        kl_loss = xp.array(r + c + s, 'f')

        return kl_loss,

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)

        mu, kappa = inputs
        gl, = grad_outputs

        k = cuda.to_cpu(kappa).astype('float64')
        dim = mu.shape[-1]

        r = 0.5 * k * ive(dim / 2 + 1, k) / (ive(dim / 2 - 1, k) + 1e-150)
        r -= 0.5 * k * ive(dim / 2, k) * (ive(dim / 2 - 2, k) + ive(dim / 2, k)) / (ive(dim / 2 - 1, k) + 1e-150) ** 2
        r += 0.5 * k

        gm = xp.zeros(mu.shape, 'f')
        gk = gl * xp.array(r, 'f')

        return gm, gk


def spherical_kl_divergence(mu, kappa):
    return SphericalKLDivergence()(mu, kappa)


def von_mises_fisher(mu, kappa):

    vmf = VonMisesFisher(loc=mu, scale=kappa)
    z = vmf.sample()

    return z
