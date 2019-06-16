from chainer import cuda
from chainer import Function
import numpy as np
import scipy.special


class IveFunction(Function):

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)

        v, z = inputs
        z_cpu = cuda.to_cpu(z)
        v_cpu = cuda.to_cpu(v)

        if np.isclose(v_cpu, 0):
            out = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v_cpu, 1):
            out = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:
            out = scipy.special.ive(v_cpu, z_cpu, dtype=z_cpu.dtype)

        return xp.array(out),

    def backward(self, inputs, grad_outputs):

        v, z = inputs
        gi, = grad_outputs

        gv = None
        gz = gi * (ive(v - 1, z) - ive(v, z) * (v + z) / z)

        return gv, gz


def ive(v, z):
    return IveFunction()(v, z)

