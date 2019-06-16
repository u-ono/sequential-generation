import math
import chainer
import chainer.functions as F
from chainer.distribution import register_kl
from chainer import Variable

from src.distribution.hypterspherical_uniform import HyperSphericalUniform
from src.functions.ive import ive


class VonMisesFisher(chainer.Distribution):

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.__m = self.xp.array(loc.shape[-1], 'f')
        self.__e1 = self.xp.array([1] + [0] * (loc.shape[-1] - 1), 'f')

    @property
    def batch_shape(self):
        return self.loc.shape[:-1]

    @property
    def event_shape(self):
        return self.loc.shape[-1:]

    @property
    def entropy(self):
        e = - self.scale * ive(self.__m / 2, self.scale) / ive(self.__m / 2 - 1, self.scale)
        return e + self.__log_normalization()

    @property
    def params(self):
        return {'loc': self.loc, 'scale': self.scale}

    def __log_normalization(self):
        xp = self.xp
        ln = - (self.__m / 2 - 1) * F.log(self.scale) \
             + (self.__m / 2) * F.log(2 * xp.array(math.pi, 'f')) \
             + (self.scale + F.log(ive(self.__m / 2 - 1, self.scale)))
        return F.reshape(ln, ln.shape[:-1])

    def sample_n(self, n):
        xp = self.xp

        if self.__m == 3:
            w = self.__sample_w_3(n)
        else:
            w = self.__sample_w_r(n)

        size = (n,) + self.batch_shape + (self.event_shape[0] - 1,)
        v = xp.random.normal(size=size)
        norm = xp.linalg.norm(v, axis=-1, keepdims=True) + 1e-05
        v = v / norm

        x = F.concat([w, F.sqrt(1 - w ** 2) * v], axis=-1)
        z = self.__householder_rotation(x)

        return z

    def __sample_w_3(self, n):
        xp = self.xp
        u = xp.random.uniform(size=(n,) + self.scale.shape)
        self.__w = 1 + F.logsumexp(F.stack([F.log(u), F.log(1 - u) - 2 * self.scale], axis=0), axis=0) / self.scale
        return self.__w

    def __sample_w_r(self, n):
        xp = self.xp

        c = F.sqrt(4 * self.scale ** 2 + (self.__m - 1) ** 2)

        b_true = (- 2 * self.scale + c) / (self.__m - 1)
        b_app = (self.__m - 1) / (4 * self.scale)

        s = F.minimum(
            F.maximum(self.scale - 10, xp.zeros(self.scale.shape, 'f')),
            xp.ones(self.scale.shape, 'f')
        )

        b = b_app * s + b_true * (1 - s)

        a = (self.__m - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self.__m - 1) * xp.log(self.__m - 1)

        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, n)
        return self.__w

    def __while_loop(self, b, a, d, n):
        xp = self.xp

        b, a, d = [F.tile(e, (n, *([1] * len(self.scale.shape)))) for e in (b, a, d)]

        w = Variable(xp.zeros_like(b))
        e = Variable(xp.zeros_like(b))
        mask = xp.ones_like(b) == 1

        size = (n,) + self.scale.shape

        while xp.sum(mask) != 0:
            e_ = xp.random.beta((self.__m - 1) / 2, (self.__m - 1) / 2, size=size)
            u = xp.random.uniform(size=size)

            w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
            t = (2 * a * b) / (1 - (1 - b) * e_)

            accept = ((self.__m - 1) * xp.log(t.data) - t.data + d.data) > xp.log(u)
            reject = 1 - accept

            w += F.where(accept * mask, w_, xp.zeros_like(b))
            e += F.where(accept * mask, w_, xp.zeros_like(b))

            mask[mask * accept] = reject[mask * accept]

        return e, w

    def __householder_rotation(self, x):
        u = F.normalize(self.__e1 - self.loc, axis=-1)
        z = x - 2 * F.sum(x * u, axis=-1, keepdims=True) * u
        return z


@register_kl(VonMisesFisher, HyperSphericalUniform)
def _kl_vmf_uniform(vmf, hsu):
    return - vmf.entropy + hsu.entropy
