import chainer
import chainer.functions as F
import math


class HyperSphericalUniform(chainer.Distribution):

    def __init__(self, dim, batch_shape, xp):
        self.__dim = dim
        self.__batch_shape = batch_shape
        self.__xp = xp

    @property
    def xp(self):
        return self.__xp

    @property
    def dim(self):
        return self.__dim

    @property
    def event_shape(self):
        return self.__dim + 1,

    @property
    def batch_shape(self):
        return self.__batch_shape

    def sample_n(self, n):
        xp = self.xp

        z = xp.random.normal(size=(n,)+self.batch_shape+self.event_shape)
        norm = xp.linalg.norm(z, axis=-1, keepdims=True) + 1e-05

        return z / norm

    @property
    def entropy(self):
        return self.__log_surface_area()

    def log_prob(self, x):
        xp = self.xp
        lp = - xp.ones(x.shape[:-1]) * self.__log_surface_area()
        return lp

    def __log_surface_area(self):
        xp = self.xp
        c = (self.dim + 1) / 2
        s = F.log(xp.array(2, 'f')) + (c * F.log(xp.array(math.pi, 'f'))) - F.lgamma(xp.array(c, 'f'))
        return s
