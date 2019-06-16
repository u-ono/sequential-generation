import chainer
import chainer.functions as F
import chainer.links as L
from nets.resblocks.resblock import DownBlock as Block
from nets.resblocks.resblock import OptimizedDownBlock as OptimizedBlock
from nets.resblocks.sn_resblock import DownBlock as SNBlock
from nets.resblocks.sn_resblock import OptimizedDownBlock as SNOptimizedBlock
from src.links.sn_linear import SNLinear


class ResNetEncoder(chainer.Chain):

    def __init__(self, dim_z, ch=64, activation=F.leaky_relu, distribution='normal'):
        super(ResNetEncoder, self).__init__()
        self.activation = activation
        self.distribution = distribution
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.block1 = OptimizedBlock(3, ch)
            self.block2 = Block(ch, ch * 2, activation=activation, downsample=True)
            self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
            self.block4 = Block(ch * 4, ch * 8, activation=activation, downsample=True)
            self.block5 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
            if distribution == 'normal':
                self.l_m = L.Linear(ch * 16, dim_z, initialW=initializer)
                self.l_v = L.Linear(ch * 16, dim_z, initialW=initializer)
            elif distribution == 'vmf':
                self.l_m = L.Linear(ch * 16, dim_z, initialW=initializer)
                self.l_v = L.Linear(ch * 16, 1, initialW=initializer)
            else:
                raise NotImplementedError

    def __call__(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = F.sum(h, axis=(2, 3))  # Global pooling
        if self.distribution == 'normal':
            mu = self.l_m(h)
            var = F.softplus(self.l_v(h))
        elif self.distribution == 'vmf':
            mu = F.normalize(self.l_m(h), axis=-1)
            var = F.softplus(self.l_v(h)) + 1
        else:
            raise NotImplementedError
        return mu, var


class SNResNetEncoder(chainer.Chain):

    def __init__(self, dim_z, ch=64, activation=F.relu, distribution='normal'):
        super(SNResNetEncoder, self).__init__()
        self.activation = activation
        self.distribution = distribution
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.block1 = SNOptimizedBlock(3, ch)
            self.block2 = SNBlock(ch, ch * 2, activation=activation, downsample=True)
            self.block3 = SNBlock(ch * 2, ch * 4, activation=activation, downsample=True)
            self.block4 = SNBlock(ch * 4, ch * 8, activation=activation, downsample=True)
            self.block5 = SNBlock(ch * 8, ch * 16, activation=activation, downsample=True)
            if distribution == 'normal':
                self.l_m = SNLinear(ch * 16, dim_z, initialW=initializer)
                self.l_v = SNLinear(ch * 16, dim_z, initialW=initializer)
            elif distribution == 'vmf':
                self.l_m = SNLinear(ch * 16, dim_z, initialW=initializer)
                self.l_v = SNLinear(ch * 16, 1, initialW=initializer)
            else:
                raise NotImplementedError

    def __call__(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = F.sum(h, axis=(2, 3))  # Global pooling
        if self.distribution == 'normal':
            mu = self.l_m(h)
            var = F.softplus(self.l_v(h))
        elif self.distribution == 'vmf':
            mu = F.normalize(self.l_m(h), axis=-1)
            var = F.softplus(self.l_v(h)) + 1
        else:
            raise NotImplementedError
        return mu, var


class ResNetHSEncoder(chainer.Chain):

    def __init__(self, dim_z, ch=64, activation=F.leaky_relu):
        super(ResNetHSEncoder, self).__init__()
        self.activation = activation
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.block1 = OptimizedBlock(3, ch)
            self.block2 = Block(ch, ch * 2, activation=activation, downsample=True)
            self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
            self.block4 = Block(ch * 4, ch * 8, activation=activation, downsample=True)
            self.block5 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
            self.l_m = L.Linear(ch * 16, dim_z, initialW=initializer)
            self.l_v = L.Linear(ch * 16, 1, initialW=initializer)

    def __call__(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = F.sum(h, axis=(2, 3))  # Global pooling
        mu = F.normalize(self.l_m(h), axis=-1)
        var = F.softplus(self.l_v(h)) + 1
        return mu, var
