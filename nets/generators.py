import chainer
import chainer.links as L
from chainer import functions as F
from nets.resblocks.resblock import UpBlock as Block
from src.miscs.random_samples import sample_categorical, sample_continuous


class ResNetGenerator(chainer.Chain):

    def __init__(self, ch=64, dim_z=256, bottom_width=4, activation=F.relu, n_classes=0, distribution="normal"):
        super(ResNetGenerator, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        self.n_classes = n_classes
        with self.init_scope():
            self.l1 = L.Linear(dim_z, (bottom_width ** 2) * ch * 16, initialW=initializer)
            self.block2 = Block(ch * 16, ch * 8, activation=activation, upsample=True, n_classes=n_classes)
            self.block3 = Block(ch * 8, ch * 4, activation=activation, upsample=True, n_classes=n_classes)
            self.block4 = Block(ch * 4, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
            self.block5 = Block(ch * 2, ch, activation=activation, upsample=True, n_classes=n_classes)
            self.b6 = L.BatchNormalization(ch)
            self.l6 = L.Convolution2D(ch, 3, ksize=3, stride=1, pad=1, initialW=initializer)

    def __call__(self, z, y=None, **kwargs):
        if (y is not None) and z.shape[0] != y.shape[0]:
            raise Exception('z.shape[0] != y.shape[0], z.shape[0]={}, y.shape[0]={}'.format(z.shape[0], y.shape[0]))
        h = z
        h = self.l1(h)
        h = F.reshape(h, (h.shape[0], -1, self.bottom_width, self.bottom_width))
        h = self.block2(h, y, **kwargs)
        h = self.block3(h, y, **kwargs)
        h = self.block4(h, y, **kwargs)
        h = self.block5(h, y, **kwargs)
        h = self.b6(h)
        h = self.activation(h)
        h = F.tanh(self.l6(h))
        return h

    def make_hidden(self, batchsize, distribution='normal'):
        if distribution == 'normal':
            z = sample_continuous(self.dim_z, batchsize, distribution=distribution, xp=self.xp)
        elif distribution == 'vmf':
            z = self.xp.random.normal(size=(batchsize, self.dim_z)).astype('f')
            norm = self.xp.linalg.norm(z, axis=-1, keepdims=True) + 1e-5
            z = z / norm
        else:
            raise NotImplementedError
        return z

    def make_condition(self, batchsize):
        y = sample_categorical(self.n_classes, batchsize, distribution="uniform",
                               xp=self.xp) if self.n_classes > 0 else None
        return y
