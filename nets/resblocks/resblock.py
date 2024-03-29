import math
import chainer
import chainer.links as L
from chainer import functions as F
from src.links.categorical_conditional_batch_normalization import CategoricalConditionalBatchNormalization


def _upsample(x):
    h, w = x.shape[2:]
    return F.unpooling_2d(x, 2, outsize=(h * 2, w * 2))


def upsample_conv(x, conv):
    return conv(_upsample(x))


class UpBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, n_classes=0):
        super(UpBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        with self.init_scope():
            self.c1 = L.Convolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = L.Convolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            if n_classes > 0:
                self.b1 = CategoricalConditionalBatchNormalization(in_channels, n_cat=n_classes)
                self.b2 = CategoricalConditionalBatchNormalization(hidden_channels, n_cat=n_classes)
            else:
                self.b1 = L.BatchNormalization(in_channels)
                self.b2 = L.BatchNormalization(hidden_channels)
            if self.learnable_sc:
                self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x, y=None, z=None, **kwargs):
        h = x
        h = self.b1(h, y, **kwargs) if y is not None else self.b1(h, **kwargs)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, y, **kwargs) if y is not None else self.b2(h, **kwargs)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def __call__(self, x, y=None, z=None, **kwargs):
        return self.residual(x, y, z, **kwargs) + self.shortcut(x)


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return F.average_pooling_2d(x, 2)


class DownBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.leaky_relu, downsample=False):
        super(DownBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        with self.init_scope():
            self.c1 = L.Convolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = L.Convolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.b1 = L.BatchNormalization(in_channels)
            self.b2 = L.BatchNormalization(hidden_channels)
            if self.learnable_sc:
                self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)


class OptimizedDownBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.leaky_relu):
        super(OptimizedDownBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        with self.init_scope():
            self.c1 = L.Convolution2D(in_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = L.Convolution2D(out_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)
