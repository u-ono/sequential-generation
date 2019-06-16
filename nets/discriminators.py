import chainer
from chainer import functions as F
from chainer import links as L
from src.links.sn_embed_id import SNEmbedID
from src.links.sn_linear import SNLinear
from nets.resblocks.sn_resblock import DownBlock as SNBlock
from nets.resblocks.sn_resblock import OptimizedDownBlock as SNOptimizedBlock
from nets.resblocks.resblock import OptimizedDownBlock as OptimizedBlock
from nets.resblocks.resblock import DownBlock as Block


class SNResNetProjectionDiscriminator(chainer.Chain):
    def __init__(self, ch=64, n_classes=0, activation=F.relu):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.activation = activation
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.block1 = SNOptimizedBlock(3, ch)
            self.block2 = SNBlock(ch, ch * 2, activation=activation, downsample=True)
            self.block3 = SNBlock(ch * 2, ch * 4, activation=activation, downsample=True)
            self.block4 = SNBlock(ch * 4, ch * 8, activation=activation, downsample=True)
            self.block5 = SNBlock(ch * 8, ch * 16, activation=activation, downsample=True)
            self.l6 = SNLinear(ch * 16, 1, initialW=initializer)
            if n_classes > 0:
                self.l_y = SNEmbedID(n_classes, ch * 16, initialW=initializer)

    def __call__(self, x, y=None):
        h0 = x
        h1 = self.block1(h0)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        h5 = self.block5(h4)
        h5 = self.activation(h5)
        h5 = F.sum(h5, axis=(2, 3))  # Global pooling
        logit = self.l6(h5)
        if y is not None:
            w_y = self.l_y(y)
            logit += F.sum(w_y * h5, axis=1, keepdims=True)
        return logit, h2


class ResNetProjectionDiscriminator(chainer.Chain):
    def __init__(self, ch=64, n_classes=0, activation=F.leaky_relu):
        super(ResNetProjectionDiscriminator, self).__init__()
        self.activation = activation
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.block1 = OptimizedBlock(3, ch)
            self.block2 = Block(ch, ch * 2, activation=activation, downsample=True)
            self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
            self.block4 = Block(ch * 4, ch * 8, activation=activation, downsample=True)
            self.block5 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
            self.l6 = L.Linear(ch * 16, 1, initialW=initializer)
            if n_classes > 0:
                self.l_y = L.EmbedID(n_classes, ch * 16, initialW=initializer)

    def __call__(self, x, y=None):
        h0 = x
        h1 = self.block1(h0)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        h5 = self.block5(h4)
        h5 = self.activation(h5)
        h5 = F.sum(h5, axis=(2, 3))  # Global pooling
        logit = self.l6(h5)
        if y is not None:
            w_y = self.l_y(y)
            logit += F.sum(w_y * h5, axis=1, keepdims=True)
        return logit, h2
