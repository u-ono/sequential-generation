import chainer
import chainer.links as L
import chainer.functions as F


class Extractor(chainer.Chain):

    def __init__(self):
        super(Extractor, self).__init__()
        with self.init_scope():
            self.resnet = L.ResNet50Layers()

    def __call__(self, x):

        x = self.convert(x)
        f = self.resnet(x, layers=['pool5'])['pool5']

        return f

    def convert(self, x):

        x = (x + 1) * 255. / 2.
        x = F.transpose(x, (0, 2, 3, 1))
        x = x[:, :, :, ::-1]
        x -= self.xp.array([103.063, 115.903, 123.152], 'f')
        x = F.transpose(x, (0, 3, 1, 2))

        return x
