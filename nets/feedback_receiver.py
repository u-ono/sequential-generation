import chainer
import chainer.links as L
import chainer.functions as F


class FeedbackReceiver(chainer.Chain):

    def __init__(self, dim_z, num_voc, dim_emb, dim_txt, distribution='normal', init_emb=None):
        self.distribution = distribution
        super(FeedbackReceiver, self).__init__()
        with self.init_scope():
            self.txt_enc = TextEncoder(num_voc, dim_emb, dim_txt, init_emb)
            self.lin_1 = L.Linear(dim_z + dim_txt, dim_z)
            self.GRU = L.GRU(dim_z, dim_z)
            self.lin_2 = L.Linear(dim_z, dim_z)
            if distribution == 'normal':
                self.lin_m = L.Linear(dim_z, dim_z)
                self.lin_v = L.Linear(dim_z, dim_z)
            elif distribution == 'vmf':
                self.lin_m = L.Linear(dim_z, dim_z)
                self.lin_v = L.Linear(dim_z, 1)
            else:
                raise NotImplementedError

    def __call__(self, z, t):
        t_emb = F.normalize(self.txt_enc(t))
        z_emb = F.normalize(z)
        h = F.concat([z_emb, t_emb], axis=1)
        h = F.relu(self.lin_1(h))
        h = F.relu(self.GRU(h))
        h = F.relu(self.lin_2(h))
        if self.distribution == 'normal':
            m = self.lin_m(h)
            v = F.softplus(self.lin_v(h))
        elif self.distribution == 'vmf':
            m = F.normalize(self.lin_m(h), axis=-1)
            v = F.softplus(self.lin_v(h)) + 1
        else:
            raise NotImplementedError
        return m, v

    def reset_state(self):
        self.GRU.reset_state()


class TextEncoder(chainer.Chain):

    def __init__(self, num_voc, dim_emb, dim_enc, embed_init=None):
        super(TextEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(num_voc, dim_emb, ignore_label=-1, initialW=embed_init)
            self.CNN3Gram = L.Convolution2D(dim_emb, dim_emb//3, ksize=(3, 1), stride=1, pad=(2, 0), nobias=True)
            self.CNN4Gram = L.Convolution2D(dim_emb, dim_emb//3, ksize=(4, 1), stride=1, pad=(3, 0), nobias=True)
            self.CNN5Gram = L.Convolution2D(dim_emb, dim_emb//3, ksize=(5, 1), stride=1, pad=(4, 0), nobias=True)
            self.l1 = L.Linear(None, dim_enc)
            self.l2 = L.Linear(dim_enc, dim_enc)

    def __call__(self, text):
        emb = self.embed(text)
        emb = F.transpose(emb, (0, 2, 1))
        emb = emb[:, :, :, None]
        h3 = F.max(self.CNN3Gram(emb), axis=2)
        h4 = F.max(self.CNN4Gram(emb), axis=2)
        h5 = F.max(self.CNN5Gram(emb), axis=2)
        h = F.concat([h3, h4, h5], axis=1)
        h = F.relu(self.l1(h))
        return self.l2(h)
