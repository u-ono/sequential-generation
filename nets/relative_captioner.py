import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from src.miscs.nlp_utils import get_tops, update_beam_state, finish_beam


class RelativeCaptioner(chainer.Chain):

    def __init__(self, vocab, dim_feature=512, num_source=49, dim_source=512, dropout=0.5):
        self.vocab = vocab
        self.num_source = num_source
        self.dim_source = dim_source
        super(RelativeCaptioner, self).__init__()
        with self.init_scope():
            self.base_encoder1 = L.VGG16Layers()
            self.base_encoder2 = L.VGG16Layers()
            self.linear = L.Linear(2 * dim_feature, dim_source)
            self.image_encoder = ImageEncoder(
                num_source=num_source, dim_source=dim_source, dim_feature=dim_feature, dropout=dropout
            )
            self.text_decoder = TextDecoder(
                num_vocab=len(vocab), num_source=num_source, dim_source=dim_source, dropout=dropout
            )

    def encode(self, target_image, provided_image, raw=False):
        if not raw:
            target_feat = target_image
            provided_feat = provided_image
        else:
            target_image = chainer.cuda.to_cpu(target_image)
            provided_image = chainer.cuda.to_cpu(provided_image)
            target_feat = self.base_encoder1.extract(target_image, layers=['pool5'])['pool5']
            provided_feat = self.base_encoder2.extract(provided_image, layers=['pool5'])['pool5']
        v_local, keys = self.image_encoder.calc_local_feature(target_feat)
        v_global = F.relu(self.linear(F.concat(
            [F.mean(target_feat, axis=(2, 3)), F.mean(provided_feat, axis=(2, 3))]
        )))
        return v_global, v_local, keys

    def decode(self, caption, v_global, v_local, keys, num_word):
        y, alpha = self.text_decoder.forward(
            caption=caption, v_global=v_global, v_local=v_local, keys=keys, num_word=num_word
        )
        return y, alpha

    def reset_state(self):
        self.text_decoder.reset_state()

    def set_state(self, h, c):
        self.text_decoder.set_state(h, c)

    def get_state(self):
        return self.text_decoder.get_state()

    def forward(self, target, candid, **kwargs):

        beam = kwargs.get('beam', 10)
        max_len = kwargs.get('max_len', 12)
        raw = kwargs.get('raw', False)
        score = kwargs.get('score', False)
        keep = kwargs.get('keep', False)

        xp = self.xp
        bos = self.vocab['<bos>']
        eos = self.vocab['<eos>']
        batch_size = len(target)
        self.reset_state()
        v_global, v_local, keys = self.encode(target, candid, raw)
        if beam <= 1:  # greedy algorithm
            tokens = xp.array([[bos]] * batch_size, 'i')
            for i in range(max_len):
                x = Variable(tokens[:, i])
                y, _ = self.decode(x, v_global, v_local, keys, i)
                y_id = xp.argmax(y.data, axis=1)
                y_id = y_id[:, None].astype('int32')
                tokens = xp.hstack([tokens, y_id])
        else:  # beam search
            tokens = xp.array([[bos]] * batch_size * beam, 'i')
            x = Variable(xp.array([bos] * batch_size, 'i'))
            total_score = None
            h, c = None, None
            for i in range(max_len):
                self.set_state(h, c)
                k = len(x) * self.num_source // len(keys)
                v_local = F.reshape(
                    F.tile(
                        F.reshape(v_local, (-1, self.num_source, self.dim_source)),
                        (1, k, 1)
                    ),
                    (-1, self.dim_source)
                )
                keys = F.reshape(
                    F.tile(
                        F.reshape(keys, (-1, self.num_source, self.dim_source)),
                        (1, k, 1)
                    ),
                    (-1, self.dim_source)
                )
                y, _ = self.decode(x, v_global, v_local, keys, i)
                h, c = self.get_state()
                tops, top_scores = get_tops(F.log_softmax(y).data, beam)
                assert(xp.all(top_scores <= 0.))
                tokens, total_score, vectors = update_beam_state(
                    tokens, total_score, tops, top_scores, (h, c), eos
                )
                h, c = vectors
                assert(xp.all(top_scores <= 0.)), i
                x = Variable(tokens[:, -1])
            if not keep:
                tokens = finish_beam(tokens, total_score, batch_size, score)
        return tokens


class ImageEncoder(chainer.Chain):

    def __init__(self, num_source, dim_source, dim_feature, dropout):
        super(ImageEncoder, self).__init__()
        self.num_source = num_source
        self.dim_source = dim_source
        self.dim_feature = dim_feature
        self.dropout = dropout
        with self.init_scope():
            self.local_value_layer = L.Linear(dim_feature, dim_source)
            self.key_layer = L.Linear(dim_source, dim_source)

    def calc_local_feature(self, feature):
        # reshape the feature
        feat = F.reshape(
            F.transpose(feature, (0, 2, 3, 1)),
            (-1, self.dim_feature)
        )
        values = F.dropout(
            F.relu(self.local_value_layer(feat)),
            self.dropout
        )
        keys = self.key_layer(values)
        return values, keys


class TextDecoder(chainer.Chain):

    def __init__(self, num_vocab, num_source, dim_source, dropout):
        super(TextDecoder, self).__init__()
        self.dropout = dropout
        with self.init_scope():
            self.embed = L.EmbedID(num_vocab, dim_source)
            self.RNN = SentinelLSTM(dim_source, dim_source, dropout)
            self.attention = AdaptiveAttention(num_source, dim_source, dropout)
            self.linear1 = L.Linear(dim_source, dim_source)
            self.linear2 = L.Linear(dim_source, num_vocab)

    def forward(self, caption, v_global, v_local, keys, num_word):
        x = self.embed(caption)

        if num_word == 0:
            s, h = self.RNN(bos=x, vis=v_global)
        else:
            s, h = self.RNN(word=x)

        c, alpha = self.attention.forward(v_local, keys, h, s)
        h = F.tanh(self.linear1(h + c))
        h = self.linear2(F.dropout(h, ratio=self.dropout))
        alpha = 1
        return h, alpha

    def reset_state(self):
        self.RNN.reset_state()

    def set_state(self, h, c):
        self.RNN.set_state(h, c)

    def get_state(self):
        return self.RNN.get_state()


class AdaptiveAttention(chainer.Chain):

    def __init__(self, num_source, dim_source, dropout):
        super(AdaptiveAttention, self).__init__()
        self.num_source = num_source
        self.dim_source = dim_source
        self.dropout = dropout
        with self.init_scope():
            self.sentinel_layer = L.Linear(dim_source, dim_source)
            self.query_layer = L.Linear(dim_source, dim_source, nobias=True)
            self.linear_layer = L.Linear(dim_source, 1)

    def forward(self, values, keys, hidden, sentinel):
        values = F.reshape(values, (-1, self.num_source, self.dim_source))
        batch_size = values.shape[0]
        v_calib = F.concat([values, sentinel[:, None, :]], axis=1)

        s_key = self.sentinel_layer(sentinel)
        query = self.query_layer(hidden)
        beta = self.linear_layer(F.dropout(F.tanh(s_key + query), ratio=self.dropout))

        query = F.reshape(
            F.broadcast_to(query[:, None, :], (batch_size, self.num_source, self.dim_source)),
            (-1, self.dim_source)
        )
        z = F.reshape(
            self.linear_layer(F.dropout(F.tanh(keys + query), self.dropout)),
            (-1, self.num_source)
        )
        alpha = F.softmax(F.concat([z, beta]))
        context = F.sum(
            F.broadcast_to(alpha[:, :, None], (batch_size, self.num_source + 1, self.dim_source)) * v_calib,
            axis=1
        )
        return context, alpha


class SentinelLSTM(chainer.Chain):

    def __init__(self, input_size, output_size, dropout):
        super(SentinelLSTM, self).__init__()
        self.dropout = dropout
        with self.init_scope():
            self.vis_receiver = L.Linear(input_size, 4 * output_size)
            self.bos_receiver = L.Linear(input_size, 4 * output_size, nobias=True)
            self.input_layer = L.Linear(input_size, output_size)
            self.hidden_layer = L.Linear(input_size, output_size)
            self.RNN = L.LSTM(input_size, output_size)

    def reset_state(self):
        self.RNN.reset_state()

    def set_state(self, h, c):
        self.RNN.h = h
        self.RNN.c = c

    def get_state(self):
        return self.RNN.h, self.RNN.c

    def __call__(self, bos=None, vis=None, word=None):
        if bos is not None:
            h = self.vis_receiver(vis) + self.bos_receiver(bos)
            a, i, o, g = F.split_axis(h, 4, axis=1)
            a = F.tanh(a)
            i = F.sigmoid(i)
            o = F.sigmoid(o)
            g = F.sigmoid(g)
            c = a * i
            h = F.dropout(o * F.tanh(c), ratio=self.dropout)
            self.RNN.set_state(c, h)
        else:
            word = F.dropout(word, ratio=self.dropout)
            g = F.sigmoid(self.input_layer(word) + self.hidden_layer(self.RNN.h))
            h = F.dropout(self.RNN(word), ratio=self.dropout)
        s = F.dropout(g * F.tanh(self.RNN.c), ratio=self.dropout)
        return s, h
