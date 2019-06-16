import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from src.miscs.nlp_utils import get_tops, update_beam_state, finish_beam


class ESSpeaker(chainer.Chain):

    def __init__(self, *args, **kwargs):

        super(ESSpeaker, self).__init__()

        with self.init_scope():
            self.emitter = Speaker(*args, **kwargs)
            self.suppressor = Speaker(*args, **kwargs)

        self.vocab = self.emitter.vocab
        self.num_source = self.emitter.num_source
        self.dim_source = self.emitter.dim_source

    def __call__(self, target, candid, **kwargs):

        beam = kwargs.get('beam', 10)
        lam = kwargs.get('lam', 0.4)
        max_len = kwargs.get('max_len', 12)
        raw_target = kwargs.get('raw_target', False)
        raw_candid = kwargs.get('raw_candid', False)
        keep = kwargs.get('keep', False)
        n_rand = kwargs.get('n_rand', 1)
        r_eos = kwargs.get('r_eos', 1.)
        r_dup = kwargs.get('r_dup', 1.)

        xp = self.xp
        bos = self.vocab['<bos>']
        eos = self.vocab['<eos>']
        batch_size = len(target)

        tokens = xp.array([[bos]] * batch_size * beam, 'i')
        x = Variable(xp.array([bos] * batch_size, 'i'))

        vg_e, vl_e, k_e = self.emitter.encode(target, raw=raw_target)
        vg_s, vl_s, k_s = self.suppressor.encode(candid, raw=raw_candid)

        total_score = None

        h_e, c_e, h_s, c_s = None, None, None, None

        for i in range(max_len):

            k = len(x) * self.num_source // len(k_e)

            vl_e = self.tile(vl_e, k)
            vl_s = self.tile(vl_s, k)
            k_e = self.tile(k_e, k)
            k_s = self.tile(k_s, k)

            self.emitter.set_state(h_e, c_e)
            y_e, _ = self.emitter.decode(x, vg_e, vl_e, k_e, i)
            h_e, c_e = self.emitter.get_state()

            self.suppressor.set_state(h_s, c_s)
            y_s, _ = self.suppressor.decode(x, vg_s, vl_s, k_s, i)
            h_s, c_s = self.suppressor.get_state()

            rate = xp.ones(y_e.shape, 'f')
            if i > 0:
                rate[:, eos] = r_eos
                row = xp.broadcast_to(xp.arange(len(y_e))[:, None], tokens.shape)
                rate[row, tokens] = r_dup

            log_p = F.log_softmax(y_e * rate) - (1 - lam) * F.log_softmax(y_s)

            top_k, top_k_score = get_tops(log_p.data, beam)
            vectors = [h_e, c_e, h_s, c_s]
            tokens, total_score, vectors = update_beam_state(
                tokens=tokens, total_score=total_score, tops=top_k, top_scores=top_k_score, vectors=vectors, eos=eos
            )
            h_e, c_e, h_s, c_s = vectors
            x = Variable(tokens[:, -1])

        if not keep:
            tokens = finish_beam(tokens, total_score, batch_size, n_rand=n_rand)

        return tokens

    def tile(self, v, k):

        v = F.reshape(
            F.tile(
                F.reshape(v, (-1, self.num_source, self.dim_source)),
                (1, k, 1)
            ),
            (-1, self.dim_source)
        )

        return v


class Speaker(chainer.Chain):

    def __init__(self, vocab, dim_feature=512, num_source=49, dim_source=512, dropout=0.5):
        self.vocab = vocab
        self.num_source = num_source
        self.dim_source = dim_source
        super(Speaker, self).__init__()
        with self.init_scope():
            self.base_encoder = L.VGG16Layers()
            self.linear = L.Linear(2 * dim_feature, dim_source)
            self.image_encoder = ImageEncoder(
                num_source=num_source, dim_source=dim_source, dim_feature=dim_feature, dropout=dropout
            )
            self.text_decoder = TextDecoder(
                num_vocab=len(vocab), num_source=num_source, dim_source=dim_source, dropout=dropout
            )

    def encode(self, image, raw=False):
        if not raw:
            feat = image
        else:
            image = [chainer.cuda.to_cpu(i) for i in image]
            feat = self.base_encoder.extract(image, layers=['pool5'])['pool5']
        v_local, keys = self.image_encoder.calc_local_feature(feat)
        v_global = self.image_encoder.calc_global_feature(feat)
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

    def __call__(self, image, beam=5, max_len=12, raw=False, score=False, keep=False):
        xp = self.xp
        bos = self.vocab['<bos>']
        eos = self.vocab['<eos>']
        batch_size = len(image)
        self.reset_state()
        v_global, v_local, keys = self.encode(image, raw)
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
                vectors = (h, c)
                tokens, total_score, vectors = update_beam_state(
                    tokens=tokens, total_score=total_score, tops=tops, top_scores=top_scores, vectors=vectors, eos=eos
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
            self.global_value_layer = L.Linear(dim_feature, dim_source)
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

    def calc_global_feature(self, feature):
        # reshape the feature from (|B|, dim_feature, h, w)
        feat = F.reshape(
            F.transpose(feature, (0, 2, 3, 1)),
            (-1, self.num_source, self.dim_feature)
        )
        # calculate the mean
        feat = F.mean(feat, axis=1)
        # calculate the global value
        value = F.dropout(
            F.relu(self.global_value_layer(feat)),
            self.dropout
        )
        return value


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


