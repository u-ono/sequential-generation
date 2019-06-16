import chainer
import tqdm


class KNNRetriever:

    def __init__(self, feature):
        self.pool = feature
        self.xp = chainer.cuda.get_array_module(feature)

    def __call__(self, batch, target_id, return_rank=False):
        xp = self.xp
        dist = -xp.dot(self.pool, batch.T)
        dist = xp.argsort(dist, axis=0)
        nearest_id = dist[0, :]
        nearest_id = chainer.cuda.to_cpu(nearest_id)

        if return_rank:
            rank = xp.where((xp.array(target_id) == dist).T)[1]
            rank = 1 - rank / len(self.pool)
            return nearest_id, rank
        else:
            return nearest_id


def compute_z(enc, xs, batchsize):

    xp = enc.xp
    iterator = chainer.iterators.SerialIterator(xs, batchsize, repeat=False, shuffle=False)

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        for i, batch in tqdm.tqdm(enumerate(iterator)):
            m, v = enc(xp.array(batch))
            if i == 0:
                ms = m.data
                vs = v.data
            else:
                ms = xp.vstack((ms, m.data))
                vs = xp.vstack((vs, v.data))

    return ms, vs


def compute_f(enc, xs, batchsize):

    xp = enc.xp
    iterator = chainer.iterators.SerialIterator(xs, batchsize, repeat=False, shuffle=False)

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        for i, batch in tqdm.tqdm(enumerate(iterator)):
            f = enc(xp.array(batch))
            if i == 0:
                fs = f.data
            else:
                fs = xp.vstack((fs, f.data))

    return fs
