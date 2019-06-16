import collections
import io

import numpy

import chainer
from chainer.backends import cuda
import chainer.functions as F


def split_text(text, char_based=False):
    if char_based:
        return list(text)
    else:
        return text.split()


def normalize_text(text):
    return text.strip().lower()


def make_vocab(dataset, max_vocab_size=20000, min_freq=2):
    counts = collections.defaultdict(int)
    for tokens, _ in dataset:
        for token in tokens:
            counts[token] += 1

    vocab = {'<bos>': 0, '<eos>': 1, '<unk>': 2}
    for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if len(vocab) >= max_vocab_size or c < min_freq:
            break
        vocab[w] = len(vocab)
    return vocab


def read_vocab_list(path, max_vocab_size=20000):
    vocab = {'<bos>': 0, '<eos>': 1, '<unk>': 2}
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        for l in f:
            w = l.strip()
            if w not in vocab and w:
                vocab[w] = len(vocab)
            if len(vocab) >= max_vocab_size:
                break
    return vocab


def make_array(tokens, vocab, add_bos=True, add_eos=True):
    unk_id = vocab['<unk>']
    bos_id = vocab['<bos>']
    eos_id = vocab['<eos>']
    ids = [vocab.get(token, unk_id) for token in tokens]
    if add_bos:
        ids.insert(0, bos_id)
    if add_eos:
        ids.append(eos_id)
    return numpy.array(ids, numpy.int32)


def transform_to_array(dataset, vocab, with_label=True):
    if with_label:
        return [(make_array(tokens, vocab), numpy.array([cls], numpy.int32))
                for tokens, cls in dataset]
    else:
        return [make_array(tokens, vocab)
                for tokens in dataset]


def convert_seq(batch, device=None, with_label=True):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x)
                                     for x in batch[:-1]], dtype=numpy.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    if with_label:
        return {'xs': to_device_batch([x for x, _ in batch]),
                'ys': to_device_batch([y for _, y in batch])}
    else:
        return to_device_batch([x for x in batch])


def get_tops(output, beam_width):
    batch_size, num_vocab = output.shape
    xp = chainer.cuda.get_array_module(output)
    argsort = xp.argsort(output, axis=1)
    argtops = argsort[:, ::-1][:, :beam_width]
    assert (argtops.shape == (batch_size, beam_width)), (argtops.shape, (batch_size, beam_width))
    top_scores = xp.take(
        output,
        argtops + xp.arange(batch_size)[:, None] * num_vocab
    )
    return argtops, top_scores


def update_beam_state(tokens, total_score, tops, top_scores, vectors, eos):
    xp = chainer.cuda.get_array_module(vectors[0])
    full = tokens.shape[0]
    prev_full, k = top_scores.shape
    batch_size = full // k
    prev_k = prev_full // batch_size
    assert(prev_k in [1, k])
    if total_score is None:
        total_score = top_scores
    else:
        end = xp.broadcast_to(
            xp.max(tokens == eos, axis=1)[:, None],
            top_scores.shape
        )
        bias = xp.zeros(top_scores.shape, 'f')
        bias[:, 1:] = -10000.
        total_score = xp.where(
            end,
            total_score[:, None] + bias,
            total_score[:, None] + top_scores
        )
        #assert(xp.all(total_score < 0.))
        tops = xp.where(end, eos, tops)
    total_score = xp.reshape(
        total_score,
        (prev_full // prev_k, prev_k * k)
    )
    argtops, total_top_scores = get_tops(total_score, beam_width=k)
    assert(argtops.shape == (prev_full // prev_k, k))
    assert(total_top_scores.shape == (prev_full // prev_k, k))
    total_tops = xp.take(
        tops,
        argtops + xp.arange(prev_full // prev_k)[:, None] * prev_k * k
    )
    total_tops = xp.reshape(total_tops, (full, ))
    total_top_scores = xp.reshape(total_top_scores, (full, ))
    argtops = argtops // k + xp.arange(prev_full // prev_k)[:, None] * prev_k
    argtops = xp.reshape(argtops, (full, )).tolist()

    next_vs = []
    for v in vectors:
        vs = F.separate(v, axis=0)
        next_v = F.stack([vs[i] for i in argtops], axis=0)
        next_vs.append(next_v)
    '''
    hs = F.separate(h, axis=0)
    cs = F.separate(c, axis=0)
    next_h = F.stack([hs[i] for i in argtops], axis=0)
    next_c = F.stack([cs[i] for i in argtops], axis=0)
    '''
    tokens = xp.stack([tokens[i] for i in argtops], axis=0)
    tokens = xp.concatenate(
        [tokens, total_tops[:, None]],
        axis=1
    )
    return tokens.astype('i'), total_top_scores, next_vs


def finish_beam(tokens, total_score, batch_size, score=False, n_rand=1):
    k = tokens.shape[0] // batch_size
    if score:
        results = collections.defaultdict(
            lambda: {'tokens': [], 'score': -1e8}
        )
        for i in range(batch_size):
            for j in range(k):
                _score = total_score[i * k + j]
                if results[i]['score'] < _score:
                    token = tokens[i * k + j].tolist()
                    results[i] = {'tokens': token, 'score': _score}
        results = [
            result for i, result in sorted(results.items(), key=lambda x: x[0])
        ]
        return results
    else:
        xp = chainer.cuda.get_array_module(tokens)
        total_score = xp.reshape(total_score, (batch_size, k))
        #argtop = xp.argmax(total_score, axis=1) + xp.arange(batch_size) * k
        arg_sort = xp.argsort(-total_score, axis=1)
        arg_rand = arg_sort[xp.arange(batch_size), xp.random.randint(n_rand, size=batch_size)]
        arg_rand = arg_rand + xp.arange(batch_size) * k
        return tokens[arg_rand]
