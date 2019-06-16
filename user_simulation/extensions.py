import chainer
import chainer.functions as F
from chainer import Variable
from chainer.training import extensions
from chainer import reporter as reporter_module
import copy
import numpy as np
from PIL import Image
import os
from itertools import product


class VanillaEvaluator(extensions.Evaluator):

    default_name = 'val'

    def __init__(self, *args, **kwargs):

        self.model = kwargs.pop('model')
        self.vocab = kwargs.pop('vocab')
        self.raw = kwargs.pop('raw')

        super(VanillaEvaluator, self).__init__(*args, **kwargs, target=self.model)

    def evaluate(self):

        xp = self.model.xp

        summary = reporter_module.DictSummary()

        iterator = self.get_iterator('main')

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        for batch in it:

            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

                images, captions = self.converter(batch, device=self.device, padding=self.vocab['<eos>'])
                len_caption = captions.shape[1]

                # Compute target loss

                self.model.reset_state()
                vg_t, vl_t, k_t = self.model.encode(images, raw=self.raw)

                loss = 0
                acc = 0
                size = 0

                for i in range(len_caption-1):
                    subject = (captions[:, i] != self.vocab['<eos>']).astype('float32')

                    if (subject == 0).all():
                        break

                    x = Variable(xp.asarray(captions[:, i]))
                    t = Variable(xp.asarray(captions[:, i+1]))

                    y, _ = self.model.decode(x, vg_t, vl_t, k_t, i)
                    y_id = xp.argmax(y.data, axis=1)
                    mask = F.broadcast_to(subject[:, None], y.data.shape)
                    y = y * mask

                    loss += F.softmax_cross_entropy(y, t)
                    acc += xp.sum((y_id == t.data) * subject)
                    size += xp.sum(subject)

                loss = loss * len(batch) / size
                acc /= size

            summary.add({VanillaEvaluator.default_name + '/loss': loss})
            summary.add({VanillaEvaluator.default_name + '/acc': acc})

        return summary.compute_mean()


class RelativeEvaluator(extensions.Evaluator):

    default_name = 'val'

    def __init__(self, *args, **kwargs):

        self.model = kwargs.pop('model')
        self.vocab = kwargs.pop('vocab')
        self.raw = kwargs.pop('raw')

        super(RelativeEvaluator, self).__init__(*args, **kwargs, target=self.model)

    def evaluate(self):

        xp = self.model.xp

        summary = reporter_module.DictSummary()

        iterator = self.get_iterator('main')

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        for batch in it:

            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

                target, candid, captions = self.converter(batch, device=self.device, padding=self.vocab['<eos>'])
                len_caption = captions.shape[1]

                self.model.reset_state()
                v_global, v_local, keys = self.model.encode(target, candid, raw=self.raw)

                loss = 0
                acc = 0
                size = 0

                for i in range(len_caption-1):
                    subject = (captions[:, i] != self.vocab['<eos>']).astype('float32')

                    if (subject == 0).all():
                        break

                    x = Variable(xp.asarray(captions[:, i]))
                    t = Variable(xp.asarray(captions[:, i+1]))

                    y, _ = self.model.decode(x, v_global, v_local, keys, i)
                    y_id = xp.argmax(y.data, axis=1)
                    mask = F.broadcast_to(subject[:, None], y.data.shape)
                    y = y * mask

                    loss += F.softmax_cross_entropy(y, t)
                    acc += xp.sum((y_id == t.data) * subject)
                    size += xp.sum(subject)

                loss = loss * len(batch) / size
                acc /= size

            summary.add({VanillaEvaluator.default_name + '/loss': loss})
            summary.add({VanillaEvaluator.default_name + '/acc': acc})

        return summary.compute_mean()


class RelativeMMIEvaluator(extensions.Evaluator):

    default_name = 'val'

    def __init__(self, *args, **kwargs):

        self.model = kwargs.pop('model')
        self.vocab = kwargs.pop('vocab')
        self.raw = kwargs.pop('raw')
        self.margin = kwargs.pop('margin')
        self.lam = kwargs.pop('lam')

        super(RelativeMMIEvaluator, self).__init__(*args, **kwargs, target=self.model)

    def evaluate(self):

        xp = self.model.xp

        summary = reporter_module.DictSummary()

        iterator = self.get_iterator('main')

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        for batch in it:

            target, candid, captions = self.converter(batch, device=self.device, padding=self.vocab['<eos>'])
            len_caption = captions.shape[1]

            # Compute target loss

            self.model.reset_state()
            v_global, v_local, keys = self.model.encode(target, candid, raw=self.raw)

            loss_target = 0
            acc = 0
            size = 0

            for i in range(len_caption-1):
                subject = (captions[:, i] != self.vocab['<eos>']).astype('float32')

                if (subject == 0).all():
                    break

                x = Variable(xp.asarray(captions[:, i]))
                t = Variable(xp.asarray(captions[:, i+1]))

                y, _ = self.model.decode(x, v_global, v_local, keys, i)
                y_id = xp.argmax(y.data, axis=1)
                mask = F.broadcast_to(subject[:, None], y.data.shape)
                y = y * mask

                loss_target += F.softmax_cross_entropy(y, t)
                acc += xp.sum((y_id == t.data) * subject)
                size += xp.sum(subject)

            loss_target = loss_target * len(batch) / size
            acc /= size

            # Compute candidate caption likelihood

            self.model.reset_state()
            v_global, v_local, keys = self.model.encode(candid, target, raw=self.raw)

            loss_candid = 0
            size = 0

            for i in range(len_caption-1):
                subject = (captions[:, i] != self.vocab['<eos>']).astype('float32')

                if (subject == 0).all():
                    break

                x = Variable(xp.asarray(captions[:, i]))
                t = Variable(xp.asarray(captions[:, i+1]))

                y, _ = self.model.decode(x, v_global, v_local, keys, i)
                mask = F.broadcast_to(subject[:, None], y.data.shape)
                y = y * mask

                loss_candid += F.softmax_cross_entropy(y, t)
                size += xp.sum(subject)

            loss_candid = loss_candid * len(batch) / size

            # Max Margin Loss

            loss_max_margin = F.maximum(
                self.margin + loss_target - loss_candid,
                Variable(xp.zeros_like(loss_target))
            )

            loss = loss_target + self.lam * loss_max_margin

            summary.add({VanillaEvaluator.default_name + '/loss/target': loss_target})
            summary.add({VanillaEvaluator.default_name + '/loss/total': loss})
            summary.add({VanillaEvaluator.default_name + '/acc': acc})

        return summary.compute_mean()


def visualize(model, images, out_path, args):

    @chainer.training.make_extension()
    def extension(trainer):

        preview_dir = '{}/preview/'.format(out_path)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        vocab = model.vocab
        id_to_word = {}
        for k, v in vocab.items():
            id_to_word[v] = k

        num_data = len(images)
        num_case = args.row * args.col

        np.random.seed(args.seed)
        target_index = np.random.choice(num_data, num_case, replace=False)
        candid_index = np.random.choice(num_data, num_case, replace=False)
        np.random.seed()

        target = [images.get_example(i) for i in target_index]
        candid = [images.get_example(i) for i in candid_index]

        if args.model == 'relative':

            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                text = model(target, candid, beam_width=args.beam, raw=True)
                text = chainer.cuda.to_cpu(text)

            s = ''
            for case in range(num_case):
                s += '<case {}>\n'.format(case)
                tokens = [id_to_word[token] for token in text[case]]
                while '<bos>' in tokens:
                    tokens.remove('<bos>')
                while '<eos>' in tokens:
                    tokens.remove('<eos>')
                s += ' '.join(tokens) + '\n'

            preview_path = preview_dir + '{:0>3}.txt'.format(trainer.updater.epoch)
            with open(preview_path, 'w') as file:
                file.write(s)

        elif args.model == 'es':
            from nets.es_speaker import ESSpeaker

            sim = ESSpeaker(vocab=vocab)
            sim.emitter = model
            sim.suppressor = model

            if args.gpu >= 0:
                sim.to_gpu()

            for r_eos, r_dup in product([1.0, 1.2, 1.5], [1.0, 0.8, 0.6]):

                texts = []
                lams = [0.2, 0.4, 0.6, 0.8, 1.0]
                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    for lam in lams:
                        out = sim(
                            target,
                            candid,
                            beam=args.beam,
                            lam=lam,
                            r_eos=r_eos,
                            r_dup=r_dup,
                            raw_target=True,
                            raw_candid=True,
                            n_rand=1
                        )
                        texts.append(chainer.cuda.to_cpu(out))

                    texts = np.array(texts).transpose((1, 0, 2))

                    s = ''
                    for case in range(num_case):
                        s += '<case {}>\n'.format(case)
                        for i, text in enumerate(texts[case]):
                            s += 'lam={}: '.format(lams[i])
                            tokens = [id_to_word[token] for token in text]
                            while '<bos>' in tokens:
                                tokens.remove('<bos>')
                            while '<eos>' in tokens:
                                tokens.remove('<eos>')
                            s += ' '.join(tokens) + '\n'

                    preview_path = preview_dir + '{:0>3}_eos_{}_dup_{}.txt'.format(trainer.updater.epoch, r_eos, r_dup)
                    with open(preview_path, 'w') as file:
                        file.write(s)

        else:

            raise NotImplementedError

        target = np.array(target).reshape(args.row, args.col, 64, 64, 3).transpose(0, 2, 1, 3, 4).reshape(args.row * 64, args.col * 64, 3)
        candid = np.array(candid).reshape(args.row, args.col, 64, 64, 3).transpose(0, 2, 1, 3, 4).reshape(args.row * 64, args.col * 64, 3)

        preview_path = preview_dir + 'target.jpg'.format(trainer.updater.epoch)
        Image.fromarray(target).save(preview_path)

        preview_path = preview_dir + 'candid.jpg'.format(trainer.updater.epoch)
        Image.fromarray(candid).save(preview_path)

    return extension
