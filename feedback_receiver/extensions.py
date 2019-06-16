import chainer
import chainer.functions as F
from chainer import Variable
from chainer import reporter as reporter_module
from chainer.training import extensions
import numpy as np
import os
from PIL import Image
import csv
from src.miscs.knn_retriever import KNNRetriever
from src.functions.vmf import von_mises_fisher, spherical_kl_divergence
from src.miscs.array_utils import denormalize


class _StateTrackerEvaluator(extensions.Evaluator):

    default_name = 'val'

    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.data = kwargs.pop('data')
        self.args = kwargs.pop('args')
        super(_StateTrackerEvaluator, self).__init__(*args, **kwargs, target=self.models[0])

    def evaluate(self):

        args = self.args
        data = self.data

        fbr, sim = self.models
        xp = fbr.xp

        # Check how many times to iterate
        batch = self.get_iterator('main').next()
        num_batch = len(batch)
        num_data = len(data['name'])
        iteration = num_data // num_batch

        summary = reporter_module.DictSummary()

        for i in range(iteration):

            # Define indices for training
            target_index = np.random.choice(num_data, num_batch, replace=False)
            candid_index = np.random.choice(num_data, num_batch, replace=False)

            # Define retrieval model
            image_retriever = KNNRetriever(xp.array(data['m']))

            # Define the value holders
            ranks = []
            loss_triplet = 0
            loss_kl = 0

            # Start training loop
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

                fbr.reset_state()

                for t in range(args.turn):
                    # User feedback to the comparison image
                    text = sim(
                        Variable(self.converter(data['f'][target_index], self.device)),
                        Variable(self.converter(data['f'][candid_index], self.device)),
                        beam=args.beam,
                        lam=args.lam,
                        bias=args.bias,
                        n_rand=args.n_rand
                    )
                    z = Variable(self.converter(data['m'][candid_index], self.device))

                    # Transform z
                    mu, var = fbr(z, text)

                    # Compute KL divergence & sample z
                    if args.distribution == 'normal':
                        ln_var = F.log(var + 1e-10)
                        z = F.gaussian(mu, ln_var)
                        loss_kl += F.gaussian_kl_divergence(mu, ln_var) / mu.data.size
                    elif args.distribution == 'vmf':
                        z = von_mises_fisher(mu, var)
                        loss_kl += F.sum(spherical_kl_divergence(mu, var)) / mu.data.size
                    else:
                        raise NotImplementedError

                    # Calculate triplet loss
                    random_index = np.random.choice(num_data, num_batch, replace=False)
                    loss_triplet += F.triplet(
                        anchor=z,
                        positive=self.converter(data['m'][target_index], self.device),
                        negative=self.converter(data['m'][random_index], self.device),
                        margin=args.margin
                    )

                    # Retrieve the nearest image by index, and return percentile rank
                    candid_index, rank = image_retriever(z.data, target_index, return_rank=True)
                    ranks.append(chainer.cuda.to_cpu(rank))

            loss = loss_triplet + args.c_kld * loss_kl

            summary.add({_StateTrackerEvaluator.default_name + '/loss/total': loss})
            summary.add({_StateTrackerEvaluator.default_name + '/loss/kl': loss_kl})
            summary.add({_StateTrackerEvaluator.default_name + '/loss/triplet': loss_triplet})

            ranks = np.mean(np.array(ranks), axis=1)
            for t in range(args.turn):
                summary.add({_StateTrackerEvaluator.default_name + '/rank{}'.format(t+1): ranks[t]})

        return summary.compute_mean()


class FeedbackReceiverEvaluator(extensions.Evaluator):

    default_name = 'val'

    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.data = kwargs.pop('data')
        self.args = kwargs.pop('args')
        super(FeedbackReceiverEvaluator, self).__init__(*args, **kwargs, target=self.models[0])

    def evaluate(self):

        args = self.args
        data = self.data

        fbr, gen, sim = self.models
        xp = fbr.xp

        # Check how many times to iterate
        batch = self.get_iterator('main').next()
        num_batch = len(batch)
        num_data = len(data['name'])
        iteration = num_data // num_batch

        summary = reporter_module.DictSummary()

        for i in range(iteration):

            # Define indices for training
            target_index = np.random.choice(num_data, num_batch, replace=False)
            z_target = self.converter(data['m'][target_index], self.device)

            # Define candid index
            '''
            candid_index = np.random.choice(num_data, num_batch, replace=False)
            z_candid = self.converter(data['m'][candid_index], self.device)
            x_candid = self.converter(data['x'][candid_index], self.device)
            '''
            z_candid = self.converter(np.zeros(z_target.shape, 'f'), self.device)
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                x_candid = gen(z_candid).data

            # Define retrieval model
            knn = KNNRetriever(xp.array(data['m']))

            # Define the value holders
            ranks = []
            loss_triplet = 0
            loss_kl = 0

            # Start training loop
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

                fbr.reset_state()

                for t in range(args.turn):

                    lam = 1.0 if t == 0 else args.lam

                    # User feedback to the comparison image
                    text = sim(
                        target=Variable(self.converter(data['f'][target_index], self.device)),
                        candid=[denormalize(x) for x in x_candid],
                        beam=args.beam,
                        lam=lam,
                        r_eos=args.r_eos,
                        r_dup=args.r_dup,
                        n_rand=args.n_rand,
                        raw_candid=True
                    )

                    # Transform z
                    mu, var = fbr(z_candid, text)

                    if not args.no_kld:
                        # Compute KL divergence & sample z
                        if args.distribution == 'normal':
                            ln_var = F.log(var + 1e-10)
                            z_candid = F.gaussian(mu, ln_var)
                            loss_kl += F.gaussian_kl_divergence(mu, ln_var) / mu.data.size
                        elif args.distribution == 'vmf':
                            z_candid = von_mises_fisher(mu, var)
                            loss_kl += F.sum(spherical_kl_divergence(mu, var)) / mu.data.size
                        else:
                            raise NotImplementedError
                    else:
                        z_candid = mu

                    # Calculate triplet loss
                    random_index = np.random.choice(num_data, num_batch, replace=False)
                    z_random = self.converter(data['m'][random_index], self.device)

                    loss_triplet += F.triplet(
                        anchor=z_candid,
                        positive=z_target,
                        negative=z_random,
                        margin=args.margin
                    )

                    # Retrieve the nearest image by index, and return percentile rank
                    _, rank = knn(z_candid.data, target_index, return_rank=True)
                    x_candid = gen(z_candid).data

                    ranks.append(chainer.cuda.to_cpu(rank))

            loss = loss_triplet + args.c_kld * loss_kl

            summary.add({FeedbackReceiverEvaluator.default_name + '/loss/total': loss})
            summary.add({FeedbackReceiverEvaluator.default_name + '/loss/kl': loss_kl})
            summary.add({FeedbackReceiverEvaluator.default_name + '/loss/triplet': loss_triplet})

            ranks = np.mean(np.array(ranks), axis=1)
            for t in range(args.turn):
                summary.add({FeedbackReceiverEvaluator.default_name + '/rank{}'.format(t+1): ranks[t]})

        return summary.compute_mean()


class RetrievalEvaluator(extensions.Evaluator):

    default_name = 'val'

    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.data = kwargs.pop('data')
        self.args = kwargs.pop('args')
        super(RetrievalEvaluator, self).__init__(*args, **kwargs, target=self.models[0])

    def evaluate(self):

        args = self.args
        data = self.data

        fbr, sim = self.models
        xp = fbr.xp

        # Check how many times to iterate
        batch = self.get_iterator('main').next()
        num_batch = len(batch)
        num_data = len(data['name'])
        iteration = num_data // num_batch

        summary = reporter_module.DictSummary()

        for i in range(iteration):

            # Define indices for training
            target_index = np.random.choice(num_data, num_batch, replace=False)
            z_target = self.converter(data['m'][target_index], self.device)

            candid_index = np.random.choice(num_data, num_batch, replace=False)
            z_candid = self.converter(data['m'][candid_index], self.device)

            # Define retrieval model
            image_retriever = KNNRetriever(xp.array(data['m']))

            # Define the value holders
            ranks = []
            loss_triplet = 0
            loss_kl = 0

            # Start training loop
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

                fbr.reset_state()

                for t in range(args.turn):
                    # User feedback to the comparison image
                    text = sim(
                        Variable(self.converter(data['f'][target_index], self.device)),
                        Variable(self.converter(data['f'][candid_index], self.device)),
                        beam=args.beam,
                        lam=args.lam,
                        r_eos=args.r_eos,
                        r_dup=args.r_dup,
                        n_rand=args.n_rand
                    )

                    # Transform z
                    mu, var = fbr(z_candid, text)

                    # Compute KL divergence & sample z
                    if not args.no_kld:
                        if args.distribution == 'normal':
                            ln_var = F.log(var + 1e-10)
                            z_candid = F.gaussian(mu, ln_var)
                            loss_kl += F.gaussian_kl_divergence(mu, ln_var) / mu.data.size
                        elif args.distribution == 'vmf':
                            z_candid = von_mises_fisher(mu, var)
                            loss_kl += F.sum(spherical_kl_divergence(mu, var)) / mu.data.size
                        else:
                            raise NotImplementedError
                    else:
                        z_candid = mu

                    # Calculate triplet loss
                    random_index = np.random.choice(num_data, num_batch, replace=False)
                    z_random = self.converter(data['m'][random_index], self.device)

                    loss_triplet += F.triplet(
                        anchor=z_candid,
                        positive=z_target,
                        negative=z_random,
                        margin=args.margin
                    )

                    # Retrieve the nearest image by index, and return percentile rank
                    candid_index, rank = image_retriever(z_candid.data, target_index, return_rank=True)
                    z_candid = Variable(self.converter(data['m'][candid_index], self.device))

                    ranks.append(chainer.cuda.to_cpu(rank))

            loss = loss_triplet + args.c_kld * loss_kl

            summary.add({FeedbackReceiverEvaluator.default_name + '/loss/total': loss})
            summary.add({FeedbackReceiverEvaluator.default_name + '/loss/kl': loss_kl})
            summary.add({FeedbackReceiverEvaluator.default_name + '/loss/triplet': loss_triplet})

            ranks = np.mean(np.array(ranks), axis=1)
            for t in range(args.turn):
                summary.add({FeedbackReceiverEvaluator.default_name + '/rank{}'.format(t+1): ranks[t]})

        return summary.compute_mean()


def visualize_retrieval(models, data, args, out, seed=0):

    @chainer.training.make_extension()
    def evaluate(trainer):

        device = trainer.updater.device
        converter = trainer.updater.converter

        fbr, sim, gen = models
        num_data = len(data['name'])

        # Set vocabulary and reverse dict
        vocab = sim.vocab
        id_to_word = {}
        for k, v in vocab.items():
            id_to_word[v] = k

        # Define indices for training
        np.random.seed(seed)
        target_index = np.random.choice(num_data, args.n_case, replace=False)
        candid_index = np.random.choice(num_data, args.n_case, replace=False)
        np.random.seed()

        z_candid = converter(data['m'][candid_index], device)

        # Define retrieval model
        xp = fbr.xp
        knn = KNNRetriever(xp.array(data['m']))

        # Define the value holders
        text_hists = []
        rank_hists = []
        index_hists = [candid_index]
        x_gen = []

        # Start the training loop
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

            fbr.reset_state()

            for t in range(args.turn):
                # User feedback to the comparison image
                text = sim(
                    Variable(converter(data['f'][target_index], device)),
                    Variable(converter(data['f'][candid_index], device)),
                    beam=args.beam,
                    lam=args.lam,
                    r_eos=args.r_eos,
                    r_dup=args.r_dup,
                    n_rand=args.n_rand
                )

                mu, var = fbr(z_candid, text)

                # sample z
                if not args.no_kld:
                    if args.distribution == 'normal':
                        ln_var = F.log(var + 1e-10)
                        z_candid = F.gaussian(mu, ln_var)
                    elif args.distribution == 'vmf':
                        z_candid = von_mises_fisher(mu, var)
                    else:
                        raise NotImplementedError
                else:
                    z_candid = mu

                # Retrieve the nearest image by index, and return percentile rank
                candid_index, rank = knn(z_candid.data, target_index, return_rank=True)
                z_candid = Variable(converter(data['m'][candid_index], device))

                x = gen(z_candid)

                x_gen.append(chainer.cuda.to_cpu(x.data))
                text_hists.append(chainer.cuda.to_cpu(text))
                rank_hists.append(chainer.cuda.to_cpu(rank))
                index_hists.append(candid_index)

        preview_dir = '{}/preview/'.format(out)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        # history of generated image
        x_gen = np.clip((np.array(x_gen)+1) * 255 / 2., 0, 255).astype(np.uint8)
        x_gen = np.reshape(
            np.transpose(x_gen, [1, 3, 0, 4, 2]),
            (args.n_case * 64, args.turn * 64, 3)
        )
        preview_path = preview_dir + 'gen_{:0>3}.png'.format(trainer.updater.epoch)
        Image.fromarray(x_gen).save(preview_path)

        # history image
        index_hists.append(target_index)
        index_hists = np.array(index_hists)
        idx = np.transpose(index_hists).flatten()
        x_knn = data['x'][idx]
        x_knn = np.clip((np.array(x_knn)+1) * 255 / 2., 0, 255).astype(np.uint8)
        _, _, h, w = x_knn.shape
        x_knn = np.reshape(x_knn, (args.n_case, args.turn + 2, 3, h, w))
        x_knn = np.reshape(
            np.transpose(x_knn, [0, 3, 1, 4, 2]),
            (args.n_case * h, (args.turn + 2) * w, 3)
        )
        preview_path = preview_dir + 'knn_{:0>3}.png'.format(trainer.updater.epoch)
        Image.fromarray(x_knn).save(preview_path)

        # history caption
        text_hists = np.array(text_hists)
        text_hists = np.transpose(text_hists, [1, 0, 2]).tolist()
        s = ''
        for case in range(len(text_hists)):
            s += '<case {}>\n'.format(case)
            for t, text in enumerate(text_hists[case]):
                s += str(t + 1) + ': '
                text = [id_to_word[token] for token in text]
                while '<bos>' in text:
                    text.remove('<bos>')
                while '<eos>' in text:
                    text.remove('<eos>')
                s += ' '.join(text) + '\n'
        preview_path = preview_dir + 'text_{:0>3}.txt'.format(trainer.updater.epoch)
        with open(preview_path, 'w') as file:
            file.write(s)

        # history rank
        rank_hists = np.transpose(np.array(rank_hists))
        preview_path = preview_dir + 'rank_{:0>3}.csv'.format(trainer.updater.epoch)
        with open(preview_path, 'w') as file:
            writer = csv.writer(file)
            header = ['turn{}'.format(t + 1) for t in range(args.turn)]
            writer.writerow(header)
            writer.writerows(rank_hists.tolist())

    return evaluate


def visualize(models, data, args, out, seed=0):

    @chainer.training.make_extension()
    def evaluate(trainer):

        device = trainer.updater.device
        converter = trainer.updater.converter

        fbr, sim, gen = models
        num_data = len(data['name'])

        # Set vocabulary and reverse dict
        vocab = sim.vocab
        id_to_word = {}
        for k, v in vocab.items():
            id_to_word[v] = k

        # Define indices for training
        np.random.seed(seed)
        target_index = np.random.choice(num_data, args.n_case, replace=False)
        # candid_index = np.random.choice(num_data, args.n_case, replace=False)
        np.random.seed()

        z_target = Variable(converter(data['m'][target_index], device))
        # z_candid = Variable(converter(data['m'][candid_index], device))
        z_candid = converter(np.zeros(z_target.shape, 'f'), device)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x_candid = gen(z_candid).data

        # Define retrieval model
        xp = fbr.xp
        knn = KNNRetriever(xp.array(data['m']))

        # Define the value holders
        text_hists = []
        rank_hists = []
        index_hists = []
        x_gen = []

        # Start the training loop
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

            fbr.reset_state()

            for t in range(args.turn):

                lam = 1.0 if t == 0 else args.lam

                # User feedback to the comparison image
                text = sim(
                    target=Variable(converter(data['f'][target_index], device)),
                    candid=[denormalize(x) for x in x_candid],
                    beam=args.beam,
                    lam=lam,
                    r_eos=args.r_eos,
                    r_dup=args.r_dup,
                    n_rand=args.n_rand,
                    raw_candid=True
                )

                mu, var = fbr(z_candid, text)

                # sample z
                if not args.no_kld:
                    if args.distribution == 'normal':
                        ln_var = F.log(var + 1e-10)
                        z_candid = F.gaussian(mu, ln_var)
                    elif args.distribution == 'vmf':
                        z_candid = von_mises_fisher(mu, var)
                    else:
                        raise NotImplementedError
                else:
                    z_candid = mu

                # Retrieve the nearest image by index, and return percentile rank
                candid_index, rank = knn(z_candid.data, target_index, return_rank=True)

                x_candid = gen(z_candid).data

                x_gen.append(chainer.cuda.to_cpu(x_candid))
                text_hists.append(chainer.cuda.to_cpu(text))
                rank_hists.append(chainer.cuda.to_cpu(rank))
                index_hists.append(candid_index)

            x_target = gen(z_target).data
            x_gen.append(chainer.cuda.to_cpu(x_target))

        preview_dir = '{}/preview/'.format(out)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        # history of generated image
        x_gen = np.clip((np.array(x_gen)+1) * 255 / 2., 0, 255).astype(np.uint8)
        x_gen = np.reshape(
            np.transpose(x_gen, [1, 3, 0, 4, 2]),
            (args.n_case * 64, (args.turn + 1) * 64, 3)
        )
        preview_path = preview_dir + 'gen_{:0>3}.png'.format(trainer.updater.epoch)
        Image.fromarray(x_gen).save(preview_path)

        # history image
        index_hists.append(target_index)
        index_hists = np.array(index_hists)
        idx = np.transpose(index_hists).flatten()
        x_knn = data['x'][idx]
        x_knn = np.clip((np.array(x_knn)+1) * 255 / 2., 0, 255).astype(np.uint8)
        _, _, h, w = x_knn.shape
        x_knn = np.reshape(x_knn, (args.n_case, args.turn + 1, 3, h, w))
        x_knn = np.reshape(
            np.transpose(x_knn, [0, 3, 1, 4, 2]),
            (args.n_case * h, (args.turn + 1) * w, 3)
        )
        preview_path = preview_dir + 'knn_{:0>3}.png'.format(trainer.updater.epoch)
        Image.fromarray(x_knn).save(preview_path)

        # history caption
        text_hists = np.array(text_hists)
        text_hists = np.transpose(text_hists, [1, 0, 2]).tolist()
        s = ''
        for case in range(len(text_hists)):
            s += '<case {}>\n'.format(case)
            for t, text in enumerate(text_hists[case]):
                s += str(t + 1) + ': '
                text = [id_to_word[token] for token in text]
                while '<bos>' in text:
                    text.remove('<bos>')
                while '<eos>' in text:
                    text.remove('<eos>')
                s += ' '.join(text) + '\n'
        preview_path = preview_dir + 'text_{:0>3}.txt'.format(trainer.updater.epoch)
        with open(preview_path, 'w') as file:
            file.write(s)

        # history rank
        rank_hists = np.transpose(np.array(rank_hists))
        preview_path = preview_dir + 'rank_{:0>3}.csv'.format(trainer.updater.epoch)
        with open(preview_path, 'w') as file:
            writer = csv.writer(file)
            header = ['turn{}'.format(t + 1) for t in range(args.turn)]
            writer.writerow(header)
            writer.writerows(rank_hists.tolist())

    return evaluate
