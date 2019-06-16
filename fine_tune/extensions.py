import chainer
import numpy as np
import chainer.functions as F
from chainer.training import extensions
from chainer import reporter as reporter_module
from chainer import Variable
import os
from PIL import Image
import csv

from src.miscs.knn_retriever import KNNRetriever, compute_z, compute_f
from src.miscs.array_utils import denormalize
from src.functions.vmf import von_mises_fisher, spherical_kl_divergence


class FineTuneEvaluator(extensions.Evaluator):

    default_name = 'val'

    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.data = kwargs.pop('data')
        self.args = kwargs.pop('args')
        enc, gen, dis, fbr, sim = self.models
        target = {'enc': enc, 'gen': gen, 'dis': dis, 'fbr': fbr}
        super(FineTuneEvaluator, self).__init__(*args, **kwargs, target=target)

    def evaluate(self):

        args = self.args
        data = self.data

        enc, gen, dis, fbr, sim = self.models

        # Check how many times to iterate
        batch = self.get_iterator('main').next()
        num_batch = len(batch)
        num_data = len(data['name'])
        iteration = num_data // num_batch

        summary = reporter_module.DictSummary()

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

            # Define retrieval model
            zs, _ = compute_z(enc, data['x'], num_batch)
            knn = KNNRetriever(zs)

            for i in range(iteration):

                # Define indices for training
                target_index = np.random.choice(num_data, num_batch, replace=False)
                x_target = self.converter(data['x'][target_index], self.device)
                z_target, _ = enc(x_target)

                z_candid = self.converter(np.zeros(z_target.shape, 'f'), self.device)
                x_candid = gen(z_candid)

                # Define the value holders
                ranks = []
                loss_enc = 0
                loss_gen = 0
                loss_dis = 0
                loss_fbr = 0

                # Start training loop

                fbr.reset_state()

                #
                # target real image
                #

                y_target, l_target = dis(x_target)
                size = l_target.shape[2] * l_target.shape[3]

                loss_dis += F.sum(F.softplus(-y_target)) / num_batch * args.turn

                for t in range(args.turn):

                    lam = 1.0 if t == 0 else args.lam

                    # User feedback to the comparison image
                    text = sim(
                        target=Variable(self.converter(data['f'][target_index], self.device)),
                        candid=[denormalize(x) for x in x_candid.data],
                        beam=args.beam,
                        lam=lam,
                        r_eos=args.r_eos,
                        r_dup=args.r_dup,
                        n_rand=args.n_rand,
                        raw_candid=True
                    )

                    #
                    # encode candidate image
                    #

                    z_candid, _ = enc(x_candid)
                    mu_candid, var_candid = fbr(z_candid, text)

                    if not args.no_kld:
                        # Compute KL divergence & sample z
                        if args.distribution == 'normal':
                            ln_var_candid = F.log(var_candid + 1e-10)
                            z_candid = F.gaussian(mu_candid, ln_var_candid)
                            loss_kl = F.gaussian_kl_divergence(mu_candid, ln_var_candid) / mu_candid.data.size
                        elif args.distribution == 'vmf':
                            z_candid = von_mises_fisher(mu_candid, var_candid)
                            loss_kl = F.sum(spherical_kl_divergence(mu_candid, var_candid)) / mu_candid.data.size
                        else:
                            raise NotImplementedError
                    else:
                        z_candid = mu_candid
                        loss_kl = 0

                    loss_enc += loss_kl
                    loss_fbr += loss_kl

                    #
                    # decode next candidate image
                    #

                    x_candid = gen(z_candid)
                    y_candid, l_candid = dis(x_candid)

                    loss_enc += args.c_rec * F.mean_squared_error(l_candid, l_target) * size
                    loss_fbr += args.c_rec * F.mean_squared_error(l_candid, l_target) * size
                    loss_gen += args.c_rec * F.mean_squared_error(l_candid, l_target) * size
                    loss_gen += F.sum(F.softplus(-y_candid)) / num_batch
                    loss_dis += F.sum(F.softplus(y_candid)) / num_batch

                    #
                    # triplet loss
                    #

                    random_index = np.random.choice(num_data, num_batch, replace=False)
                    x_random = self.converter(data['x'][random_index], self.device)
                    z_random, _ = enc(x_random)

                    loss_triplet = F.triplet(
                        anchor=z_candid,
                        positive=z_target,
                        negative=z_random,
                        margin=args.margin
                    )

                    loss_fbr += loss_triplet
                    loss_enc += loss_triplet
                    loss_gen += loss_triplet

                    # Retrieve the nearest image by index, and return percentile rank
                    _, rank = knn(z_candid.data, target_index, return_rank=True)

                    ranks.append(chainer.cuda.to_cpu(rank))

                loss_enc /= args.turn
                loss_gen /= args.turn
                loss_dis /= args.turn
                loss_fbr /= args.turn

                summary.add({FineTuneEvaluator.default_name + '/enc/loss': loss_enc})
                summary.add({FineTuneEvaluator.default_name + '/gen/loss': loss_gen})
                summary.add({FineTuneEvaluator.default_name + '/dis/loss': loss_dis})
                summary.add({FineTuneEvaluator.default_name + '/fbr/loss': loss_fbr})

                ranks = np.mean(np.array(ranks), axis=1)
                for t in range(args.turn):
                    summary.add({FineTuneEvaluator.default_name + '/rank{}'.format(t+1): ranks[t]})

        return summary.compute_mean()


class ResFineTuneEvaluator(extensions.Evaluator):

    default_name = 'val'

    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.data = kwargs.pop('data')
        self.args = kwargs.pop('args')
        enc, gen, dis, fbr, sim = self.models
        target = {'enc': enc, 'gen': gen, 'dis': dis, 'fbr': fbr}
        super(ResFineTuneEvaluator, self).__init__(*args, **kwargs, target=target)

    def evaluate(self):

        args = self.args
        data = self.data

        enc, gen, dis, fbr, sim = self.models

        # Check how many times to iterate
        batch = self.get_iterator('main').next()
        num_batch = len(batch)
        num_data = len(data['name'])
        iteration = num_data // num_batch

        summary = reporter_module.DictSummary()

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

            # Define retrieval model
            fs = compute_f(enc, data['x'], num_batch)
            knn = KNNRetriever(fs)

            for i in range(iteration):

                # Define indices for training
                target_index = np.random.choice(num_data, num_batch, replace=False)
                x_target = self.converter(data['x'][target_index], self.device)
                f_target = enc(x_target)

                z_candid = self.converter(np.zeros((num_batch, args.dim_z), 'f'), self.device)
                x_candid = gen(z_candid)

                # Define the value holders
                ranks = []
                loss_gen = 0
                loss_dis = 0
                loss_fbr = 0

                # Start training loop

                fbr.reset_state()

                #
                # target real image
                #

                y_target, _ = dis(x_target)
                loss_dis += F.sum(F.softplus(-y_target)) / num_batch * args.turn

                for t in range(args.turn):

                    lam = 1.0 if t == 0 else args.lam

                    # User feedback to the comparison image
                    text = sim(
                        target=Variable(self.converter(data['f'][target_index], self.device)),
                        candid=[denormalize(x) for x in x_candid.data],
                        beam=args.beam,
                        lam=lam,
                        r_eos=args.r_eos,
                        r_dup=args.r_dup,
                        n_rand=args.n_rand,
                        raw_candid=True
                    )

                    #
                    # encode candidate image
                    #

                    mu_candid, var_candid = fbr(z_candid, text)

                    if not args.no_kld:
                        # Compute KL divergence & sample z
                        if args.distribution == 'normal':
                            ln_var_candid = F.log(var_candid + 1e-10)
                            z_candid = F.gaussian(mu_candid, ln_var_candid)
                            loss_kl = F.gaussian_kl_divergence(mu_candid, ln_var_candid) / mu_candid.data.size
                        elif args.distribution == 'vmf':
                            z_candid = von_mises_fisher(mu_candid, var_candid)
                            loss_kl = F.sum(spherical_kl_divergence(mu_candid, var_candid)) / mu_candid.data.size
                        else:
                            raise NotImplementedError
                    else:
                        z_candid = mu_candid
                        loss_kl = 0

                    loss_fbr += loss_kl

                    #
                    # decode next candidate image
                    #

                    x_candid = gen(z_candid)
                    y_candid, _ = dis(x_candid)

                    loss_gen += F.sum(F.softplus(-y_candid)) / num_batch
                    loss_dis += F.sum(F.softplus(y_candid)) / num_batch

                    #
                    # triplet loss
                    #

                    f_candid = enc(x_candid)

                    random_index = np.random.choice(num_data, num_batch, replace=False)
                    x_random = self.converter(data['x'][random_index], self.device)
                    f_random = enc(x_random)

                    loss_triplet = F.triplet(
                        anchor=f_candid,
                        positive=f_target,
                        negative=f_random,
                        margin=args.margin
                    )

                    loss_fbr += loss_triplet
                    loss_gen += loss_triplet

                    # Retrieve the nearest image by index, and return percentile rank
                    _, rank = knn(f_candid.data, target_index, return_rank=True)

                    ranks.append(chainer.cuda.to_cpu(rank))

                loss_gen /= args.turn
                loss_dis /= args.turn
                loss_fbr /= args.turn

                summary.add({FineTuneEvaluator.default_name + '/gen/loss': loss_gen})
                summary.add({FineTuneEvaluator.default_name + '/dis/loss': loss_dis})
                summary.add({FineTuneEvaluator.default_name + '/fbr/loss': loss_fbr})

                ranks = np.mean(np.array(ranks), axis=1)
                for t in range(args.turn):
                    summary.add({FineTuneEvaluator.default_name + '/rank{}'.format(t+1): ranks[t]})

        return summary.compute_mean()


def visualize(models, data, args, out, seed=0):

    @chainer.training.make_extension()
    def evaluate(trainer):

        device = trainer.updater.device
        converter = trainer.updater.converter

        enc, gen, dis, fbr, sim = models
        num_data = len(data['name'])

        # Set vocabulary and reverse dict
        vocab = sim.vocab
        id_to_word = {}
        for k, v in vocab.items():
            id_to_word[v] = k

        # Define indices for training
        np.random.seed(seed)
        target_index = np.random.choice(num_data, args.n_case, replace=False)
        np.random.seed()

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

            x_target = Variable(converter(data['x'][target_index], device))
            z_target, _ = enc(x_target)

            z_candid = converter(np.zeros(z_target.shape, 'f'), device)
            x_candid = gen(z_candid)

            # Define retrieval model
            zs, _ = compute_z(enc, data['x'], args.batch_size)
            knn = KNNRetriever(zs)

            # Define the value holders
            text_hists = []
            rank_hists = []
            index_hists = []
            x_gen = []

            # Start the training loop

            fbr.reset_state()

            for t in range(args.turn):

                #
                # generate feedback
                #

                lam = 1.0 if t == 0 else args.lam

                text = sim(
                    target=Variable(converter(data['f'][target_index], device)),
                    candid=[denormalize(x) for x in x_candid.data],
                    beam=args.beam,
                    lam=lam,
                    r_eos=args.r_eos,
                    r_dup=args.r_dup,
                    n_rand=args.n_rand,
                    raw_candid=True
                )

                #
                # encode candidate image
                #

                z_candid, _ = enc(x_candid)
                mu_candid, var_candid = fbr(z_candid, text)

                if not args.no_kld:
                    # Compute KL divergence & sample z
                    if args.distribution == 'normal':
                        ln_var_candid = F.log(var_candid + 1e-10)
                        z_candid = F.gaussian(mu_candid, ln_var_candid)
                    elif args.distribution == 'vmf':
                        z_candid = von_mises_fisher(mu_candid, var_candid)
                    else:
                        raise NotImplementedError
                else:
                    z_candid = mu_candid

                #
                # decode next candidate image
                #

                x_candid = gen(z_candid)

                # Retrieve the nearest image by index, and return percentile rank
                candid_index, rank = knn(z_candid.data, target_index, return_rank=True)

                x_gen.append(chainer.cuda.to_cpu(x_candid.data))
                text_hists.append(chainer.cuda.to_cpu(text))
                rank_hists.append(chainer.cuda.to_cpu(rank))
                index_hists.append(candid_index)

            x_target = gen(z_target)
            x_gen.append(chainer.cuda.to_cpu(x_target.data))

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


def res_visualize(models, data, args, out, seed=0):

    @chainer.training.make_extension()
    def evaluate(trainer):

        device = trainer.updater.device
        converter = trainer.updater.converter

        enc, gen, dis, fbr, sim = models
        num_data = len(data['name'])

        # Set vocabulary and reverse dict
        vocab = sim.vocab
        id_to_word = {}
        for k, v in vocab.items():
            id_to_word[v] = k

        # Define indices for training
        np.random.seed(seed)
        target_index = np.random.choice(num_data, args.n_case, replace=False)
        np.random.seed()

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

            x_target = converter(data['x'][target_index], device)

            z_candid = converter(np.zeros((args.n_case, args.dim_z), 'f'), device)
            x_candid = gen(z_candid)

            # Define retrieval model
            fs = compute_f(enc, data['x'], args.batch_size)
            knn = KNNRetriever(fs)

            # Define the value holders
            text_hists = []
            rank_hists = []
            index_hists = []
            x_gen = []

            # Start the training loop

            fbr.reset_state()

            for t in range(args.turn):

                #
                # generate feedback
                #

                lam = 1.0 if t == 0 else args.lam

                text = sim(
                    target=Variable(converter(data['f'][target_index], device)),
                    candid=[denormalize(x) for x in x_candid.data],
                    beam=args.beam,
                    lam=lam,
                    r_eos=args.r_eos,
                    r_dup=args.r_dup,
                    n_rand=args.n_rand,
                    raw_candid=True
                )

                #
                # encode candidate image
                #

                mu_candid, var_candid = fbr(z_candid, text)

                if not args.no_kld:
                    # Compute KL divergence & sample z
                    if args.distribution == 'normal':
                        ln_var_candid = F.log(var_candid + 1e-10)
                        z_candid = F.gaussian(mu_candid, ln_var_candid)
                    elif args.distribution == 'vmf':
                        z_candid = von_mises_fisher(mu_candid, var_candid)
                    else:
                        raise NotImplementedError
                else:
                    z_candid = mu_candid

                #
                # decode next candidate image
                #

                x_candid = gen(z_candid)
                f_candid = enc(x_candid)

                # Retrieve the nearest image by index, and return percentile rank
                candid_index, rank = knn(f_candid.data, target_index, return_rank=True)

                x_gen.append(chainer.cuda.to_cpu(x_candid.data))
                text_hists.append(chainer.cuda.to_cpu(text))
                rank_hists.append(chainer.cuda.to_cpu(rank))
                index_hists.append(candid_index)

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
