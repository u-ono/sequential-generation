import chainer
import chainer.functions as F
from chainer import Variable
import numpy as np
from src.miscs.knn_retriever import KNNRetriever
from src.functions.vmf import von_mises_fisher, spherical_kl_divergence
from src.miscs.array_utils import denormalize


class _RetrievalUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.data = kwargs.pop('data')
        self.args = kwargs.pop('args')
        super(_RetrievalUpdater, self).__init__(*args, **kwargs)

    def update_core(self):

        args = self.args
        data = self.data

        # Define the models
        fbr, sim = self.models
        xp = fbr.xp

        # Fake iterator (just for epoch count)
        batch = self.get_iterator('main').next()
        num_batch = len(batch)

        # Define indices for training
        num_data = len(self.data['name'])
        target_index = np.random.choice(num_data, num_batch, replace=False)
        next_index = np.random.choice(num_data, num_batch, replace=False)

        # Set up KNN retriever
        knn_retriever = KNNRetriever(xp.array(data['m']))

        # Define the value holders
        ranks = []
        loss_triplet = 0
        loss_kl = 0

        # Start training loop
        fbr.reset_state()

        for t in range(args.turn):

            # User feedback to the comparison image
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                text = sim(
                    Variable(self.converter(data['f'][target_index], self.device)),
                    Variable(self.converter(data['f'][next_index], self.device)),
                    beam=args.beam,
                    lam=args.lam,
                    bias=args.bias,
                    n_rand=args.n_rand
                )

            # Hidden z this turn
            z = Variable(self.converter(data['m'][next_index], self.device))

            # Transform hidden z
            mu, var = fbr(z, text)

            if not args.no_kld:
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
            else:
                z = mu

            # Calculate triplet loss
            random_index = np.random.choice(num_data, num_batch, replace=False)

            loss_triplet += F.triplet(
                anchor=z,
                positive=self.converter(data['m'][target_index], self.device),
                negative=self.converter(data['m'][random_index], self.device),
                margin=args.margin
            )

            # Retrieve the nearest image by index, and return percentile rank
            next_index, rank = knn_retriever(z.data, target_index, return_rank=True)
            ranks.append(chainer.cuda.to_cpu(rank))

        loss = loss_triplet + args.c_kld * loss_kl

        # Update parameters of FE
        fbr.cleargrads()
        loss.backward()
        self.get_optimizer('main').update()

        # Report loss/rank
        chainer.report({'loss/triplet': loss_triplet, 'loss/kl': loss_kl, 'loss/total': loss}, fbr)
        ranks = np.mean(np.array(ranks), axis=1)
        for t in range(args.turn):
            chainer.report({'rank{}'.format(t+1): ranks[t]}, fbr)


class FeedbackReceiverUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.data = kwargs.pop('data')
        self.args = kwargs.pop('args')
        super(FeedbackReceiverUpdater, self).__init__(*args, **kwargs)

    def update_core(self):

        args = self.args
        data = self.data

        # Define the models
        fbr, gen, sim = self.models
        xp = fbr.xp

        # Fake iterator (just for epoch count)
        batch = self.get_iterator('main').next()
        num_batch = len(batch)
        num_data = len(self.data['name'])

        # Define indices for training
        target_index = np.random.choice(num_data, num_batch, replace=False)
        z_target = self.converter(data['m'][target_index], self.device)

        # Initialize z & x
        '''
        candid_index = np.random.choice(num_data, num_batch, replace=False)
        z_candid = self.converter(data['m'][candid_index], self.device)
        x_candid = self.converter(data['x'][candid_index], self.device)
        '''
        z_candid = self.converter(np.zeros(z_target.shape, 'f'), self.device)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x_candid = gen(z_candid).data

        # Set up KNN retriever
        knn = KNNRetriever(xp.array(data['m']))

        # Define the value holders
        ranks = []
        loss_triplet = 0
        loss_kl = 0

        # Start training loop
        fbr.reset_state()

        for t in range(args.turn):

            # User feedback to the comparison image
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

                lam = 1.0 if t == 0 else args.lam

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

            # Transform hidden z
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

            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                x_candid = gen(z_candid).data

            ranks.append(chainer.cuda.to_cpu(rank))

        loss = loss_triplet + args.c_kld * loss_kl

        # Update parameters of FE
        fbr.cleargrads()
        loss.backward()
        self.get_optimizer('main').update()

        # Report loss/rank
        chainer.report({'loss/triplet': loss_triplet, 'loss/kl': loss_kl, 'loss/total': loss}, fbr)
        ranks = np.mean(np.array(ranks), axis=1)
        for t in range(args.turn):
            chainer.report({'rank{}'.format(t+1): ranks[t]}, fbr)


class RetrievalUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.data = kwargs.pop('data')
        self.args = kwargs.pop('args')
        super(RetrievalUpdater, self).__init__(*args, **kwargs)

    def update_core(self):

        args = self.args
        data = self.data

        # Define the models
        fbr, sim = self.models
        xp = fbr.xp

        # Fake iterator (just for epoch count)
        batch = self.get_iterator('main').next()
        num_batch = len(batch)

        # Define indices for training
        num_data = len(self.data['name'])

        target_index = np.random.choice(num_data, num_batch, replace=False)
        z_target = self.converter(data['m'][target_index], self.device)

        candid_index = np.random.choice(num_data, num_batch, replace=False)
        z_candid = self.converter(data['m'][candid_index], self.device)

        # Set up KNN retriever
        knn_retriever = KNNRetriever(xp.array(data['m']))

        # Define the value holders
        ranks = []
        loss_triplet = 0
        loss_kl = 0

        # Start training loop
        fbr.reset_state()

        for t in range(args.turn):

            # User feedback to the comparison image
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                text = sim(
                    Variable(self.converter(data['f'][target_index], self.device)),
                    Variable(self.converter(data['f'][candid_index], self.device)),
                    beam=args.beam,
                    lam=args.lam,
                    r_eos=args.r_eos,
                    r_dup=args.r_dup,
                    n_rand=args.n_rand
                )

            # Transform hidden z
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
            candid_index, rank = knn_retriever(z_candid.data, target_index, return_rank=True)
            z_candid = self.converter(data['m'][candid_index], self.device)

            ranks.append(chainer.cuda.to_cpu(rank))

        loss = loss_triplet + args.c_kld * loss_kl

        # Update parameters of FE
        fbr.cleargrads()
        loss.backward()
        self.get_optimizer('main').update()

        # Report loss/rank
        chainer.report({'loss/triplet': loss_triplet, 'loss/kl': loss_kl, 'loss/total': loss}, fbr)
        ranks = np.mean(np.array(ranks), axis=1)
        for t in range(args.turn):
            chainer.report({'rank{}'.format(t+1): ranks[t]}, fbr)
