import chainer
import chainer.functions as F
import numpy as np
from src.miscs.knn_retriever import KNNRetriever, compute_z
from src.miscs.array_utils import denormalize
from src.functions.vmf import von_mises_fisher, spherical_kl_divergence


class FineTuneUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):

        self.models = kwargs.pop('models')
        self.data = kwargs.pop('data')
        self.args = kwargs.pop('args')

        super(FineTuneUpdater, self).__init__(*args, **kwargs)

    def update_core(self):

        args = self.args
        data = self.data

        enc, gen, dis, fbr, sim = self.models

        opt_enc = self.get_optimizer('enc')
        opt_gen = self.get_optimizer('gen')
        opt_dis = self.get_optimizer('dis')
        opt_fbr = self.get_optimizer('fbr')

        # Fake iterator just for counting epoch
        batch = self.get_iterator('main').next()

        num_data = len(data['name'])
        num_batch = len(batch)

        # Set target
        target_index = np.random.choice(num_data, num_batch, replace=False)
        x_target = self.converter(data['x'][target_index], self.device)
        z_target, _ = enc(x_target)

        # Initialize candidate
        z_candid = self.converter(np.zeros(z_target.shape, 'f'), self.device)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x_candid = gen(z_candid)

        '''
        # Set up KNN retriever
        zs, _ = compute_z(enc, data['x'], num_batch)
        knn = KNNRetriever(zs)

        ranks = []
        '''

        loss_enc = 0
        loss_gen = 0
        loss_dis = 0
        loss_fbr = 0

        fbr.reset_state()

        #
        # target real image
        #

        y_target, l_target = dis(x_target)
        size = l_target.shape[2] * l_target.shape[3]

        loss_dis += F.sum(F.softplus(-y_target)) / num_batch * args.turn

        for t in range(args.turn):

            #
            # generate feedback
            #

            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

                lam = 1.0 if t == 0 else args.lam

                text = sim(
                    target=self.converter(data['f'][target_index], self.device),
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
            # decode next candidate
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

            '''
            # Retrieve the nearest image by index, and return percentile rank
            _, rank = knn(z_candid.data, target_index, return_rank=True)

            ranks.append(chainer.cuda.to_cpu(rank))
            '''

        loss_enc /= args.turn
        loss_gen /= args.turn
        loss_dis /= args.turn
        loss_fbr /= args.turn

        enc.cleargrads()
        loss_enc.backward()
        opt_enc.update()

        gen.cleargrads()
        loss_gen.backward()
        opt_gen.update()

        dis.cleargrads()
        loss_dis.backward()
        opt_dis.update()

        fbr.cleargrads()
        loss_fbr.backward()
        opt_fbr.update()

        chainer.report({'loss': loss_enc}, enc)
        chainer.report({'loss': loss_gen}, gen)
        chainer.report({'loss': loss_dis}, dis)
        chainer.report({'loss': loss_fbr}, fbr)

        '''
        ranks = np.mean(np.array(ranks), axis=1)
        for t in range(args.turn):
            chainer.report({'rank{}'.format(t + 1): ranks[t]}, fbr)
        '''


class ResFineTuneUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):

        self.models = kwargs.pop('models')
        self.data = kwargs.pop('data')
        self.args = kwargs.pop('args')

        super(ResFineTuneUpdater, self).__init__(*args, **kwargs)

    def update_core(self):

        args = self.args
        data = self.data

        enc, gen, dis, fbr, sim = self.models

        opt_gen = self.get_optimizer('gen')
        opt_dis = self.get_optimizer('dis')
        opt_fbr = self.get_optimizer('fbr')

        # Fake iterator just for counting epoch
        batch = self.get_iterator('main').next()

        num_data = len(data['name'])
        num_batch = len(batch)

        # Set target
        target_index = np.random.choice(num_data, num_batch, replace=False)
        x_target = self.converter(data['x'][target_index], self.device)
        f_target = enc(x_target)

        # Initialize candidate
        z_candid = self.converter(np.zeros((num_batch, args.dim_z), 'f'), self.device)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x_candid = gen(z_candid)

        loss_gen = 0
        loss_fbr = 0
        loss_dis = 0

        fbr.reset_state()

        #
        # target real image
        #

        y_target, _ = dis(x_target)
        loss_dis += F.sum(F.softplus(-y_target)) / num_batch * args.turn

        for t in range(args.turn):

            #
            # generate feedback
            #

            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

                lam = 1.0 if t == 0 else args.lam

                text = sim(
                    target=self.converter(data['f'][target_index], self.device),
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
            # decode next candidate
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

        loss_gen /= args.turn
        loss_dis /= args.turn
        loss_fbr /= args.turn

        enc.cleargrads()

        gen.cleargrads()
        loss_gen.backward()
        opt_gen.update()

        dis.cleargrads()
        loss_dis.backward()
        opt_dis.update()

        fbr.cleargrads()
        loss_fbr.backward()
        opt_fbr.update()

        chainer.report({'loss': loss_gen}, gen)
        chainer.report({'loss': loss_dis}, dis)
        chainer.report({'loss': loss_fbr}, fbr)
