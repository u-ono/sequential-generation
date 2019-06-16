import chainer
import chainer.functions as F
from chainer import Variable

from src.functions.vmf import spherical_kl_divergence, von_mises_fisher


class VAEUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.enc, self.gen = kwargs.pop('models')
        self.coeff = kwargs.pop('coeff')
        self.distribution = kwargs.pop('distribution')
        super(VAEUpdater, self).__init__(*args, **kwargs)

    def update_core(self):

        opt_gen = self.get_optimizer('gen')
        opt_enc = self.get_optimizer('enc')

        x_batch = self._iterators['main'].next()
        x_real = self.converter(x_batch, self.device)

        # Fake image from reconstruction

        mu, var = self.enc(Variable(x_real))

        if self.distribution == 'normal':
            ln_var = F.log(var + 1e-5)
            z_rec = F.gaussian(mu, ln_var)
            loss_kl = F.gaussian_kl_divergence(mu, ln_var) / mu.data.size
        elif self.distribution == 'vmf':
            z_rec = von_mises_fisher(mu, var)
            loss_kl = F.sum(spherical_kl_divergence(mu, var)) / mu.data.size
        else:
            raise NotImplementedError

        x_rec = self.gen(z_rec)

        loss_rec = F.mean_squared_error(x_rec, x_real)

        loss_enc = loss_rec + self.coeff * loss_kl
        loss_gen = loss_rec

        # Update

        self.enc.cleargrads()
        loss_enc.backward()
        opt_enc.update()

        self.gen.cleargrads()
        loss_gen.backward()
        opt_gen.update()

        # Report losses

        chainer.report({'loss': loss_enc}, self.enc)
        chainer.report({'loss': loss_gen}, self.gen)


class HSVAEUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.enc, self.gen = kwargs.pop('models')
        self.coeff = kwargs.pop('coeff')
        super(HSVAEUpdater, self).__init__(*args, **kwargs)

    '''
    @staticmethod
    def reparameterize(mu, var):
        xp = chainer.cuda.get_array_module(mu)
        q = VonMisesFisher(mu, var)
        p = HyperSphericalUniform(dim=mu.shape[-1] - 1, batch_shape=mu.shape[:-1], xp=xp)
        z = q.sample(sample_shape=())
        return z, q, p
    '''

    def update_core(self):

        opt_gen = self.get_optimizer('gen')
        opt_enc = self.get_optimizer('enc')

        x_batch = self._iterators['main'].next()
        x_real = self.converter(x_batch, self.device)

        # Fake image from reconstruction

        mu, var = self.enc(Variable(x_real))
        #z_rec, q, p = self.reparameterize(mu, var)
        z_rec = von_mises_fisher(mu, var)
        x_rec = self.gen(z_rec)

        #loss_kl = F.mean(chainer.kl_divergence(q, p))
        loss_kl = F.mean(spherical_kl_divergence(mu, var))
        loss_rec = F.mean_squared_error(x_rec, x_real)

        loss_enc = loss_rec + self.coeff * loss_kl
        loss_gen = loss_rec

        # Update

        self.enc.cleargrads()
        loss_enc.backward()
        opt_enc.update()

        self.gen.cleargrads()
        loss_gen.backward()
        opt_gen.update()

        # Report losses

        chainer.report({'loss': loss_enc}, self.enc)
        chainer.report({'loss': loss_gen}, self.gen)


class _VAEGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.enc, self.gen, self.dis = kwargs.pop('models')
        self.c_rec = kwargs.pop('c_rec')
        self.c_kld = kwargs.pop('c_kld')
        self.n_dis = kwargs.pop('n_dis')
        self.distribution = kwargs.pop('distribution')
        super(_VAEGANUpdater, self).__init__(*args, **kwargs)

    def update_core(self):

        opt_gen = self.get_optimizer('gen')
        opt_dis = self.get_optimizer('dis')
        opt_enc = self.get_optimizer('enc')

        x_batch = self._iterators['main'].next()
        batchsize = len(x_batch)
        x_real = self.converter(x_batch, self.device)

        loss_enc = 0
        loss_gen = 0
        loss_dis = 0

        # Fake image from reconstruction

        mu, var = self.enc(Variable(x_real))

        if self.distribution == 'normal':
            ln_var = F.log(var + 1e-5)
            z_rec = F.gaussian(mu, ln_var)
            loss_kl = F.gaussian_kl_divergence(mu, ln_var) / batchsize
        elif self.distribution == 'vmf':
            z_rec = von_mises_fisher(mu, var)
            loss_kl = F.sum(spherical_kl_divergence(mu, var)) / batchsize
        else:
            raise NotImplementedError

        x_rec = self.gen(z_rec)
        y_rec, l_rec = self.dis(x_rec)
        loss_enc += self.c_kld * loss_kl
        loss_gen += (1 - self.c_rec) * F.sum(F.softplus(-y_rec)) / batchsize
        loss_dis += F.sum(F.softplus(y_rec)) / batchsize

        # Fake image from random noise

        z_rand = self.gen.make_hidden(batchsize)
        x_fake = self.gen(z_rand)
        y_fake, l_fake = self.dis(x_fake)

        loss_gen += (1 - self.c_rec) * F.sum(F.softplus(-y_fake)) / batchsize
        loss_dis += F.sum(F.softplus(y_fake)) / batchsize

        # Real image

        y_real, l_real = self.dis(x_real)
        size = l_real.shape[2] * l_real.shape[3]

        loss_enc += (1 - self.c_kld) * F.mean_squared_error(l_rec, l_real) * size
        loss_gen += self.c_rec * F.mean_squared_error(l_rec, l_real) * size
        loss_dis += F.sum(F.softplus(-y_real)) / batchsize

        self.enc.cleargrads()
        loss_enc.backward()
        opt_enc.update()

        self.gen.cleargrads()
        loss_gen.backward()
        opt_gen.update()

        self.dis.cleargrads()
        loss_dis.backward()
        opt_dis.update()

        # Report losses

        chainer.report({'loss': loss_enc}, self.enc)
        chainer.report({'loss': loss_gen}, self.gen)
        chainer.report({'loss': loss_dis}, self.dis)


class VAEGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.enc, self.gen, self.dis = kwargs.pop('models')
        self.c_rec = kwargs.pop('c_rec')
        self.c_kld = kwargs.pop('c_kld')
        self.n_dis = kwargs.pop('n_dis')
        self.distribution = kwargs.pop('distribution')
        super(VAEGANUpdater, self).__init__(*args, **kwargs)

    def update_core(self):

        opt_gen = self.get_optimizer('gen')
        opt_dis = self.get_optimizer('dis')
        opt_enc = self.get_optimizer('enc')

        for i in range(self.n_dis):

            if i == 0:  # Update Enc, Gen, Dis

                x_batch = self._iterators['main'].next()
                batchsize = len(x_batch)
                x_real = self.converter(x_batch, self.device)

                loss_enc = 0
                loss_gen = 0
                loss_dis = 0

                # Fake image from reconstruction

                mu, var = self.enc(Variable(x_real))

                if self.distribution == 'normal':
                    ln_var = F.log(var + 1e-5)
                    z_rec = F.gaussian(mu, ln_var)
                    loss_kl = F.gaussian_kl_divergence(mu, ln_var) / batchsize
                elif self.distribution == 'vmf':
                    z_rec = von_mises_fisher(mu, var)
                    loss_kl = F.sum(spherical_kl_divergence(mu, var)) / batchsize
                else:
                    raise NotImplementedError

                x_rec = self.gen(z_rec)
                y_rec, l_rec = self.dis(x_rec)
                loss_enc += loss_kl
                loss_gen += (1 - self.c_rec) * F.sum(F.softplus(-y_rec)) / batchsize
                loss_dis += F.sum(F.softplus(y_rec)) / batchsize

                # Fake image from random noise

                z_rand = self.gen.make_hidden(batchsize)
                x_fake = self.gen(z_rand)
                y_fake, l_fake = self.dis(x_fake)

                loss_gen += (1 - self.c_rec) * F.sum(F.softplus(-y_fake)) / batchsize
                loss_dis += F.sum(F.softplus(y_fake)) / batchsize

                # Real image

                y_real, l_real = self.dis(x_real)
                size = l_real.shape[2] * l_real.shape[3]

                loss_enc += F.mean_squared_error(l_rec, l_real) * size
                loss_gen += self.c_rec * F.mean_squared_error(l_rec, l_real) * size
                loss_dis += F.sum(F.softplus(-y_real)) / batchsize

                self.enc.cleargrads()
                loss_enc.backward()
                opt_enc.update()

                self.gen.cleargrads()
                loss_gen.backward()
                opt_gen.update()

                self.dis.cleargrads()
                loss_dis.backward()
                opt_dis.update()

                chainer.report({'loss': loss_enc}, self.enc)
                chainer.report({'loss': loss_gen}, self.gen)
                chainer.report({'loss': loss_dis}, self.dis)

            else:  # Update Dis only

                x_batch = self._iterators['main'].next()
                batchsize = len(x_batch)
                x_real = self.converter(x_batch, self.device)

                loss_dis = 0

                with chainer.using_config('enable_backprop', False):

                    # Fake image from reconstruction

                    mu, var = self.enc(Variable(x_real))

                    if self.distribution == 'normal':
                        ln_var = F.log(var + 1e-5)
                        z_rec = F.gaussian(mu, ln_var)
                    elif self.distribution == 'vmf':
                        z_rec = von_mises_fisher(mu, var)
                    else:
                        raise NotImplementedError

                    x_rec = self.gen(z_rec)

                    # Fake image from random noise

                    z_rand = self.gen.make_hidden(batchsize)
                    x_fake = self.gen(z_rand)

                # Fake image from reconstruction

                y_rec, l_rec = self.dis(x_rec)
                loss_dis += F.sum(F.softplus(y_rec)) / batchsize

                # Fake image from random noise

                y_fake, l_fake = self.dis(x_fake)
                loss_dis += F.sum(F.softplus(y_fake)) / batchsize

                # Real image

                y_real, l_real = self.dis(x_real)
                loss_dis += F.sum(F.softplus(-y_real)) / batchsize

                self.dis.cleargrads()
                loss_dis.backward()
                opt_dis.update()

                # Report losses
                chainer.report({'loss': loss_dis}, self.dis)
