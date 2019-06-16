import chainer
import chainer.functions as F
from chainer import Variable


class VanillaUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):

        self.model = kwargs.pop('model')
        self.vocab = kwargs.pop('vocab')
        self.raw = kwargs.pop('raw')

        super(VanillaUpdater, self).__init__(*args, **kwargs)

    def update_core(self):

        xp = self.model.xp
        self.model.cleargrads()
        optimizer = self.get_optimizer('main')

        batch = self.get_iterator('main').next()

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

        chainer.report({'loss': loss.data}, self.model)
        chainer.report({'acc': acc}, self.model)

        loss.backward()
        optimizer.update()


class VanillaMMIUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):

        self.model = kwargs.pop('model')
        self.vocab = kwargs.pop('vocab')
        self.raw = kwargs.pop('raw')

        self.margin = kwargs.pop('margin')
        self.lam = kwargs.pop('lam')

        super(VanillaMMIUpdater, self).__init__(*args, **kwargs)

    def update_core(self):

        xp = self.model.xp
        self.model.cleargrads()
        optimizer = self.get_optimizer('main')

        batch = self.get_iterator('main').next()

        target, candid, captions = self.converter(batch, device=self.device, padding=self.vocab['<eos>'])
        len_caption = captions.shape[1]

        # Compute target loss

        self.model.reset_state()
        vg_t, vl_t, k_t = self.model.encode(target, raw=self.raw)

        loss_target = 0
        acc_target = 0
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

            loss_target += F.softmax_cross_entropy(y, t)
            acc_target += xp.sum((y_id == t.data) * subject)
            size += xp.sum(subject)

        loss_target = loss_target * len(batch) / size
        acc_target /= size

        # Compute candidate caption likelihood

        self.model.reset_state()
        vg_c, vl_c, k_c = self.model.encode(candid, raw=self.raw)

        loss_candid = 0
        acc_candid = 0
        size = 0

        for i in range(len_caption-1):
            subject = (captions[:, i] != self.vocab['<eos>']).astype('float32')

            if (subject == 0).all():
                break

            x = Variable(xp.asarray(captions[:, i]))
            t = Variable(xp.asarray(captions[:, i+1]))

            y, _ = self.model.decode(x, vg_c, vl_c, k_c, i)
            y_id = xp.argmax(y.data, axis=1)
            mask = F.broadcast_to(subject[:, None], y.data.shape)
            y = y * mask

            loss_candid += F.softmax_cross_entropy(y, t)
            acc_candid += xp.sum((y_id == t.data) * subject)
            size += xp.sum(subject)

        loss_candid = loss_candid * len(batch) / size
        acc_candid /= size

        # Max Margin Loss

        loss_max_margin = F.maximum(
            self.margin + loss_target - loss_candid,
            Variable(xp.zeros_like(loss_target))
        )

        loss = loss_target + self.lam * loss_max_margin

        chainer.report({'loss/target': loss_target.data, 'loss/total': loss.data}, self.model)
        chainer.report({'acc/target': acc_target, 'acc/candid': acc_candid}, self.model)

        loss.backward()
        optimizer.update()


class RelativeMMIUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop('model')
        self.vocab = kwargs.pop('vocab')
        self.raw = kwargs.pop('raw')
        self.margin = kwargs.pop('margin')
        self.lam = kwargs.pop('lam')
        super(RelativeMMIUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        xp = self.model.xp
        self.model.cleargrads()
        optimizer = self.get_optimizer('main')

        batch = self.get_iterator('main').next()

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

        chainer.report({'loss/target': loss_target.data, 'loss/total': loss.data, 'acc': acc}, self.model)

        loss.backward()
        optimizer.update()


class RelativeUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop('model')
        self.vocab = kwargs.pop('vocab')
        self.raw = kwargs.pop('raw')
        super(RelativeUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        xp = self.model.xp
        self.model.cleargrads()
        optimizer = self.get_optimizer('main')

        batch = self.get_iterator('main').next()

        target, provided, captions = self.converter(batch, device=self.device, padding=self.vocab['<eos>'])
        len_caption = captions.shape[1]

        self.model.reset_state()
        v_global, v_local, keys = self.model.encode(target, provided, raw=self.raw)

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
        chainer.report({'loss': loss.data, 'acc': acc}, self.model)

        loss.backward()
        optimizer.update()
