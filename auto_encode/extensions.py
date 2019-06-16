import os

import numpy as np
from PIL import Image

import chainer
from chainer.dataset import concat_examples

import chainer.functions as F
from src.functions.vmf import von_mises_fisher


def visualize(enc, gen, iterator, out_path, data_type, device, distribution='normal', rows=5, cols=5, seed=0):

    if data_type == 'random':

        @chainer.training.make_extension()
        def extension(trainer):
            # Prepare input data
            np.random.seed(seed)
            z = gen.make_hidden(batchsize=rows * cols, distribution=distribution)
            np.random.seed()

            # Generate
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                y = gen(z)

            # Save generated images
            y = chainer.cuda.to_cpu(y.data)
            y = np.asarray(np.clip((y + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
            H = 64
            W = 64
            y = y.reshape((rows, cols, 3, H, W))
            y = y.transpose(0, 3, 1, 4, 2)
            y = y.reshape((rows * H, cols * W, 3))

            # Save images
            preview_dir = '{}/preview/'.format(out_path) + '/{}/'.format(data_type)
            preview_path = preview_dir + '{:0>4}.jpg'.format(trainer.updater.epoch)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(y).save(preview_path)

    else:

        @chainer.training.make_extension()
        def extension(trainer):
            # Reconstruct image from z

            # Prepare input data
            np.random.seed(seed)
            index = np.random.choice(len(iterator), rows * cols, replace=False).tolist()
            x_real = concat_examples([iterator.get_example(i) for i in index], device)
            np.random.seed()

            # Generate
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                mu, var = enc(x_real)
                if distribution == 'normal':
                    z = F.gaussian(mu, F.log(var + 1e-10))
                elif distribution == 'vmf':
                    z = von_mises_fisher(mu, var)
                else:
                    raise NotImplementedError
                y = gen(z)

            # Save generated images
            y = chainer.cuda.to_cpu(y.data)
            y = np.asarray(np.clip((y + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
            H = 64
            W = 64
            y = y.reshape((rows, cols, 3, H, W))
            y = y.transpose(0, 3, 1, 4, 2)
            y = y.reshape((rows * H, cols * W, 3))

            # Post process real images in the same way as generated images
            x_real = chainer.cuda.to_cpu(x_real)
            raw_img = np.clip((x_real + 1.) * (255. / 2.), 0.0, 255.0).astype(np.uint8)
            _, _, H, W = raw_img.shape
            H = 64
            W = 64
            raw_img = raw_img.reshape((rows, cols, 3, H, W))
            raw_img = raw_img.transpose(0, 3, 1, 4, 2)
            raw_img = raw_img.reshape((rows * H, cols * W, 3))

            # Save images
            preview_dir = '{}/preview/'.format(out_path) + '/{}/'.format(data_type)
            preview_path = preview_dir + '{:0>4}.jpg'.format(trainer.updater.epoch)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(np.hstack((raw_img, y))).save(preview_path)

    return extension