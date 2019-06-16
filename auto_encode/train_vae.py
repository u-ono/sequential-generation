import os
import sys
import chainer
from chainer import training
from chainer.training import extensions
import argparse
import matplotlib

sys.path.append(os.getcwd())
matplotlib.use('Agg')

from misc.set_debugger import set_debugger
set_debugger()

from auto_encode.dataset import BundleImageDataset
from auto_encode.updaters import VAEUpdater
from auto_encode.extensions import visualize


def parse_arguments():

    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=3000, type=int,
                        help='number of epochs (default: 600)')
    parser.add_argument('--angle', '-a', default=1, type=int,
                        help='Viewpoint of the chair image.')
    parser.add_argument('--mode', '-m', default='chair', type=str,
                        help='Data mode (chair, table, or all)')
    parser.add_argument('--back', default='white', type=str,
                        help='Background color.')
    parser.add_argument('--enc', default='res', type=str,
                        help='Encoder type.')
    parser.add_argument('--gen', default='res', type=str,
                        help='Generator type.')
    parser.add_argument('--dis', default='res', type=str,
                        help='Discriminator type.')
    parser.add_argument('--distribution', '-d', required=True, type=str,
                        help='Distribution type.')
    parser.add_argument('--dim_z', '-z', default=128, type=int,
                        help='Dimension of latent noise.')
    parser.add_argument('--size', default=64, type=int,
                        help='Input and Output image size.')
    parser.add_argument('--coeff', '-c', default=1, type=float,
                        help='Coeff for reconstruction.')
    parser.add_argument('--seed', '-s', default=0, type=int,
                        help='Random Seed')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='learning minibatch size (default: 64)')
    parser.add_argument('--log_interval', '-li', type=int, default=1,
                        help='log interval epoch (default: 1)')
    parser.add_argument('--display_interval', '-di', type=int, default=10,
                        help='display interval iteration (default: 10)')
    parser.add_argument('--visualize_interval', '-vi', type=int, default=1,
                        help='visualize interval epoch (default: 1)')
    parser.add_argument('--snapshot_interval', '-si', type=int, default=100,
                        help='snapshot interval epoch (default: 100)')
    parser.add_argument('--out', '-o', required=True,
                        help='Output directory')
    parser.add_argument('--data_dir', default='',
                        help='data root directory')
    parser.add_argument('--adam_decay_epoch', '-ade', type=int, default=50,
                        help='adam decay epoch (default: 50)')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('dim z: {}'.format(args.dim_z))
    print('Angle: {}'.format(args.angle))
    print('')

    return args


def main():

    # parse arguments from command line
    args = parse_arguments()

    # Define the paths
    DATA_PATH = args.data_dir + 'data/'
    OUT_PATH = args.data_dir + 'vae/' + args.out

    # Load the dataset
    dataset = BundleImageDataset(DATA_PATH, args.angle, mode=args.mode, back=args.back)
    train, test = chainer.datasets.split_dataset_random(dataset, int(0.9 * len(dataset)), args.seed)

    # Define the iterator
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Define the models
    print('start to define the model.')
    if args.gen == 'res':
        from nets.generators import ResNetGenerator
        gen = ResNetGenerator(dim_z=args.dim_z)
    elif args.gen == 'dc':
        from nets.generators import Generator
        gen = Generator(dim_z=args.dim_z)
    else:
        raise NotImplementedError

    if args.enc == 'res' and args.distribution == 'vmf':
        from nets.encoders import ResNetHSEncoder
        enc = ResNetHSEncoder(dim_z=args.dim_z)
    elif args.enc == 'res' and args.distribution == 'normal':
        from nets.encoders import ResNetEncoder
        enc = ResNetEncoder(dim_z=args.dim_z)
    elif args.enc == 'dc' and args.distribution == 'vmf':
        from nets.encoders import HSEncoder
        enc = HSEncoder(dim_z=args.dim_z)
    elif args.enc == 'dc' and args.distribution == 'normal':
        from nets.encoders import Encoder
        enc = Encoder(dim_z=args.dim_z)
    else:
        raise NotImplementedError
    print('done.')

    # Convert the models into GPU version
    if args.gpu >= 0:
        print('convert the models to GPU.')
        chainer.cuda.get_device_from_id(args.gpu).use()
        enc.to_gpu()
        gen.to_gpu()
        print('done.')

    # Set up optimizers
    opt_enc = make_optimizer(enc, alpha=1e-3, beta1=0.9)
    opt_gen = make_optimizer(gen, alpha=1e-3, beta1=0.9)

    # Set up updater
    updater = VAEUpdater(
        models=(enc, gen),
        iterator=train_iter,
        optimizer={'enc': opt_enc, 'gen': opt_gen},
        distribution=args.distribution,
        device=args.gpu,
        coeff=args.coeff
    )

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=OUT_PATH)

    # intervals
    display_interval = (args.display_interval, 'iteration')
    snapshot_interval = (args.snapshot_interval, 'epoch')
    visualize_interval = (args.visualize_interval, 'epoch')
    log_interval = (args.log_interval, 'epoch')

    # Report log
    trainer.extend(extensions.LogReport(trigger=log_interval))

    # Plot losses
    trainer.extend(extensions.PlotReport(
        ['enc/loss', 'gen/loss'],
        trigger=visualize_interval,
        file_name='loss.png'
    ))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'gen/loss', 'enc/loss']),
        trigger=display_interval
    )
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Snapshot objects
    trainer.extend(
        extensions.snapshot(filename='snapshot_{.updater.epoch}'),
        trigger=snapshot_interval
    )
    trainer.extend(
        extensions.snapshot_object(gen, 'gen_{.updater.epoch}.npz'),
        trigger=snapshot_interval
    )
    trainer.extend(
        extensions.snapshot_object(enc, 'enc_{.updater.epoch}.npz'),
        trigger=snapshot_interval
    )

    # Visualize
    trainer.extend(
        visualize(enc, gen, train, OUT_PATH, 'train', args.gpu, distribution=args.distribution),
        trigger=visualize_interval
    )
    trainer.extend(
        visualize(enc, gen, test, OUT_PATH, 'test', args.gpu, distribution=args.distribution),
        trigger=visualize_interval
    )
    trainer.extend(
        visualize(enc, gen, None, OUT_PATH, 'random', args.gpu, distribution=args.distribution),
        trigger=visualize_interval
    )

    # Learning rate decay
    if args.adam_decay_epoch:
        trainer.extend(
            extensions.ExponentialShift("alpha", 0.5, optimizer=opt_gen),
            trigger=(args.adam_decay_epoch, 'epoch')
        )
        trainer.extend(
            extensions.ExponentialShift("alpha", 0.5, optimizer=opt_enc),
            trigger=(args.adam_decay_epoch, 'epoch')
        )

    # Loading trainer resume if specified
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


def make_optimizer(model, alpha=1e-4, beta1=0.5, beta2=0.999, epsilon=1e-8):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2, eps=epsilon)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
    optimizer.add_hook(chainer.optimizer.GradientClipping(threshold=5))
    return optimizer


if __name__ == '__main__':
    main()
