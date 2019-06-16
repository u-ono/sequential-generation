import os
import sys
import argparse
import chainer
from chainer import training
from chainer.training import extensions
import matplotlib
import pickle

matplotlib.use('Agg')
sys.path.append(os.getcwd())

from misc.set_debugger import set_debugger

set_debugger()

from nets.es_speaker import Speaker
from user_simulation.updaters import VanillaUpdater
from user_simulation.extensions import VanillaEvaluator, visualize
from user_simulation.datasets import VanillaCaptionDataset, RelativeImageDataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training of user simulator')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', required=True,
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--raw', action='store_true',
                        help='Raw input or not (default: False)')
    parser.add_argument('--cut_eos', action='store_true',
                        help='Cut down <eos>.')
    parser.add_argument('--angle', '-a', type=int, default=1,
                        help='Viewpoint of the chair images (random for -1)')
    parser.add_argument('--mode', '-m', type=str, default='chair',
                        help='Data mode (chair, table, or all)')
    parser.add_argument('--back', type=str, default='white',
                        help='Background color.')
    parser.add_argument('--snapshot_interval', type=int, default=10,
                        help='Interval of snapshot (epoch)')
    parser.add_argument('--plot_interval', type=int, default=100,
                        help='Interval of logging (iteration)')
    parser.add_argument('--display_interval', type=int, default=1,
                        help='Interval of displaying log to console (iteration)')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Interval of evaluation. (epoch)')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--row', type=int, default=5,
                        help='Row for visualization')
    parser.add_argument('--col', type=int, default=5,
                        help='Col for visualization')
    parser.add_argument('--beam', type=int, default=10,
                        help='Beam width for modified ES beam search.')
    parser.add_argument('--model', type=str, default='es',
                        help='Do not change this.')
    _args = parser.parse_args()

    print('GPU: {}'.format(_args.gpu))
    print('# Mini-batch: {}'.format(_args.batch_size))
    print('# Epoch: {}'.format(_args.epoch))
    print('Angle: {}'.format(_args.angle))
    print('')

    return _args


def make_optimizer(_model):
    _optimizer = chainer.optimizers.Adam()
    _optimizer.setup(_model)
    return _optimizer


def main():

    args = parse_arguments()

    PATH = '/data/unagi0/ono/thesis/sequential_generation_2D/'
    DATA_PATH = PATH + 'data/'
    OUT_PATH = PATH + 'caption/' + args.out

    print('Load the datasets.')
    dataset = VanillaCaptionDataset(
        data_path=DATA_PATH,
        mode=args.mode,
        back=args.back,
        angle=args.angle,
        raw=args.raw,
        add_eos=not args.cut_eos
    )
    vocab = dataset.vocab
    train, test = chainer.datasets.split_dataset_random(dataset, int(len(dataset)*0.8), seed=args.seed)

    images = RelativeImageDataset(
        data_path=DATA_PATH,
        back=args.back,
        angle=args.angle
    )

    print('Define the model.')
    model = Speaker(
        vocab=vocab
    )

    # Convert the models to GPU
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Setup for optimizers
    optimizer = make_optimizer(model)

    # Setup for an iterator
    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size, repeat=False, shuffle=False)

    # Setup for an updater
    updater = VanillaUpdater(
        model=model,
        vocab=vocab,
        raw=args.raw,
        iterator=train_iter,
        optimizer=optimizer,
        device=args.gpu
    )

    # Setup for a trainer
    trainer = training.Trainer(
        updater=updater,
        stop_trigger=(args.epoch, 'epoch'),
        out=OUT_PATH
    )
    trainer.extend(
        VanillaEvaluator(
            model=model,
            vocab=vocab,
            raw=args.raw,
            iterator=test_iter,
            device=args.gpu
        ),
        trigger=(args.eval_interval, 'epoch')
    )
    trainer.extend(
        visualize(
            model=model,
            images=images,
            out_path=OUT_PATH,
            args=args
        ),
        trigger=(args.eval_interval, 'epoch')
    )

    trainer.extend(
        extensions.LogReport(
            trigger=(args.display_interval, 'iteration')
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/loss', 'val/loss'],
            trigger=(args.plot_interval, 'iteration'),
            file_name='loss.png'
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/acc', 'val/acc'],
            trigger=(args.plot_interval, 'iteration'),
            file_name='acc.png'
        )
    )
    trainer.extend(
        extensions.PrintReport(
            ['elapsed_time', 'epoch', 'iteration', 'main/loss', 'main/acc']
        ),
        trigger=(args.display_interval, 'iteration')
    )
    trainer.extend(
        extensions.snapshot_object(model, 'sim_{.updater.epoch}'),
        trigger=(args.snapshot_interval, 'epoch')
    )
    trainer.extend(
        extensions.snapshot(filename='snapshot_{.updater.epoch}'),
        trigger=(args.snapshot_interval, 'epoch')
    )
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
