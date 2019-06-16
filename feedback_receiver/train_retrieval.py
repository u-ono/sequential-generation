import pickle
import numpy as np
import random
import tqdm
import argparse
import chainer
from chainer import training
from chainer.training import extensions
import os
import sys
import matplotlib

sys.path.append(os.getcwd())

from misc.set_debugger import set_debugger
set_debugger()

from feedback_receiver.updaters import RetrievalUpdater
from feedback_receiver.extensions import RetrievalEvaluator, visualize_retrieval


def parse_arguments():

    parser = argparse.ArgumentParser(description='State Tracker Trainer')

    parser.add_argument('--batch_size', '-b', type=int, default=64,
                        help='Number of images in each mini-batch.')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train.')
    parser.add_argument('--turn', '-t', type=int, default=5,
                        help='Number of dialogue.')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', required=True,
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')

    parser.add_argument('--dim_z', '-z', type=int, default=256,
                        help='Dimension of z.')
    parser.add_argument('--dim_t', type=int, default=256,
                        help='Dimension of text code.')
    parser.add_argument('--dim_emb', type=int, default=256,
                        help='Dimension of word embeddings.')
    parser.add_argument('--distribution', '-d', type=str, required=True,
                        help='Distribution (normal or vmf)')
    parser.add_argument('--angle', '-a', type=int, default=1,
                        help='Viewpoint of the chair images (random for -1)')
    parser.add_argument('--back', type=str, default='white',
                        help='Background color of the images.')
    parser.add_argument('--object', type=str, default='chair',
                        help='What kind of object are in images.')
    parser.add_argument('--margin', '-m', type=float, default=0.2,
                        help='Margin for triplet loss.')
    parser.add_argument('--c_kld', '-k', type=float, default=0.1,
                        help='Coeff for KL Divergence.')
    parser.add_argument('--no_kld', action='store_true',
                        help='Without KL restriction.')

    parser.add_argument('--speaker', '-s', type=str, required=True,
                        help='Which type of speaker for user simulator.')
    parser.add_argument('--caption', '-c', type=str, required=True,
                        help='Caption Dataset Type. (vanilla or relative)')
    parser.add_argument('--sim_param', '-sp', type=str, required=True,
                        help='Model parameter for user simulator.')
    parser.add_argument('--lam', type=float, default=0.4,
                        help='Co-eff for ES beam search')
    parser.add_argument('--r_eos', type=float, default=1.5,
                        help='EOS correction term for modified ES beam search')
    parser.add_argument('--r_dup', type=float, default=0.5,
                        help='Duplication correction term for modified ES beam search')
    parser.add_argument('--beam', type=int, default=10,
                        help='Beam width for caption.')
    parser.add_argument('--n_rand', type=int, default=5,
                        help='Width for selecting caption at random.')

    parser.add_argument('--enc', default='res', type=str,
                        help='Encoder type.')
    parser.add_argument('--gen', default='res', type=str,
                        help='Generator type.')
    parser.add_argument('--enc_param', '-ep', type=str, required=True,
                        help='Model parameter for encoder.')
    parser.add_argument('--gen_param', '-gp', type=str, required=True,
                        help='Model parameter for generator.')
    parser.add_argument('--n_case', type=int, default=5,
                        help='Number of cases to be saved.')

    parser.add_argument('--data_dir', default='',
                        help='data root directory')
    parser.add_argument('--snapshot_interval', type=int, default=10,
                        help='Interval of snapshot (epoch)')
    parser.add_argument('--plot_interval', type=int, default=10,
                        help='Interval of logging (iteration)')
    parser.add_argument('--display_interval', type=int, default=1,
                        help='Interval of displaying log to console (iteration)')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Interval of evaluation (epoch)')

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Mini-batch: {}'.format(args.batch_size))
    print('# Epoch: {}'.format(args.epoch))
    print('# Turn: {}'.format(args.turn))
    print('Angle: {}'.format(args.angle))
    print('margin: {}'.format(args.margin))
    print('KL coeff: {}'.format(args.c_kld))
    print('')

    return args


def load_user_simulator(args):

    data_path = args.data_dir + 'data/'

    if args.caption == 'vanilla':
        from user_simulation.datasets import VanillaCaptionDataset
        caption = VanillaCaptionDataset(
            data_path=data_path,
            back=args.back,
            mode=args.object,
            angle=args.angle,
            raw=True
        )
    elif args.caption == 'relative':
        from user_simulation.datasets import RelativeCaptionDataset
        caption = RelativeCaptionDataset(
            data_path=data_path,
            back=args.back,
            angle=args.angle,
            raw=True
        )
    else:
        raise NotImplementedError

    vocab = caption.vocab

    param_path = args.data_dir + args.sim_param

    if args.speaker == 'es':
        from nets.es_speaker import ESSpeaker
        sim = ESSpeaker(vocab=vocab)
        chainer.serializers.load_npz(param_path, sim.emitter)
        chainer.serializers.load_npz(param_path, sim.suppressor)
    elif args.speaker == 'rc':
        from nets.relative_captioner import RelativeCaptioner
        sim = RelativeCaptioner(vocab=vocab)
        chainer.serializers.load_npz(param_path, sim)
    else:
        raise NotImplementedError

    return sim, vocab


def load_auto_encoder(args):

    enc_param_path = args.data_dir + args.enc_param
    gen_param_path = args.data_dir + args.gen_param

    if args.gen == 'res':
        from nets.generators import ResNetGenerator
        gen = ResNetGenerator(dim_z=args.dim_z)
    elif args.gen == 'dc':
        from nets.generators import Generator
        gen = Generator(dim_z=args.dim_z)
    else:
        raise NotImplementedError

    chainer.serializers.load_npz(gen_param_path, gen)

    if args.enc == 'res':
        from nets.encoders import ResNetEncoder
        enc = ResNetEncoder(dim_z=args.dim_z, distribution=args.distribution)
    elif args.enc == 'sn_res':
        from nets.encoders import SNResNetEncoder
        enc = SNResNetEncoder(dim_z=args.dim_z, distribution=args.distribution)
    elif args.enc == 'dc':
        from nets.encoders import Encoder
        enc = Encoder(dim_z=args.dim_z, distribution=args.distribution)
    elif args.enc == 'sn_dc':
        from nets.encoders import SNEncoder
        enc = SNEncoder(dim_z=args.dim_z, distribution=args.distribution)
    else:
        raise NotImplementedError

    chainer.serializers.load_npz(enc_param_path, enc)

    return enc, gen


def load_feedback_receiver(args, num_voc):

    from nets.feedback_receiver import FeedbackReceiver

    feedback_receiver = FeedbackReceiver(
        dim_z=args.dim_z,
        dim_txt=args.dim_t,
        num_voc=num_voc,
        dim_emb=args.dim_emb
    )

    return feedback_receiver


def make_dataset(args, enc):

    image_path = args.data_dir + 'data/{}_image_{}.pkl'.format(args.object, args.back)
    with open(image_path, 'rb') as file:
        images = pickle.load(file)

    feat_path = args.data_dir + 'data/{}_feat_{}.pkl'.format(args.object, args.back)
    with open(feat_path, 'rb') as file:
        feat = pickle.load(file)

    names = sorted(images)

    num = {0: 0, 1: 1, 2: 2, 6: 3, 7: 4}[args.angle] if args.angle >= 0 else random.randrange(5)
    num_data = len(names) // 5
    names = [names[5 * i + num] for i in range(num_data)]

    xs = np.array([images[name] for name in names])
    fs = np.array([feat[name] for name in names])
    ms, vs = compute_z(enc, xs, args.batch_size)

    return {'name': np.array(names), 'x': xs, 'm': ms, 'v': vs, 'f': fs}


def compute_z(enc, xs, batchsize):

    xp = enc.xp
    iterator = chainer.iterators.SerialIterator(xs, batchsize, repeat=False, shuffle=False)

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        for i, batch in tqdm.tqdm(enumerate(iterator)):
            m, v = enc(xp.array(batch))
            if i == 0:
                ms = chainer.cuda.to_cpu(m.data)
                vs = chainer.cuda.to_cpu(v.data)
            else:
                ms = np.vstack((ms, chainer.cuda.to_cpu(m.data)))
                vs = np.vstack((vs, chainer.cuda.to_cpu(v.data)))

    return ms, vs


def split_dataset(dataset):

    num_data = len(dataset['name'])
    num_train = int(0.8 * num_data)

    np.random.seed(0)
    data_index = np.random.choice(num_data, num_data, replace=False)
    np.random.seed()

    train_index = data_index[:num_train]
    test_index = data_index[num_train:]

    train_data = {}
    test_data = {}

    for k, v in dataset.items():
        train_data[k] = v[train_index]
        test_data[k] = v[test_index]

    return train_index, test_index, train_data, test_data


def make_optimizer(_model):
    _optimizer = chainer.optimizers.Adam()
    _optimizer.setup(_model)
    return _optimizer


def report_train_info(trainer, args):

    trainer.extend(
        extensions.LogReport(
            trigger=(args.display_interval, 'iteration')
        )
    )
    trainer.extend(
        extensions.PrintReport(
            ['elapsed_time', 'epoch', 'iteration', 'main/loss/total', 'main/rank5']
        ),
        trigger=(args.display_interval, 'iteration')
    )

    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        extensions.PlotReport(
            ['main/loss/total', 'main/loss/triplet', 'main/loss/kl'],
            trigger=(args.plot_interval, 'iteration'),
            file_name='loss_train.png'
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ['val/loss/total', 'val/loss/triplet', 'val/loss/kl'],
            trigger=(args.plot_interval, 'iteration'),
            file_name='loss_test.png'
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/rank1', 'main/rank2', 'main/rank3', 'main/rank4', 'main/rank5'],
            trigger=(args.plot_interval, 'iteration'),
            file_name='rank_train.png'
        )
    )
    trainer.extend(
       extensions.PlotReport(
           ['val/rank1', 'val/rank2', 'val/rank3', 'val/rank4', 'val/rank5'],
           trigger=(args.plot_interval, 'iteration'),
           file_name='rank_test.png'
       )
    )


def main():

    args = parse_arguments()

    print('Load models.')

    sim, vocab = load_user_simulator(args)
    enc, gen = load_auto_encoder(args)
    fbr = load_feedback_receiver(args, num_voc=len(vocab))

    if args.gpu >= 0:
        print('Convert models to the GPU.')
        chainer.cuda.get_device_from_id(args.gpu).use()
        sim.to_gpu()
        fbr.to_gpu()
        enc.to_gpu()
        gen.to_gpu()

    print('Make dataset.')

    dataset = make_dataset(args, enc)
    train_index, test_index, train_data, test_data = split_dataset(dataset)

    # Setup for optimizers
    optimizer = make_optimizer(fbr)

    # Setup for an iterator (fake)
    train_iter = chainer.iterators.SerialIterator(train_index, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test_index, args.batch_size)

    print('Define updater.')

    updater = RetrievalUpdater(
        models=(fbr, sim),
        iterator=train_iter,
        optimizer=optimizer,
        data=train_data,
        device=args.gpu,
        args=args
    )

    print('Define trainer.')

    out_path = args.data_dir + 'retrieval/' + args.out

    trainer = training.Trainer(
        updater=updater,
        stop_trigger=(args.epoch, 'epoch'),
        out=out_path
    )

    print('Extend trainer.')

    trainer.extend(
        RetrievalEvaluator(
            models=(fbr, sim),
            iterator=test_iter,
            data=test_data,
            device=args.gpu,
            args=args
        ),
        trigger=(args.eval_interval, 'epoch')
    )
    trainer.extend(
        visualize_retrieval(
            models=(fbr, sim, gen),
            data=test_data,
            args=args,
            out=out_path
        ),
        trigger=(args.eval_interval,  'epoch')
    )

    report_train_info(trainer, args)

    trainer.extend(
        extensions.snapshot_object(fbr, 'fbr_{.updater.epoch}'),
        trigger=(args.snapshot_interval, 'epoch')
    )
    trainer.extend(
        extensions.snapshot(filename='snapshot_{.updater.epoch}'),
        trigger=(args.snapshot_interval, 'epoch')
    )

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    print('Run the trainer.')

    trainer.run()


if __name__ == '__main__':
    matplotlib.use('Agg')
    main()
