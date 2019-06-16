import pickle
import numpy as np
import random
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

from fine_tune.updaters import ResFineTuneUpdater
from fine_tune.extensions import ResFineTuneEvaluator, res_visualize


def parse_arguments():

    parser = argparse.ArgumentParser(description='State Tracker Trainer')

    parser.add_argument('--batch_size', '-b', type=int, default=64,
                        help='Number of images in each mini-batch.')
    parser.add_argument('--epoch', '-e', type=int, default=50,
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
    parser.add_argument('--distribution', '-d', type=str, default='normal',
                        help='Distribution (normal or vmf)')
    parser.add_argument('--angle', '-a', type=int, default=1,
                        help='Viewpoint of the chair images (random for -1)')
    parser.add_argument('--back', type=str, default='white',
                        help='Background color of the images.')
    parser.add_argument('--object', type=str, default='chair',
                        help='What kind of object are in images.')
    parser.add_argument('--margin', '-m', type=float, default=0.2,
                        help='Margin for triplet loss.')
    parser.add_argument('--c_kld', type=float, default=0.1,
                        help='Coeff for KL Divergence.')
    parser.add_argument('--no_kld', action='store_true',
                        help='Without KL restriction.')

    parser.add_argument('--speaker', '-s', type=str, required=True,
                        help='Which type of speaker for user simulator.')
    parser.add_argument('--caption', '-c', type=str, required=True,
                        help='Caption Dataset Type. (vanilla or relative)')
    parser.add_argument('--sim_param', '-sp', type=str, required=True,
                        help='Model parameter for user simulator.')
    parser.add_argument('--lam', type=float, default=0.5,
                        help='Co-eff for ES beam search')
    parser.add_argument('--r_eos', type=float, default=1.2,
                        help='EOS correction term for modified ES beam search')
    parser.add_argument('--r_dup', type=float, default=0.1,
                        help='Duplication correction term for modified ES beam search')
    parser.add_argument('--beam', type=int, default=10,
                        help='Beam width for caption.')
    parser.add_argument('--n_rand', type=int, default=5,
                        help='Width for selecting caption at random.')

    parser.add_argument('--gen', default='res', type=str,
                        help='Generator type.')
    parser.add_argument('--dis', default='sn_res', type=str,
                        help='Discriminator type.')
    parser.add_argument('--gen_param', '-gp', type=str, required=True,
                        help='Model parameter for generator.')
    parser.add_argument('--dis_param', '-dp', type=str, required=True,
                        help='Model parameter for discriminator.')
    parser.add_argument('--n_case', type=int, default=5,
                        help='Number of cases to be saved.')

    parser.add_argument('--fbr_param', '-fp', type=str, required=True,
                        help='Model parameter for feedback receiver.')

    parser.add_argument('--data_dir', default='/data/unagi0/ono/thesis/sequential_generation_2D/',
                        help='data root directory')
    parser.add_argument('--snapshot_interval', type=int, default=1,
                        help='Interval of snapshot (epoch)')
    parser.add_argument('--plot_interval', type=int, default=10,
                        help='Interval of logging (iteration)')
    parser.add_argument('--display_interval', type=int, default=1,
                        help='Interval of displaying log to console (epoch)')
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

    gen_param_path = args.data_dir + args.gen_param
    dis_param_path = args.data_dir + args.dis_param

    from nets.extractor import Extractor
    enc = Extractor()

    if args.gen == 'res':
        from nets.generators import ResNetGenerator
        gen = ResNetGenerator(dim_z=args.dim_z)
    elif args.gen == 'dc':
        from nets.generators import Generator
        gen = Generator(dim_z=args.dim_z)
    else:
        raise NotImplementedError

    chainer.serializers.load_npz(gen_param_path, gen)

    if args.dis == 'res':
        from nets.discriminators import ResNetProjectionDiscriminator
        dis = ResNetProjectionDiscriminator()
    elif args.dis == 'sn_res':
        from nets.discriminators import SNResNetProjectionDiscriminator
        dis = SNResNetProjectionDiscriminator()
    elif args.dis == 'dc':
        from nets.discriminators import Discriminator
        dis = Discriminator()
    elif args.dis == 'sn_dc':
        from nets.encoders import SNDiscriminator
        dis = SNDiscriminator()
    else:
        raise NotImplementedError

    chainer.serializers.load_npz(dis_param_path, dis)

    return enc, gen, dis


def load_feedback_receiver(args, num_voc):

    fbr_param_path = args.data_dir + args.fbr_param

    from nets.feedback_receiver import FeedbackReceiver

    feedback_receiver = FeedbackReceiver(
        dim_z=args.dim_z,
        dim_txt=args.dim_t,
        num_voc=num_voc,
        dim_emb=args.dim_emb
    )

    chainer.serializers.load_npz(fbr_param_path, feedback_receiver)

    return feedback_receiver


def make_dataset(args):

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
    # ms, vs = compute_z(enc, xs, args.batch_size)

    return {'name': np.array(names), 'x': xs, 'f': fs}


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


def make_optimizer(_model, alpha=1e-5, beta1=0.5, beta2=0.999):
    _optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
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
            ['elapsed_time', 'epoch', 'iteration', 'gen/loss', 'fbr/loss']
        ),
        trigger=(args.display_interval, 'epoch')
    )

    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.extend(
        extensions.PlotReport(
            ['enc/loss', 'val/enc/loss'],
            trigger=(args.plot_interval, 'iteration'),
            file_name='loss_enc.jpg'
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ['gen/loss', 'val/gen/loss'],
            trigger=(args.plot_interval, 'iteration'),
            file_name='loss_gen.jpg'
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ['dis/loss', 'val/dis/loss'],
            trigger=(args.plot_interval, 'iteration'),
            file_name='loss_dis.jpg'
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ['fbr/loss', 'val/fbr/loss'],
            trigger=(args.plot_interval, 'iteration'),
            file_name='loss_fbr.jpg'
        )
    )
    trainer.extend(
       extensions.PlotReport(
           ['val/rank1', 'val/rank2', 'val/rank3', 'val/rank4', 'val/rank5'],
           trigger=(args.plot_interval, 'iteration'),
           file_name='rank.jpg'
       )
    )


def main():

    args = parse_arguments()

    print('Load models.')

    sim, vocab = load_user_simulator(args)
    enc, gen, dis = load_auto_encoder(args)
    fbr = load_feedback_receiver(args, num_voc=len(vocab))

    if args.gpu >= 0:
        print('Convert models to the GPU.')
        chainer.cuda.get_device_from_id(args.gpu).use()
        sim.to_gpu()
        fbr.to_gpu()
        enc.to_gpu()
        gen.to_gpu()
        dis.to_gpu()

    print('Make dataset.')

    dataset = make_dataset(args)
    train_index, test_index, train_data, test_data = split_dataset(dataset)

    # Setup for optimizers
    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)
    opt_fbr = make_optimizer(fbr)

    # Setup for an iterator (fake)
    train_iter = chainer.iterators.SerialIterator(train_index, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test_index, args.batch_size)

    print('Define updater.')

    updater = ResFineTuneUpdater(
        models=(enc, gen, dis, fbr, sim),
        iterator=train_iter,
        optimizer={'gen': opt_gen, 'dis': opt_dis, 'fbr': opt_fbr},
        data=train_data,
        device=args.gpu,
        args=args
    )

    print('Define trainer.')

    out_path = args.data_dir + 'fine_tune/' + args.out

    trainer = training.Trainer(
        updater=updater,
        stop_trigger=(args.epoch, 'epoch'),
        out=out_path
    )

    print('Extend trainer.')

    trainer.extend(
        ResFineTuneEvaluator(
            models=(enc, gen, dis, fbr, sim),
            iterator=test_iter,
            data=test_data,
            device=args.gpu,
            args=args
        ),
        trigger=(args.eval_interval, 'epoch')
    )
    trainer.extend(
        res_visualize(
            models=(enc, gen, dis, fbr, sim),
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
        extensions.snapshot_object(gen, 'gen_{.updater.epoch}'),
        trigger=(args.snapshot_interval, 'epoch')
    )
    trainer.extend(
        extensions.snapshot_object(dis, 'dis_{.updater.epoch}'),
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