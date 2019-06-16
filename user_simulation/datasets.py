import numpy as np
import chainer
import pickle
from PIL import Image
import random
import collections
from src.miscs.nlp_utils import make_array


class RelativeCaptionDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_path, back='white', raw=False, angle=1, size=(64, 64), add_eos=True):
        self.raw = raw
        self.size = size
        self.vocab = None
        self.angle = angle
        self.add_eos = add_eos
        text_path = data_path + 'rc_text_norm.pkl'
        with open(text_path, 'rb') as file:
            self.text_dict = pickle.load(file)
            self.name_list = sorted(self.text_dict)
        self.make_vocab()
        self.img_path = data_path + 'images_{}/'.format(back)
        if not raw:
            feat_path = data_path + 'chair_feat_{}.pkl'.format(back)
            with open(feat_path, 'rb') as file:
                self.feature = pickle.load(file)

    def __len__(self):
        return len(self.name_list)

    def get_example(self, i):
        target_name, candid_name = self.name_list[i]
        target_image = self.get_image(target_name)
        candid_image = self.get_image(candid_name)
        text = random.choice(self.text_dict[(target_name, candid_name)])
        text = make_array(text, self.vocab, add_bos=True, add_eos=self.add_eos)
        return target_image, candid_image, text

    def get_image(self, name):
        num = self.angle
        if self.angle < 0:
            num = random.choice([0, 1, 2, 6, 7])
        if self.raw:
            name = '{}{}/{}-000{}.jpg'.format(self.img_path, name, name, num)
            image = Image.open(name)
            image = np.asarray(image)
        else:
            name = '{}-000{}'.format(name, num)
            image = self.feature[name]
        return image

    def make_vocab(self, max_vocab_size=20000, min_freq=3):
        counts = collections.defaultdict(int)
        for _, captions in self.text_dict.items():
            for caption in captions:
                for token in caption:
                    counts[token] += 1

        vocab = {'<bos>': 0, '<eos>': 1, '<unk>': 2}
        for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            if len(vocab) >= max_vocab_size or c < min_freq:
                break
            vocab[w] = len(vocab)
        self.vocab = vocab


class VanillaCaptionDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_path, mode='chair', back='white', raw=False, angle=1, size=(64, 64), add_eos=True):
        self.raw = raw
        self.size = size
        self.vocab = None
        self.angle = angle
        self.add_eos = add_eos
        with open(data_path + '{}_text_norm.pkl'.format(mode), 'rb') as file:
            self.text_dict = pickle.load(file)
            self.name_list = sorted(self.text_dict)
        self.make_vocab()
        self.img_path = data_path + 'images_{}/'.format(back)
        if not raw:
            feat_path = data_path + '{}_feat_{}.pkl'.format(mode, back)
            with open(feat_path, 'rb') as file:
                self.feature = pickle.load(file)

    def __len__(self):
        return len(self.name_list)

    def get_example(self, i):
        name = self.name_list[i]
        image = self.get_image(name)
        text = random.choice(self.text_dict[name])
        text = make_array(text, self.vocab, add_bos=True, add_eos=self.add_eos)
        return image, text

    def get_image(self, name):
        num = self.angle
        if self.angle < 0:
            num = random.choice([0, 1, 2, 6, 7])
        if self.raw:
            name = '{}{}/{}-000{}.jpg'.format(self.img_path, name, name, num)
            image = Image.open(name)
            image = np.array(image)
        else:
            name = '{}-000{}'.format(name, num)
            image = self.feature[name]
        return image

    def make_vocab(self, max_vocab_size=20000, min_freq=3):
        counts = collections.defaultdict(int)
        for _, captions in self.text_dict.items():
            for caption in captions:
                for token in caption:
                    counts[token] += 1

        vocab = {'<bos>': 0, '<eos>': 1, '<unk>': 2}
        for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            if len(vocab) >= max_vocab_size or c < min_freq:
                break
            vocab[w] = len(vocab)
        self.vocab = vocab


class RelativeImageDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_path, back='white', angle=1):
        self.angle = angle
        img_path = data_path + 'chair_image_{}.pkl'.format(back)
        with open(img_path, 'rb') as file:
            self.xs = pickle.load(file)
            self.names = sorted(self.xs)

    def __len__(self):
        return len(self.names) // 5

    def get_example(self, i):
        num = {0: 0, 1: 1, 2: 2, 6: 3, 7: 4}[self.angle] if self.angle >= 0 else random.randrange(5)
        index = 5 * i + num
        name = self.names[index]
        image = self.get_image(name)
        return image

    def get_image(self, name):
        x = self.xs[name]
        x = np.array(np.clip((x + 1) * (255. / 2.), 0, 255), np.uint8)
        image = np.transpose(x, [1, 2, 0])
        return image

    def get_examples(self, iterator):
        images = []
        for i in iterator:
            images.append(self.get_example(i))
        return np.array(images)
