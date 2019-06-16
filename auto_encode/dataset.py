import chainer
import numpy as np
import random
from PIL import Image
import pickle


class ImageDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_path, angle=1, mode='chair', back='white'):
        self.img_path = data_path + 'images_{}/'.format(back)
        with open(data_path + '/{}_text.pkl'.format(mode), 'rb') as f:
            text_dict = pickle.load(f)
        self.names = sorted(text_dict)
        self.angle = angle

    def __len__(self):
        return len(self.names)

    def get_example(self, i):
        num = self.angle
        if self.angle < 0:
            num = random.choice([0, 1, 2, 6, 7])
        x = self.get_image(i, num)
        return x

    def get_image(self, i, angle):
        base = self.names[i]
        path = self.img_path + base + '/' + base + '-000{}.jpg'.format(angle)
        image = Image.open(path)
        image_64 = image.resize((64, 64))
        x = np.transpose(image_64, [2, 0, 1])
        x = x * (2. / 255.) - 1
        x = x.astype('f')
        return x


class BundleImageDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_path, angle=1, mode='chair', back='white'):
        img_path = data_path + '{}_image_{}.pkl'.format(mode, back)
        with open(img_path, 'rb') as f:
            self.xs = pickle.load(f)
            self.names = sorted(self.xs)
        self.angle = angle

    def __len__(self):
        return len(self.names) // 5

    def get_example(self, i):
        num = {0: 0, 1: 1, 2: 2, 6: 3, 7: 4}[self.angle] if self.angle >= 0 else random.randrange(5)
        index = 5 * i + num
        x = self.xs[self.names[index]]
        return x
