import chainer


def normalize(image):

    xp = chainer.cuda.get_array_module(image)

    x = xp.array(image, 'f')
    x = xp.transpose(x, [2, 0, 1])
    x = x * (2. / 255.) - 1.

    return x


def denormalize(x):

    xp = chainer.cuda.get_array_module(x)

    image = xp.array(x, 'uint8')
    image = xp.transpose(image, [1, 2, 0])
    image = xp.clip((image + 1) * (255. / 2.), 0, 255)

    return image
