import tensorflow as tf


class ConvDecoder:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, z, reuse=None):
        with tf.variable_scope("Decoder", reuse=reuse):
            act = tf.nn.relu
            bnorm = tf.contrib.layers.batch_norm
            pad = [[0, 0],
                   [1, 1],
                   [1, 1],
                   [0, 0]]

            kwargs = {"strides": (1, 1), "padding": "valid"}

            z = tf.layers.dense(z, 4096, activation=act)
            z = tf.reshape(z, [-1, 4, 4, 256])

            z = tf.pad(z, pad, mode="SYMMETRIC")
            z = tf.layers.conv2d(z, filters=128, kernel_size=(3, 3), **kwargs)
            z = act(bnorm(z))
            z = tf.image.resize_images(z, (16, 16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            z = tf.pad(z, pad, mode="SYMMETRIC")
            z = tf.layers.conv2d(z, filters=64, kernel_size=(5, 5), **kwargs)
            z = act(bnorm(z))
            z = tf.image.resize_images(z, (64, 64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            z = tf.pad(z, pad, mode="SYMMETRIC")
            z = tf.layers.conv2d(z, filters=16, kernel_size=(5, 5), **kwargs)
            z = act(bnorm(z))
            z = tf.image.resize_images(z, (self.img_size, self.img_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            z = tf.pad(z, pad, mode="SYMMETRIC")
            z = tf.layers.conv2d(z, filters=3, activation=tf.nn.sigmoid, kernel_size=(3, 3), **kwargs)

            return z


class FCDecoder:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, z):
        image = tf.layers.dense(z, 1024, activation=tf.nn.relu)
        image = tf.layers.dense(image, self.img_size * self.img_size * 3, activation=tf.nn.sigmoid)
        image = tf.reshape(image, [-1, self.img_size, self.img_size, 3])
        return image
