import tensorflow as tf


class ToyFCCritic:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, image, reuse=None):
        with tf.variable_scope("Critic", reuse=reuse):
            image = tf.reshape(image, [-1, self.img_size * self.img_size * 3])
            image = tf.layers.dense(image, 512, tf.nn.relu)
            image = tf.layers.dense(image, 1)
            return image


class ConvCritic:
    def __call__(self, image, reuse=None):
        with tf.variable_scope("Critic", reuse=reuse):
            act = tf.nn.relu

            def bnorm(*args):
                return tf.layers.batch_normalization(*args)

            kernel3 = (3, 3)
            kernel4 = (4, 4)
            pad1 = [[0, 0], [1, 1], [1, 1], [0, 0]]
            pad2 = [[0, 0], [2, 2], [2, 2], [0, 0]]

            kwargs3 = {"kernel_size": kernel3, "strides": (1, 1), "padding": "valid"}
            kwargs4 = {"kernel_size": kernel4, "strides": kernel4, "padding": "valid"}

            image = tf.pad(image, pad1, mode="SYMMETRIC")
            image = tf.layers.conv2d(image, filters=128, **kwargs3, kernel_regularizer=bnorm, activation=act)
            # image is 64x64x128
            image = tf.layers.conv2d(image, filters=128, **kwargs4, kernel_regularizer=bnorm, activation=act)
            # image is 16x16x128
            image = tf.pad(image, pad1, mode="SYMMETRIC")
            image = tf.layers.conv2d(image, filters=128, **kwargs3, kernel_regularizer=bnorm, activation=act)
            # image is 16x16x128
            image = tf.layers.conv2d(image, filters=128, **kwargs4, kernel_regularizer=bnorm, activation=act)

            # image is 4x4x128

            image = tf.reshape(image, [-1, 4 * 4 * 128])
            image = tf.layers.dense(image, 1)

            assert image.shape[1] == 1
            # assert image.shape[1:3] == [1, 1]
            # image = tf.squeeze(tf.squeeze(image, axis=3), axis=2)
            return image
