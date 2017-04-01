import tensorflow as tf


class Discriminator:
    def __call__(self, image, reuse=None):
        # TODO add batchnorm?
        with tf.variable_scope("Discriminator", reuse=reuse):
            act = tf.nn.relu
            kernel = (3, 3)
            image = tf.layers.conv2d(image, filters=64, kernel_size=kernel, strides=kernel, use_bias=False,
                                     activation=act, padding="same")
            image = tf.layers.conv2d(image, filters=128, kernel_size=kernel, strides=kernel, use_bias=False,
                                     activation=act, padding="same")
            image = tf.layers.conv2d(image, filters=256, kernel_size=kernel, strides=kernel, use_bias=False,
                                     activation=act, padding="same")
            image = tf.layers.conv2d(image, filters=1024, kernel_size=kernel, strides=kernel, use_bias=False,
                                     padding="same")
            return image
