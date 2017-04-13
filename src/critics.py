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
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, image, reuse=None):
        with tf.variable_scope("Critic", reuse=reuse):
            act = tf.nn.relu
            kernel = (4, 4)
            kwargs = {"kernel_size": kernel, "strides": kernel, "padding": "valid", "activation": act}

            image = tf.layers.conv2d(image, filters=1024, **kwargs)
            image = tf.reshape(image, [-1, 16 * 16 * 1024])
            image = tf.layers.dense(image, 1)

            print(image.shape)
            assert image.shape[1] == 1
            return image
