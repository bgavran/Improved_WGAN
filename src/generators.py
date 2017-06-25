import tensorflow as tf


class FCGenerator:
    def __init__(self, img_size, channels):
        self.img_size = img_size
        self.channels = channels

    def __call__(self, z):
        """
        :param z: tensor for latent space in shape of [batch_size, z_size]
        :return: 
        """
        with tf.variable_scope("Generator"):
            z = tf.layers.dense(z, 512, activation=tf.nn.relu)
            z = tf.layers.dense(z, 512, activation=tf.nn.relu)
            z = tf.layers.dense(z, self.img_size * self.img_size * self.channels, activation=tf.nn.sigmoid)
            image = tf.reshape(z, [-1, self.img_size, self.img_size, self.channels])
            return image


class ConvGenerator:
    def __init__(self, img_size, channels):
        self.img_size = img_size
        self.channels = channels

    def __call__(self, z):
        with tf.variable_scope("Generator"):
            act = tf.nn.relu
            res_met = tf.image.ResizeMethod.NEAREST_NEIGHBOR
            pad2 = [[0, 0], [2, 2], [2, 2], [0, 0]]

            kwargs = {"strides": (1, 1), "padding": "valid"}

            z = tf.layers.dense(z, 32768, activation=act)
            z = tf.reshape(z, [-1, 4, 4, 2048])

            z = tf.pad(z, pad2, mode="SYMMETRIC")
            z = tf.layers.conv2d(z, filters=1024, kernel_size=(5, 5), **kwargs, activation=act)
            z = tf.image.resize_images(z, (16, 16), method=res_met)
            #
            z = tf.pad(z, pad2, mode="SYMMETRIC")
            z = tf.layers.conv2d(z, filters=512, kernel_size=(5, 5), **kwargs, activation=act)
            z = tf.image.resize_images(z, (32, 32), method=res_met)

            z = tf.pad(z, pad2, mode="SYMMETRIC")
            z = tf.layers.conv2d(z, filters=256, kernel_size=(5, 5), **kwargs, activation=act)
            z = tf.image.resize_images(z, (self.img_size, self.img_size), method=res_met)

            z = tf.pad(z, pad2, mode="SYMMETRIC")
            z = tf.layers.conv2d(z, filters=3, activation=tf.nn.sigmoid, kernel_size=(5, 5), **kwargs)
            return z


class DCGANGenerator:
    def __init__(self, img_size, channels):
        self.channels = channels

    def __call__(self, z):
        """

        :param z:
        :return: returns tensor of shape [batch_size, 64, 64, channels]
        """
        with tf.variable_scope("Generator"):
            act = tf.nn.relu

            z = tf.layers.dense(z, 32768, activation=act)
            z = tf.reshape(z, [-1, 4, 4, 2048])

            kwargs = {"kernel_size": (5, 5), "strides": (2, 2), "padding": "same"}

            z = tf.layers.conv2d_transpose(z, filters=512, activation=act, **kwargs)
            z = tf.layers.conv2d_transpose(z, filters=256, activation=act, **kwargs)
            z = tf.layers.conv2d_transpose(z, filters=128, activation=act, **kwargs)
            z = tf.layers.conv2d_transpose(z, filters=self.channels, activation=tf.nn.sigmoid, **kwargs)
            return z
