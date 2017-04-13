import tensorflow as tf


class ConvEncoder:
    def __init__(self, img_size, z_size):
        self.img_size = img_size
        self.z_size = z_size

    def __call__(self, image, reuse=None):
        with tf.variable_scope("Encoder", reuse=reuse):
            act = tf.nn.relu
            bnorm = tf.contrib.layers.batch_norm
            kernel = (3, 3)

            kwargs = {"kernel_size": kernel, "strides": kernel, "use_bias": False, "padding": "valid"}

            image = tf.layers.conv2d(image, filters=64, **kwargs)
            image = act(bnorm(image))
            image = tf.layers.conv2d(image, filters=128, **kwargs)
            image = act(bnorm(image))
            image = tf.layers.conv2d(image, filters=256, **kwargs)
            image = act(bnorm(image))
            image = tf.layers.conv2d(image, filters=self.z_size, **kwargs)

            return image


class FCEncoder:
    def __init__(self, img_size, z_size):
        self.img_size = img_size
        self.z_size = z_size

    def __call__(self, image):
        image = tf.reshape(image, [-1, self.img_size * self.img_size * 3])
        z = tf.layers.dense(image, self.z_size, activation=tf.nn.relu)
        z = tf.layers.dense(z, self.z_size, activation=tf.nn.sigmoid)
        return z
