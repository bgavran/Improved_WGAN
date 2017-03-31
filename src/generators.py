import tensorflow as tf


class FCGenerator:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, z):
        # TODO add batchnorm?
        with tf.variable_scope("Generator"):
            z = tf.layers.dense(z, 512, activation=tf.nn.elu)
            z = tf.layers.dense(z, 512, activation=tf.nn.elu)
            z = tf.layers.dense(z, self.img_size * self.img_size * 3, activation=tf.nn.tanh)
            image = tf.reshape(z, [-1, self.img_size, self.img_size, 3])
            return image
