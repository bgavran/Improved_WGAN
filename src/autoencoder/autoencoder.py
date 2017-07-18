import tensorflow as tf
import numpy as np


class Autoencoder:
    def __init__(self, encoder, decoder, img_size, optimizer=tf.train.RMSPropOptimizer, learning_rate=0.00005):
        self.encoder = encoder
        self.decoder = decoder

        self.img_size = img_size
        self.real_image = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 3])

        self.optimizer = optimizer
        self.lr = learning_rate

        self.z = self.encoder(self.real_image)
        self.fake_image = self.decoder(self.z)

        self.cost = tf.reduce_mean((self.fake_image - self.real_image)**2)

        self.optimizer = self.optimizer(self.lr).minimize(self.cost)

        # Adding summaries for tensorflow until the end of the method
        tf.summary.image("Generated image", self.fake_image, max_outputs=4)
        tf.summary.image("Real image", self.real_image, max_outputs=4)
        tf.summary.scalar("AE cost", self.cost)

        # grads = self.optimizer(learning_rate=self.lr).compute_gradients(self.c_cost)
        #
        # from tensorflow.python.framework import ops
        # for gradient, variable in grads:
        #     if isinstance(gradient, ops.IndexedSlices):
        #         grad_values = gradient.values
        #     else:
        #         grad_values = gradient
        #     tf.summary.histogram(variable.name, variable)
        #     tf.summary.histogram(variable.name + "/gradients", grad_values)

    def run_session(self, data, hp):
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(hp.path, sess.graph)
            tf.global_variables_initializer().run()

            from time import time
            start_time = time()
            for step in range(hp.steps):
                data_batch = data.next_batch_real(hp.batch_size)
                sess.run([self.optimizer], feed_dict={self.real_image: data_batch})

                if step % 100 == 0:
                    n_images = 4
                    data_batch = data.next_batch_real(n_images)
                    summary = sess.run(merged, feed_dict={self.real_image: data_batch})
                    writer.add_summary(summary, step)
                    print("Summary generated. Step", step, " Time == %.2fs" % (time() - start_time))
                    start_time = time()
