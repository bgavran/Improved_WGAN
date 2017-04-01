import tensorflow as tf
import numpy as np


class WGAN:
    def __init__(self, generator, discriminator, z_size, img_size, optimizer=tf.train.RMSPropOptimizer, clip_value=0.01,
                 learning_rate=0.00005):
        self.generator = generator
        self.discriminator = discriminator

        self.z_size = z_size
        self.img_size = img_size
        self.x = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 3])
        self.z = tf.placeholder(tf.float32, [None, self.z_size])

        self.clip_value = clip_value
        self.optimizer = optimizer
        self.lr = learning_rate

        self.real_image = self.x
        self.fake_image = self.generator(self.z)
        tf.summary.image("Generated image", self.fake_image, max_outputs=4)
        tf.summary.image("Real image", self.real_image, max_outputs=4)

        self.c_real = self.discriminator(self.real_image)
        self.c_fake = self.discriminator(self.fake_image, reuse=True)

        self.c_cost = tf.reduce_mean(self.c_fake - self.c_real)  # here we only update the critic?
        self.g_cost = -tf.reduce_mean(self.c_fake)  # here we only update the generator
        tf.summary.scalar("Critic cost", self.c_cost)
        tf.summary.scalar("Generator cost", self.g_cost)

        c_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Discriminator")
        self.c_optimizer = self.optimizer(self.lr).minimize(self.c_cost, var_list=c_variables)
        self.c_clipper = [var.assign(tf.clip_by_value(var, -self.clip_value, self.clip_value)) for var in c_variables]

        g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Generator")
        self.g_optimizer = self.optimizer(learning_rate=self.lr).minimize(self.g_cost, var_list=g_variables)

        for i, var in enumerate(g_variables):
            tf.summary.histogram("Generator " + str(i), var)
        for i, var in enumerate(c_variables):
            tf.summary.histogram("Discriminator " + str(i), var)

        grads = self.optimizer(learning_rate=self.lr).compute_gradients(self.c_cost, var_list=c_variables)

        from tensorflow.python.framework import ops
        for gradient, variable in grads:
            if isinstance(gradient, ops.IndexedSlices):
                grad_values = gradient.values
            else:
                grad_values = gradient
            tf.summary.histogram(variable.name, variable)
            tf.summary.histogram(variable.name + "/gradients", grad_values)

    def run_session(self, data, hp):
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(hp.path, sess.graph)
            tf.global_variables_initializer().run()

            g_times = 1
            from time import time
            start_time = time()
            for step in range(hp.steps):
                print("Step", step, end="  ")
                if step < 25 or step % 500 == 0:
                    c_times = 25
                else:
                    c_times = 5

                for _ in range(c_times):
                    print("c" + str(_) + " ", end="")
                    data_batch = data.next_batch_real(hp.batch_size)
                    z = data.next_batch_fake(hp.batch_size, self.z_size)
                    sess.run([self.c_optimizer, self.c_clipper], feed_dict={self.x: data_batch, self.z: z})

                for _ in range(g_times):
                    print("g" + str(_) + " ")
                    z = data.next_batch_fake(hp.batch_size, self.z_size)
                    sess.run(self.g_optimizer, feed_dict={self.z: z})

                if step % 10 == 0:
                    x = data.next_batch_real(hp.batch_size)
                    z = data.next_batch_fake(hp.batch_size, self.z_size)
                    summary = sess.run(merged, feed_dict={self.x: x, self.z: z})
                    writer.add_summary(summary, step)
                    print("Summary generated. Time == ", time() - start_time)
                    start_time = time()
