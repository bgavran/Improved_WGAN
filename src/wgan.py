import tensorflow as tf
import numpy as np


class WGAN:
    def __init__(self, generator, critic, z_size, img_size, optimizer=tf.train.RMSPropOptimizer(learning_rate=0.0005),
                 clip_value=0.01):
        self.generator = generator
        self.critic = critic

        self.clip_value = clip_value
        self.optimizer = optimizer

        self.z_size = z_size
        self.img_size = img_size

        self.z = tf.placeholder(tf.float32, [None, self.z_size])
        self.fake_image = self.generator(self.z)
        self.real_image = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 3])

        self.c_real = self.critic(self.real_image)
        self.c_fake = self.critic(self.fake_image, reuse=True)

        # tries to minimize score for fake images and maximize for real
        self.c_cost = tf.reduce_mean(self.c_fake - self.c_real)
        # tries to maximize the score for fake images
        self.g_cost = -tf.reduce_mean(self.c_fake)

        # regulariazion of critic, satisfying the Lipschitz constraint
        self.eta = tf.placeholder(tf.float32, shape=[1])
        interp = self.eta * self.real_image + (1 - self.eta) * self.fake_image
        c_interp = self.critic(interp, reuse=True)
        c_grads = tf.gradients(c_interp, interp)[0]  # taking the zeroth and only element because it returns a list
        slopes = tf.sqrt(tf.reduce_sum(tf.square(c_grads), axis=1))
        grad_penalty = tf.reduce_mean(tf.square(slopes - 1.) ** 2)
        lambd = 10
        self.c_cost += lambd * grad_penalty

        c_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Critic")  # weights for the critic
        self.c_optimizer = self.optimizer.minimize(self.c_cost, var_list=c_variables)
        # self.c_clipper = [var.assign(tf.clip_by_value(var, -self.clip_value, self.clip_value)) for var in c_variables]

        g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Generator")  # weights for the generator
        self.g_optimizer = self.optimizer.minimize(self.g_cost, var_list=g_variables)

        # Adding summaries for tensorflow until the end of the method
        tf.summary.image("Generated image", self.fake_image, max_outputs=4)
        tf.summary.image("Real image", self.real_image, max_outputs=4)
        tf.summary.scalar("Critic cost", self.c_cost)
        tf.summary.scalar("Generator cost", self.g_cost)

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
                if step < 25 or step % 500 == 0:
                    c_times = 25
                else:
                    c_times = 5
                eta = np.random.rand(1)

                for _ in range(c_times):
                    data_batch = data.next_batch_real(hp.batch_size)
                    z = data.next_batch_fake(hp.batch_size, self.z_size)
                    sess.run([self.c_optimizer], feed_dict={self.real_image: data_batch, self.z: z, self.eta: eta})

                z = data.next_batch_fake(hp.batch_size, self.z_size)
                sess.run(self.g_optimizer, feed_dict={self.z: z})

                if step % 50 == 0:
                    n_images = 4
                    data_batch = data.next_batch_real(n_images)
                    z = data.next_batch_fake(n_images, self.z_size)
                    summary = sess.run(merged, feed_dict={self.real_image: data_batch, self.z: z, self.eta: eta})
                    writer.add_summary(summary, step)
                    print("Summary generated. Step", step, " Time == %.2fs" % (time() - start_time))
                    start_time = time()
