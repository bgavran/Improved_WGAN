import tensorflow as tf
import numpy as np


class WGAN:
    max_summary_images = 4

    def __init__(self,
                 generator,
                 critic,
                 z_size,
                 channels,
                 optimizer=tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9)):

        self.generator = generator
        self.critic = critic

        self.optimizer = optimizer
        self.z_size = z_size
        self.channels = channels

        # z shape is [batch_size, z_size]
        self.z = tf.placeholder(tf.float32, [None, self.z_size], name="Z")
        # image shape is [batch_size, height, width, channels]
        self.fake_image = self.generator(self.z)
        self.real_image = tf.placeholder(tf.float32,
                                         [None, None, None, channels],
                                         name="Real_image")

        self.c_real = self.critic(self.real_image)
        self.c_fake = self.critic(self.fake_image, reuse=True)

        # tries to minimize the score for fake images
        self.g_cost = tf.reduce_mean(self.c_fake)
        # tries to minimize score for real images and maximize for fake
        self.c_cost = tf.reduce_mean(self.c_real - self.c_fake)

        # Critic regularization, satisfying the Lipschitz constraint with gradient penalty
        with tf.name_scope("Gradient_penalty"):
            self.eta = tf.placeholder(tf.float32, shape=[None, 1, 1, 1])
            interp = self.eta * self.real_image + (1 - self.eta) * self.fake_image
            c_interp = self.critic(interp, reuse=True)

            # taking the zeroth and only element because tf.gradients returns a list
            c_grads = tf.gradients(c_interp, interp)[0]

            # L2 norm, reshaping to [batch_size]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(c_grads), axis=[1, 2, 3]))

            grad_penalty = tf.reduce_mean(tf.square(slopes - 1) ** 2)
            lambd = 10
            self.c_cost += lambd * grad_penalty

        c_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Critic")  # weights for the critic
        self.c_optimizer = self.optimizer.minimize(self.c_cost, var_list=c_variables, name="Critic_optimizer")

        g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Generator")  # weights for the generator
        self.g_optimizer = self.optimizer.minimize(self.g_cost, var_list=g_variables, name="Generator_optimizer")

        # Adding summaries for tensorflow until the end of the method
        tf.summary.image("Generated image", self.fake_image, max_outputs=WGAN.max_summary_images)
        tf.summary.image("Real image", self.real_image, max_outputs=WGAN.max_summary_images)
        tf.summary.scalar("Critic cost", self.c_cost)
        tf.summary.scalar("Generator cost", self.g_cost)

        # Gradient summary
        from tensorflow.python.framework import ops
        for gradient, variable in self.optimizer.compute_gradients(self.c_cost):
            if isinstance(gradient, ops.IndexedSlices):
                grad_values = gradient.values
            else:
                grad_values = gradient
            tf.summary.histogram(variable.name, variable)
            tf.summary.histogram(variable.name + "/gradients", grad_values)

    def run_session(self, task, hp):
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(hp.path, sess.graph)
            tf.global_variables_initializer().run()

            from time import time
            start_time = time()
            for step in range(hp.steps):
                if step < 25:
                    c_times = 100
                else:
                    c_times = 10
                eta = np.random.rand(hp.batch_size, 1, 1, 1)  # sampling from uniform distribution

                for _ in range(c_times):
                    data_batch = task.next_batch_real(hp.batch_size)
                    z = task.next_batch_fake(hp.batch_size, self.z_size)
                    sess.run(self.c_optimizer, feed_dict={self.real_image: data_batch, self.z: z, self.eta: eta})

                z = task.next_batch_fake(hp.batch_size, self.z_size)
                sess.run(self.g_optimizer, feed_dict={self.z: z})

                if step % 100 == 0:
                    data_batch = task.next_batch_real(WGAN.max_summary_images)
                    z = task.next_batch_fake(WGAN.max_summary_images, self.z_size)
                    eta = np.random.rand(WGAN.max_summary_images, 1, 1, 1)

                    summary = sess.run(merged, feed_dict={self.real_image: data_batch, self.z: z, self.eta: eta})
                    writer.add_summary(summary, step)
                    print("Summary generated. Step", step, " Time == %.2fs" % (time() - start_time))
                    start_time = time()
