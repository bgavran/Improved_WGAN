import tensorflow as tf
import numpy as np


class WGAN:
    def __init__(self, generator, discriminator, z_size, img_size, optimizer=tf.train.AdamOptimizer, clip_value=0.01):
        self.generator = generator
        self.discriminator = discriminator

        self.z_size = z_size
        self.img_size = img_size
        self.x = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 3])
        self.z = tf.placeholder(tf.float32, [None, self.z_size])

        self.clip_value = clip_value
        self.optimizer = optimizer

        self.real_image = self.x
        self.fake_image = self.generator(self.z)
        self.summary_generated_image = tf.summary.image("Generated image", self.fake_image, max_outputs=4)
        self.summary_real_image = tf.summary.image("Real image", self.real_image, max_outputs=4)

        self.c_real = self.discriminator(self.real_image)
        self.c_fake = self.discriminator(self.fake_image, reuse=True)

        self.c_cost = tf.reduce_sum(self.c_real - self.c_fake)  # here we only update the critic?
        self.g_cost = -tf.reduce_sum(self.c_fake)  # here we only update the generator
        tf.summary.scalar("Critic cost", self.c_cost)
        tf.summary.scalar("Generator cost", self.g_cost)

        self.c_optimizer = self.optimizer()
        gvs = self.c_optimizer.compute_gradients(self.c_cost)
        c_variables = [var for var in gvs if var[1].name.startswith("Discriminator")]
        # gvs = [(tf.clip_by_value(grad, -self.clip_value, self.clip_value), var) for grad, var in c_variables]
        # tf.summary.histogram("Discriminator weights", [var[0] for var in c_variables])
        self.c_optimizer = self.c_optimizer.apply_gradients(gvs)

        g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Generator")
        # tf.summary.histogram("Generator weights", g_variables)
        self.g_optimizer = self.optimizer().minimize(self.g_cost, var_list=g_variables)

    def run_session(self, data, hp):
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(hp.path, sess.graph)
            tf.global_variables_initializer().run()

            c_times = 5
            g_times = 1
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
                    sess.run(self.c_optimizer, feed_dict={self.x: data_batch, self.z: z})

                for _ in range(g_times):
                    print("g" + str(_) + " ")
                    z = data.next_batch_fake(hp.batch_size, self.z_size)
                    sess.run(self.g_optimizer, feed_dict={self.z: z})

                print("Generating summary...")
                x = data.next_batch_real(hp.batch_size)
                z = data.next_batch_fake(hp.batch_size, self.z_size)
                summary = sess.run(self.summary_generated_image, feed_dict={self.x: x, self.z: z})
                writer.add_summary(summary, step)
