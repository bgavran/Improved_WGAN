from data import *
from wgan import *
from generators import *
from critics import *


class Hp:
    batch_size = 32

    z_size = 100

    lr = 0.0001
    beta1 = 0.5
    beta2 = 0.9

    steps = 100000
    path = datapath.path


task = FacesData(img_size=64, crop_size=128)
# data = MNISTData()

generator = ConvGenerator(task.img_size, task.channels)
critic = DCGANCritic(task.img_size, task.channels)

optimizer = tf.train.AdamOptimizer(learning_rate=Hp.lr, beta1=Hp.beta1, beta2=Hp.beta2)
wgan = WGAN(generator, critic, Hp.z_size, task.channels, optimizer=optimizer)
wgan.run_session(task, Hp)
