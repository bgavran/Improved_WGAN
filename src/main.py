import matplotlib.pyplot as plt
from data import *
from wgan import *
from generators import *
from critics import *


class Hp:
    batch_size = 64
    crop_size = 128
    img_size = 64
    z_size = 100
    lr = 0.0001

    steps = 100000
    path = datapath.path


data = Data(Hp.img_size, Hp.crop_size)

generator = DCGANGenerator(Hp.img_size)
critic = DCGANCritic(Hp.img_size)

optimizer = tf.train.AdamOptimizer(learning_rate=Hp.lr, beta1=0.5, beta2=0.9)
wgan = WGAN(generator, critic, Hp.z_size, Hp.img_size, optimizer=optimizer)

wgan.run_session(data, Hp)
