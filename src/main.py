import matplotlib.pyplot as plt
from data import *
from wgan import *
from generators import *
from discriminators import *


class Hp:
    batch_size = 64

    steps = 100000
    path = datapath.path


data = Data()

generator = FCGenerator(data.img_size)
discriminator = Discriminator()
wgan = WGAN(generator, discriminator, 500, data.img_size)

wgan.run_session(data, Hp)
