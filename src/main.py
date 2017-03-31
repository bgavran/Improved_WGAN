import matplotlib.pyplot as plt
from src.data import *
from src.wgan import *
from src.generators import *
from src.discriminators import *


class Hp:
    batch_size = 100

    steps = 100000
    path = datapath.path


data = Data()

generator = FCGenerator(data.img_size)
discriminator = Discriminator()
wgan = WGAN(generator, discriminator, 500, data.img_size)

wgan.run_session(data, Hp)
