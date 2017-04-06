import matplotlib.pyplot as plt
from data import *
from wgan import *
from generators import *
from critics import *


class Hp:
    batch_size = 64
    crop_size = 128
    img_size = 64
    z_size = 256

    steps = 100000
    path = datapath.path


data = Data(Hp.img_size, Hp.crop_size)

generator = ConvGenerator(Hp.img_size)
critic = ConvCritic()
wgan = WGAN(generator, critic, Hp.z_size, Hp.img_size)

wgan.run_session(data, Hp)
