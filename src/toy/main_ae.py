import matplotlib.pyplot as plt
from data import *
from toy.autoencoder import *
from toy.encoders import *
from toy.decoders import *


class Hp:
    batch_size = 64
    img_size = 128
    z_size = 1024

    steps = 100000
    path = datapath.path


data = Data(Hp.img_size)

encoder = ConvEncoder(Hp.img_size, Hp.z_size)
decoder = ConvDecoder(Hp.img_size)
ae = Autoencoder(encoder, decoder, Hp.img_size)

ae.run_session(data, Hp)
