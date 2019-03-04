
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from statistics import mode

import tensorflow as tf
# from models.model_GAVAE_SIM import GAVAE_SIM
# from models.model_GAVAE_SIM_tf import GAVAE_SIM
from models.model_vae20 import vae_next_gen

def main(_):
    # DATA_PATH = 'D:/Vision_Images/fashion-mnist/'
    DATA_PATH = 'D:/Vision_Images/Pexels_textures/Textures/stonewall'
    # tex_gavae = GAVAE_SIM(data_path=DATA_PATH, w=160, h=160, c=1, layer_depth=3, batch_size=18)
    tex_vae = vae_next_gen(data_path=DATA_PATH, w=64, h=64, c=1, batch_size=32)
    tex_vae.train(iterations=100000, save_interval=1000, log_interval=50, model_file='vae_2_0_stonewall_stacked')
    # tex_gavae.transfer_train(iterations=50000, save_interval=100, log_interval=20, model_file='test.h5')
    # tex_gavae.test_discriminant(model_file='test.h5')
    # tex_gavae.test(model_file='test.h5')
    # tex_gavae.test_discriminator(model_file='test.h5',path='images')
    return 0


if __name__ == '__main__':
    tf.app.run()