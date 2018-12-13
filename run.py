
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# from models.model_GAVAE_SIM import GAVAE_SIM
# from models.model_GAVAE_SIM_tf import GAVAE_SIM
from models.model_GAVAE_SIM_tfk_disc import GAVAE_SIM

def main(_):
    DATA_PATH = 'D:/Vision_Images/Pexels_textures/Textures/official'
    tex_gavae = GAVAE_SIM(data_path=DATA_PATH, w=160, h=160, c=1, layer_depth=3, batch_size=8)
    tex_gavae.train(epochs=50000, save_interval=200, model_file='test.h5')
    # tex_gavae.test(model_file='test.h5')
    # tex_gavae.test_discriminator(model_file='test.h5',path='images')
    return 0


if __name__ == '__main__':
    tf.app.run()