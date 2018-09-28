
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.model_GAVAE_SIM import GAVAE_SIM

def main(_):
    DATA_PATH = 'E:/Datasets/texdat'
    tex_gavae = GAVAE_SIM(data_path=DATA_PATH, w=160, h=160, c=1, layer_depth=3, batch_size=8)
    tex_gavae.train(epochs=50000, save_interval=500, model_file='test.h5')
    return 0

if __name__ == '__main__':
    tf.app.run()