
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.model_GAVAE_SIM import LG_MSE

def main(_):
   tex_gavae = TEX_GAVAE(w=400, h=400, c=7, glob_c=24)
   lg_mse.train(epochs=25000, batch_size=16, save_interval=100, dataset_file='./datasets/data_new.h5')
   return 0

if __name__ == '__main__':
   tf.app.run()