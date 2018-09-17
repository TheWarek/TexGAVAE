from __future__ import print_function, division

from .model import ModelGAVAE
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

import numpy as np

def new_custom_loss(y_true, y_pred, sigma, kernel):
    return 0

class GAVAE_SIM(ModelGAVAE):
    def __init__(self, w, h, c, glob_c, batch_size=32, lr=0.0001):
        super(GAVAE_SIM, self).__init__(w, h, c, glob_c, batch_size)

        # Example of custom loss
        custom_loss = self.new_custom_loss(0.5, 400)

        # Additional settings (VAE, GAN, generator, discriminator etc..

    # Custom loss with additional parameters (other than y_pred y_true): How to do it
    def new_custom_loss(self, sigma, kernel):
        def custom_loss(y_true, y_pred):
            return new_custom_loss(y_true, y_pred, sigma, kernel)
        return custom_loss

    # Texture logging?
    def log_imgs(self):
        return

    # Tensorboard (Keras version)
    def write_log(self, callback, scope, names, logs, batch_no):
        with tf.name_scope(scope):
            for name, value in zip(names, logs):
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value
                summary_value.tag = scope + name
                callback.writer.add_summary(summary, batch_no)
                callback.writer.flush()

    # Train
    def train(self, epochs, dataset_file, batch_size=32, save_interval=50):

        # Epochs
        for epoch in range(epochs):
            # TODO: code here

            # Save interval
            if epoch % save_interval == 0:
                # TODO: logging
                print('Please add here some code')
