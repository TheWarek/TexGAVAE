from __future__ import print_function, division

from .model import ModelGAVAE
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda

import matplotlib.pyplot as plt

import numpy as np

def new_custom_loss(y_true, y_pred, sigma, kernel):
    return 0

class GAVAE_SIM(ModelGAVAE):
    def __init__(self, w, h, c, glob_c, batch_size=32, lr=0.0001):
        super(GAVAE_SIM, self).__init__(w, h, c, glob_c, batch_size)

        # Init TODO: research the middle layer
        self.m = 50
        self.n_z = 2

        # Example of custom loss
        custom_loss = self.new_custom_loss(0.5, 400)

        # Additional settings (VAE, GAN, generator, discriminator etc..)

        # VAE encoder part
        input, mu, z = self.get_vae_encoder_part()
        # Encoder model, to encode input into latent variable
        # We use the mean as the output as it is the center point, the representative of the gaussian
        self.vae_enc = Model(input, mu)

        # VAE decoder part
        # Generator model, generate new data given latent variable z
        # 1. define input for decoder
        input_dec = Input(shape=self.n_z)
        layer_list = self.get_vae_decoder_part()
        for i, l in enumerate(layer_list):
            if i == 1:
                l_1 = l(input_dec)
            else:
                l_1 = l(l_1)

        self.vae_dec = Model(input_dec, l_1)

        # Complete VAE model
        # we need to connect it into one model
        # combined encoder + decoder (so we can backpropagate with some loss)
        # TODO: Merge with GAN
        for i, l in enumerate(layer_list):
            if i == 1:
                l_1 = l(z)
            else:
                l_1 = l(l_1)
        self.complete_vae = Model(input, l_1)


    def sample_z(self, args):
        mu, log_sigma = args
        eps = K.random_normal(shape=(self.m, self.n_z), mean=0., std=1.)
        return mu + K.exp(log_sigma / 2) * eps

    def get_vae_decoder_part(self):

        list = []

        list.append(Dense(512, activation='leakyRELU'))

        list.append(Reshape(shape=self.mid_shape)) # TODO: need to know middle shape?

        list.append(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation='leakyRELU', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        list.append(BatchNormalization(momentum=0.8))
        list.append(UpSampling2D(size=(2,2)))
        list.append(Dropout(0.25))

        list.append(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation='leakyRELU', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        list.append(BatchNormalization(momentum=0.8))
        list.append(UpSampling2D(size=(2, 2)))
        list.append(Dropout(0.25))

        list.append(Conv2D(filters=24, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation='leakyRELU', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        list.append(BatchNormalization(momentum=0.8))
        list.append(UpSampling2D(size=(2, 2)))
        list.append(Dropout(0.25))

        list.append(Conv2D(filters=16, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation='leakyRELU', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        list.append(BatchNormalization(momentum=0.8))
        list.append(UpSampling2D(size=(2, 2)))
        list.append(Dropout(0.25))

        list.append(Conv2D(filters=1, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation='leakyRELU', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))

        return list

    def get_vae_encoder_part(self):

        input = Input(shape=self.img_shape)

        l_1 = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', data_format='channels_last',
               dilation_rate=(1, 1), activation='leakyRELU', use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
               activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(input)
        b_1 = BatchNormalization(momentum=0.8)(l_1)
        d_1 = Dropout(0.25)(b_1)

        l_2 = Conv2D(filters=24, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation='leakyRELU', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(d_1)
        b_2 = BatchNormalization(momentum=0.8)(l_2)
        d_2 = Dropout(0.25)(b_2)

        l_3 = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation='leakyRELU', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(d_2)
        b_3 = BatchNormalization(momentum=0.8)(l_3)
        d_3 = Dropout(0.25)(b_3)

        l_4 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation='leakyRELU', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(d_3)
        b_4 = BatchNormalization(momentum=0.8)(l_4)
        d_4 = Dropout(0.25)(b_4)


        # For now we can flatten it TODO: make it all convolutional. Flattening is BAD PRACTICE
        flat = Flatten()(d_4)
        dense = Dense(512, activation='leakyRELU')(flat)

        mu = Dense(self.n_z, activation='linear')(dense)
        log_sigma = Dense(self.n_z, activation='linear')(dense)

        z = Lambda(self.sample_z)([mu, log_sigma])

        return input, mu, z

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
