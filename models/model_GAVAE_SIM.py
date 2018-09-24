from __future__ import print_function, division

from .model import ModelGAVAE
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from keras.optimizers import Adam
from keras import regularizers
from sklearn import preprocessing

from keras.models import load_model

from scipy.misc import imread
import os.path

from matplotlib import pyplot as plt

import numpy as np

from data_loader import TexDAT

def new_custom_loss(y_true, y_pred, sigma, kernel):
    return 0

class GAVAE_SIM(ModelGAVAE):
    def __init__(self, data_path, w, h, c, layer_depth, batch_size=32, lr=0.00001):
        super(GAVAE_SIM, self).__init__(w, h, c, layer_depth, batch_size)
        self.patch_size = (w,h,c)

        self.texdat = TexDAT(data_path, self.batch_size)
        self.texdat.load_data(only_paths=True)

        # Init TODO: research the middle layer
        self.m = batch_size #1 #50
        self.n_z = 5

        # Optimizer
        self.optimizer = Adam(lr=lr, beta_1=0.9)
        self.loss = self.vae_loss
        self.reg = regularizers.l2(0.0001)

        # Example of custom loss
        custom_loss = self.new_custom_loss(0.5, 400)

        # Additional settings (VAE, GAN, generator, discriminator etc..)

        # VAE encoder part
        input, self.mu, self.log_sigma, self.z = self.get_vae_encoder_part()
        # Encoder model, to encode input into latent variable
        # We use the mean as the output as it is the center point, the representative of the gaussian
        self.vae_enc = Model(input, self.mu)

        # VAE decoder part
        # Generator model, generate new data given latent variable z
        # 1. define input for decoder
        input_dec = Input(shape=(self.n_z,))
        layer_list = self.get_vae_decoder_part()
        for i, l in enumerate(layer_list):
            if i == 0:
                l_1 = l(input_dec)
            else:
                l_1 = l(l_1)

        self.vae_dec = Model(input_dec, l_1)

        # Complete VAE model
        # we need to connect it into one model
        # combined encoder + decoder (so we can backpropagate with some loss)
        # TODO: Merge with GAN
        for i, l in enumerate(layer_list):
            if i == 0:
                l_1 = l(self.z)
            else:
                l_1 = l(l_1)
        self.vae_complete = Model(input, l_1)

        #160x160

        self.vae_complete.compile(loss=self.loss,
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])

    def vae_loss(self, y_true, y_pred):
        # compute the average MSE error, then scale it up, ie. simply sum on all axes
        reconstruction_loss = K.mean(K.square(y_pred - y_true))
        # compute the KL loss
        kl_loss = - 0.5 * K.mean(1 + self.log_sigma - K.square(self.mu) - K.square(K.exp(self.log_sigma)), axis=-1)
        # return the average loss over all images in batch
        total_loss = K.mean(reconstruction_loss + kl_loss)
        return total_loss



    def sample_z(self, args):
        mu, log_sigma = args
        eps = K.random_normal(shape=(self.m, self.n_z), mean=0., stddev=1.)
        #sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
        return mu + K.exp(log_sigma / 2) * eps

    def get_vae_decoder_part(self):

        list = []

        list.append(Dense(512, activation=None))
        list.append(LeakyReLU())

        list.append(Dense(self.mid_shape[0] * self.mid_shape[1]))
        list.append(LeakyReLU())

        list.append(Reshape(target_shape=self.mid_shape)) # TODO: need to know middle shape?

        list.append(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        list.append(LeakyReLU())
        list.append(BatchNormalization(momentum=0.8))
        list.append(UpSampling2D(size=(2, 2)))
        list.append(Dropout(0.25))

        list.append(Conv2D(filters=24, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        list.append(LeakyReLU())
        list.append(BatchNormalization(momentum=0.8))
        list.append(UpSampling2D(size=(2, 2)))
        list.append(Dropout(0.25))

        list.append(Conv2D(filters=16, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        list.append(LeakyReLU())
        list.append(BatchNormalization(momentum=0.8))
        list.append(UpSampling2D(size=(2, 2)))
        list.append(Dropout(0.25))

        list.append(Conv2D(filters=1, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        list.append(LeakyReLU())

        return list

    def get_vae_encoder_part(self):

        input = Input(shape=self.img_shape)

        l_1 = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', data_format='channels_last',
               dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
               activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(input)
        a_1 = LeakyReLU()(l_1)
        b_1 = BatchNormalization(momentum=0.8)(a_1)
        d_1 = Dropout(0.25)(b_1)

        l_2 = Conv2D(filters=24, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(d_1)
        a_2 = LeakyReLU()(l_2)
        b_2 = BatchNormalization(momentum=0.8)(a_2)
        d_2 = Dropout(0.25)(b_2)

        l_3 = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(d_2)
        a_3 = LeakyReLU()(l_3)
        b_3 = BatchNormalization(momentum=0.8)(a_3)
        d_3 = Dropout(0.25)(b_3)


        # For now we can flatten it TODO: make it all convolutional. Flattening is BAD PRACTICE
        flat = Flatten()(d_3)
        dense = Dense(512, activation=None)(flat)
        dense_a = LeakyReLU()(dense)

        mu = Dense(self.n_z, activation='linear')(dense_a)
        log_sigma = Dense(self.n_z, activation='linear')(dense_a)

        z = Lambda(self.sample_z)([mu, log_sigma])

        return input, mu, log_sigma, z

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
    def train(self, epochs, model_file, save_interval=50):

        # load image
        # img = imread('./assets/sample.png', mode='L')

        # texdat.next_classic_batch_from_paths(texdat.train.objectsPaths, MINIBATCH_SIZE, PATCH_SIZE)

        # img = np.reshape(img, (1, 160, 160, 1))

        batch_test = self.texdat.next_classic_batch_from_paths(self.texdat.train.objectsPaths, self.batch_size,
                                                          self.patch_size)


        if os.path.exists(model_file):
            #self.vae_complete = load_model(model_file)
            self.vae_complete.load_weights(model_file)

        # Epochs
        for epoch in range(epochs):
            # TODO: code here

            batch = self.texdat.next_classic_batch_from_paths(self.texdat.train.objectsPaths, self.batch_size, self.patch_size)
            loss = self.vae_complete.train_on_batch(batch, batch)
            print("Epoch: %d [loss: %f, acc.: %.2f%%]" % (epoch, loss[0], 100 * loss[1]))

            # Save interval
            if epoch % save_interval == 0:
                mu = self.vae_enc.predict(batch_test)
                if epoch == 0:
                    ims = np.reshape(batch_test[0], (160, 160))
                    plt.imshow(ims, cmap='gray')
                    plt.show()
                # TODO: logging
                ims = self.vae_complete.predict(batch_test)
                ims = np.reshape(ims[0],(160,160))
                plt.imshow(ims, cmap='gray')
                plt.show()

                mu[0][0] = mu[0][0] * 10.6
                mu[0][1] = mu[0][1] * -0.9
                imh = self.vae_dec.predict(mu)

                imh0 = np.reshape(imh[0], (160, 160))
                plt.imshow(imh0, cmap='gray')
                plt.show()
                # imh1 = np.reshape(imh[1], (160, 160))
                # plt.imshow(imh1, cmap='gray')
                # plt.show()

                self.vae_complete.save('test.h5')
