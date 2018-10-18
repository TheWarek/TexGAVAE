from __future__ import print_function, division

from .model import ModelGAVAE
import tensorflow as tf
import keras.backend as K
import keras
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from keras.optimizers import Adam, SGD, Adadelta

import os.path

from matplotlib import pyplot as plt
from scipy.misc import imread

import numpy as np

from time import time

from data_loader import TexDAT
from data_loader import resize_batch_images, normalize_batch_images
import sklearn.preprocessing as prep

def new_custom_loss(y_true, y_pred, sigma, kernel):
    return 0

class GAVAE_SIM(ModelGAVAE):

    class GAVAE_tensorboard(keras.callbacks.TensorBoard):
        def __init__(self, log_dir, histogram_freq, batch_size, write_graph, write_images):
            self.log_dir = log_dir
            super().__init__(log_dir=log_dir, histogram_freq=histogram_freq, batch_size=batch_size,
                             write_graph=write_graph, write_images=write_images)

        def set_model(self, model):
            self.model = model
            self.file_writer = tf.summary.FileWriter(self.log_dir)
            super().set_model(model)

        def on_epoch_end(self, epoch, logs=None):
            print(logs)
            # self.file_writer.add_summary()
            super().on_epoch_end(epoch, logs)


    def __init__(self, data_path, w, h, c, layer_depth, batch_size=32, lr=0.001, margin=4.5):
        super(GAVAE_SIM, self).__init__(w, h, c, layer_depth, batch_size)
        self.patch_size = (w,h,c)

        self.texdat = TexDAT(data_path, self.batch_size)
        self.texdat.load_data(only_paths=True)

        # Init TODO: research the middle layer

        self.m = batch_size #1 #50
        self.n_z = 256
        self.margin = margin
        self.dropout = 0.1

        # Optimizer
        self.optimizer = Adam(lr=lr, beta_1=0.9)
        self.optimizer_disc = Adam(lr=lr, beta_1=0.9) #SGD(lr=lr / 100.) #
        # self.optimizer = SGD(lr=lr, momentum=0.9, decay=1e-6, nesterov=True)
        # self.optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        self.loss = self.vae_loss
        self.reg = None #regularizers.l2(1e-4)

        # Example of custom loss
        # custom_loss = self.new_custom_loss(0.5, 400)

        # Additional settings (VAE, GAN, generator, discriminator etc..)

        # VAE encoder part
        self.input, self.mu_1, self.log_sigma_1, self.z = self.get_vae_encoder_part()
        # Encoder model, to encode input into latent variable
        # We use the mean as the output as it is the center point, the representative of the gaussian
        self.vae_enc = Model(self.input, [self.mu_1, self.log_sigma_1], name='vae_enc')

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
        self.vae_dec = Model(input_dec, l_1, name='vae_dec')

        # Complete VAE model
        # we need to connect it into one model
        # combined encoder + decoder (so we can backpropagate with some loss)
        # TODO: Merge with GAN
        for i, l in enumerate(layer_list):
            if i == 0:
                l_1 = l(self.z)
            else:
                l_1 = l(l_1)

        with tf.variable_scope('vae_complete'):
            self.vae_output = l_1
            self.vae_complete = Model(self.input, self.vae_output, name='vae_complete')

        self.mu_2, self.log_sigma_2 = self.vae_enc(self.vae_complete(self.input))

        #160x160

        self.vae_complete.compile(loss=self.loss,
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])

        # Implementation with GAN:

        # Discriminator part (whole model to train)
        disc_input, disc_output = self.get_discriminator()
        # create discriminator model
        self.discriminator = Model(disc_input, disc_output, name='discriminator')
        self.discriminator.compile(loss=self.disc_loss,
                                   optimizer=self.optimizer_disc,
                                   metrics=['accuracy'])

        # Combined model = generator part which train on discriminator loss
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        img = self.vae_complete(self.input)
        valid = self.discriminator(img)

        self.custom_loss = self.custom_gen_loss_wrapper(self.input)
        self.generator_combined = Model(self.input, valid, name='generator_combined')
        self.generator_combined.compile(loss=self.custom_loss,
                                        optimizer=self.optimizer,
                                        metrics=['accuracy'])
        self.summary = tf.summary.merge_all()
        #self.discriminator.summary()
        #self.generator_combined.summary()
        #self.vae_complete.summary()
        #self.vae_enc.summary()

        log_path = './logs'
        # TODO : how to using tensorflow implementation of keras
        self.callback = keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0,
                                    write_graph=True, write_images=False)
        self.callback.set_model(self.generator_combined)

    def losses(self):
        result = {}
        result['meansquare_loss'] = 3 #self.meansquare_loss
        result['margin_loss'] = 5 #self.margin_loss
        result['reconstruction_loss'] = 7# self.reconstruction_loss
        result['kl_loss'] = 9 #self.kl_loss
        result['total_loss'] = 6 #K.mean(self.meansquare_loss + 0.02 * self.kl_loss + 0.7 * self.margin_loss + self.reconstruction_loss)

    def vae_loss(self, y_true, y_pred):
        # compute the average MSE error, then scale it up, ie. simply sum on all axes
        # reconstruction_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        reconstruction_loss = K.mean(K.square(y_pred - y_true))
        # compute the KL loss
        kl_loss = - 0.5 * K.mean(1 + self.log_sigma_1 - K.square(self.mu_1) - K.square(K.exp(self.log_sigma_1)), axis=-1)
        # return the average loss over all images in batch
        total_loss = K.mean(reconstruction_loss + 0.02 * kl_loss)
        return total_loss

    # Custom loss with additional parameters (other than y_pred y_true): How to do it
    def custom_gen_loss_wrapper(self, batch_input):
        def custom_gen_loss(y_true, y_pred):
            return self.gen_loss(y_true, y_pred, batch_input)
        return custom_gen_loss

    def gen_loss(self, y_true, y_pred, batch_input):
        # encoder loss
        z2 = Lambda(self.sample_z)([self.mu_2, self.log_sigma_2])
        self.margin_loss = K.mean(K.minimum(0., K.sqrt(K.sum(K.square(z2 - self.z),axis=1)) - self.margin)) # better name for margin loss
        # tf.summary.scalar('margin_loss', self.margin_loss)
        # compute the binary crossentropy
        # trueseems_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        self.meansquare_loss = K.mean(K.square(y_pred - y_true))
        # tf.summary.scalar('true_loss',trueseems_loss)
        reconstruction_loss = -K.sum(batch_input * K.log(1e-5 + self.vae_output) + (1-batch_input) * K.log(1e-5 + 1 - self.vae_output), 1)
        self.reconstruction_loss = K.mean(reconstruction_loss) / (self.patch_size[0]*self.patch_size[1])
        # tf.summary.scalar('reconstruction_loss',reconstruction_loss)
        # compute the KL loss
        self.kl_loss = - 0.5 * K.mean(1 + self.log_sigma_1 - K.square(self.mu_1) - K.square(K.exp(self.log_sigma_1)), axis=-1)
        # tf.summary.scalar('kl_loss', kl_loss)
        # return the average loss over all images in batch
        total_loss = K.mean(self.meansquare_loss + 0.02 * self.kl_loss + 0.7 * self.margin_loss + self.reconstruction_loss)
        return total_loss

    def disc_loss(selfself, y_true, y_pred):
        # reconstruction_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        reconstruction_loss = K.mean(K.square(y_pred - y_true))
        tf.summary.scalar('discriminator_loss',reconstruction_loss)
        return reconstruction_loss

    def sample_z(self, args):
        mu, log_sigma = args
        eps = K.random_normal(shape=(self.m, self.n_z), mean=0., stddev=1.)
        #sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
        return mu + K.exp(log_sigma / 2) * eps

    def get_vae_decoder_part(self):

        list = []

        # list.append(Dense(512, activation=None))
        # list.append(LeakyReLU())
        # list.append(BatchNormalization(momentum=0.8))

        list.append(Dense(self.mid_shape[0] * self.mid_shape[1] * 16))
        list.append(LeakyReLU())
        list.append(BatchNormalization(momentum=0.8))

        list.append(Reshape(target_shape=self.mid_shape_16))  # TODO: need to know middle shape?

        list.append(Conv2D(filters=64, kernel_size=1, strides=1, padding='same', data_format='channels_last',
                           dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                           activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        list.append(LeakyReLU())
        list.append(BatchNormalization(momentum=0.8))

        list.append(Conv2D(filters=128, kernel_size=1, strides=1, padding='same', data_format='channels_last',
                           dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                           activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        list.append(LeakyReLU())
        list.append(BatchNormalization(momentum=0.8))

        list.append(Conv2D(filters=512, kernel_size=7, strides=1, padding='same', data_format='channels_last',
                           dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                           activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        list.append(LeakyReLU())
        list.append(UpSampling2D(size=(2, 2)))
        list.append(BatchNormalization(momentum=0.8))
        list.append(Dropout(self.dropout))

        list.append(Conv2D(filters=384, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                           dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                           activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        list.append(LeakyReLU())
        list.append(UpSampling2D(size=(2, 2)))
        list.append(BatchNormalization(momentum=0.8))
        list.append(Dropout(self.dropout))

        list.append(Conv2D(filters=128, kernel_size=5, strides=1, padding='same', data_format='channels_last',
                           dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                           activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        list.append(LeakyReLU())
        list.append(UpSampling2D(size=(2, 2)))
        list.append(BatchNormalization(momentum=0.8))
        list.append(Dropout(self.dropout))

        list.append(Conv2D(filters=96, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                           dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                           activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        list.append(LeakyReLU())
        list.append(BatchNormalization(momentum=0.8))

        list.append(Conv2D(filters=1, kernel_size=1, strides=1, padding='same', data_format='channels_last',
                           dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                           activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        list.append(Activation('sigmoid'))

        return list

    def get_vae_encoder_part(self):

        input = Input(shape=self.img_shape)

        l_1 = Conv2D(filters=96, kernel_size=3, strides=2, padding='same', data_format='channels_last',
               dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
               activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(input)
        a_1 = LeakyReLU()(l_1)
        b_1 = BatchNormalization(momentum=0.8)(a_1)
        d_1 = Dropout(self.dropout)(b_1)

        l_2 = Conv2D(filters=128, kernel_size=5, strides=2, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(d_1)
        a_2 = LeakyReLU()(l_2)
        b_2 = BatchNormalization(momentum=0.8)(a_2)
        d_2 = Dropout(self.dropout)(b_2)

        l_3 = Conv2D(filters=384, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(d_2)
        a_3 = LeakyReLU()(l_3)
        b_3 = BatchNormalization(momentum=0.8)(a_3)
        d_3 = Dropout(self.dropout)(b_3)

        l_4 = Conv2D(filters=512, kernel_size=7, strides=2, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(d_3)
        a_4 = LeakyReLU()(l_4)
        b_4 = BatchNormalization(momentum=0.8)(a_4)

        l_5 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_4)
        a_5 = LeakyReLU()(l_5)
        b_5 = BatchNormalization(momentum=0.8)(a_5)

        l_6 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_5)
        a_6 = LeakyReLU()(l_6)
        b_6 = BatchNormalization(momentum=0.8)(a_6)

        l_7 = Conv2D(filters=16, kernel_size=1, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_6)
        a_7 = LeakyReLU()(l_7)
        b_7 = BatchNormalization(momentum=0.8)(a_7)


        # For now we can flatten it TODO: make it all convolutional. Flattening is BAD PRACTICE
        flat = Flatten()(b_7)
        #dense = Dense(512, activation=None)(flat)
        #dense_a = LeakyReLU()(dense)
        #bn = BatchNormalization(momentum=0.8)(dense_a)

        mu = Dense(self.n_z, activation='linear')(flat)
        log_sigma = Dense(self.n_z, activation='linear')(flat)

        z = Lambda(self.sample_z)([mu, log_sigma])

        return input, mu, log_sigma, z

    def get_discriminator(self):

        input = Input(shape=self.img_shape)

        l_1 = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(input)
        a_1 = LeakyReLU()(l_1)
        b_1 = BatchNormalization(momentum=0.8)(a_1)
        d_1 = Dropout(self.dropout)(b_1)

        l_2 = Conv2D(filters=48, kernel_size=5, strides=2, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(d_1)
        a_2 = LeakyReLU()(l_2)
        b_2 = BatchNormalization(momentum=0.8)(a_2)
        d_2 = Dropout(self.dropout)(b_2)

        l_3 = Conv2D(filters=64, kernel_size=5, strides=2, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(d_2)
        a_3 = LeakyReLU()(l_3)
        b_3 = BatchNormalization(momentum=0.8)(a_3)
        d_3 = Dropout(self.dropout)(b_3)

        # For now we can flatten it TODO: make it all convolutional. Flattening is BAD PRACTICE

        flat = Flatten()(d_3)

        dense = Dense(128)(flat)
        dense_a = LeakyReLU()(dense)
        bn = BatchNormalization(momentum=0.8)(dense_a)
        drop = Dropout(self.dropout)(bn)
        fin = Dense(1, activation='sigmoid')(drop)


        return input, fin

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

        # batch_test = self.texdat.next_classic_batch_from_paths(self.texdat.train.objectsPaths, self.batch_size,
        #                                                   self.patch_size, normalize='zeromean')
        # batch = batch_test

        sorted_list = list(self.texdat.train.objectsPaths.items())
        sorted_list.sort()

        batch_test = []
        indices = [31, 103, 137, 194]
        for ind in indices:
            for i in range(int(self.batch_size / 4)):
                batch_test.append(self.texdat.read_image_patch(sorted_list[ind][1].paths[0], patch_size=self.patch_size))
        batch_test = resize_batch_images(batch_test, self.patch_size)

        # batch_test = normalize_batch_images(batch_test, 'minmax')

        train_disc = True
        train_gen = True

        if os.path.exists(model_file):
            # self.vae_complete = load_model(model_file)
            self.generator_combined.load_weights(model_file)

        # Epochs
        for epoch in range(epochs):
            # TODO: code here

            # batch = self.texdat.next_classic_batch_from_paths(self.texdat.train.objectsPaths, self.batch_size,
            #                                                   self.patch_size, normalize='zeromean')

            batch = []
            for ind in indices:
                for i in range(int(self.batch_size / 4)):
                    batch.append(self.texdat.read_image_patch(sorted_list[ind][1].paths[0], patch_size=self.patch_size))
            batch = resize_batch_images(batch, self.patch_size)

            generated = self.vae_complete.predict(batch)

            # batch_discriminator = np.concatenate((batch, generated))
            # labels = np.concatenate((np.ones(self.batch_size, np.float32), (np.zeros(self.batch_size, np.float32))))
            # loss_disc = self.discriminator.train_on_batch(batch_discriminator,labels)
            # print("Epoch: %d [Disc. loss: %f, acc.: %.2f%%]" % (epoch, loss_disc[0], 100 * loss_disc[1]))

            # IMPROVE 2. use noisy labels
            labels_real = np.ones(self.batch_size, np.float32) + np.subtract(np.multiply(np.random.rand(self.batch_size), 0.3), 0.15)
            labels_fake = np.zeros(self.batch_size, np.float32) + np.multiply(np.random.rand(self.batch_size), 0.3)

            # IMPROVE 3. train gen/disc depending on each other:
            if train_disc:
                # IMPROVE 1. mini-batches of REAL / FAKE
                loss_real = self.discriminator.train_on_batch(batch, labels_real)
                loss_fake = self.discriminator.train_on_batch(generated, labels_fake)
                loss_disc = 0.5 * np.add(loss_real, loss_fake)
                print("Epoch: %d [Disc. loss: %f, acc.: %.2f%%]" % (epoch, loss_disc[0], 100 * loss_disc[1]))

            if train_gen:

                loss_gen = self.generator_combined.train_on_batch(batch, labels_real)
                print("Epoch: %d [Gen. loss: %f, acc.: %.2f%%]" % (epoch, loss_gen[0], 100 * loss_gen[1]))
                self.callback.on_epoch_end(epoch, self.losses())
            # we will train generator until it will overcome 1.5* of disc loss
            # as generator takes usually more time to train
            # if loss_gen[0] <= 2.5 * loss_disc[0]:
            #     train_disc = True
            # else:
            #     train_disc = False


            # Save interval
            if epoch % save_interval == 0:
                # mu = self.vae_enc.predict(batch_test)
                if epoch == 0:
                    ims = np.reshape(batch_test[0], (160, 160))
                    plt.imsave('./images/0/0_baseline.png', ims, cmap='gray')
                    ims = np.reshape(batch_test[1], (160, 160))
                    plt.imsave('./images/1/0_baseline.png', ims, cmap='gray')
                    ims = np.reshape(batch_test[2], (160, 160))
                    plt.imsave('./images/2/0_baseline.png', ims, cmap='gray')
                    ims = np.reshape(batch_test[3], (160, 160))
                    plt.imsave('./images/3/0_baseline.png', ims, cmap='gray')
                # TODO: logging
                ims = self.vae_complete.predict(batch_test)
                imss = np.reshape(ims[0], (160, 160))
                plt.imsave('./images/0/' + str(epoch) + '.png', imss, cmap='gray')
                imss = np.reshape(ims[1], (160, 160))
                plt.imsave('./images/1/' + str(epoch) + '.png', imss, cmap='gray')
                imss = np.reshape(ims[2], (160, 160))
                plt.imsave('./images/2/' + str(epoch) + '.png', imss, cmap='gray')
                imss = np.reshape(ims[3], (160, 160))
                plt.imsave('./images/3/' + str(epoch) + '.png', imss, cmap='gray')

                self.generator_combined.save(model_file)

    def test_discriminator(self, model_file, path):
        if os.path.exists(model_file):
            # self.vae_complete = load_model(model_file)
            self.generator_combined.load_weights(model_file)

        for i in range(4):
            patches = os.listdir(path + '/' + str(i))
            print('Texture folder: {:s}'.format(path + '/' + str(i)))
            batch = []
            for patch_p in patches:
                patch = (imread(os.path.join(path,str(i),patch_p), mode='L')/256).astype(np.float32)
                batch.append(patch.reshape((160,160,1)))
            batch = normalize_batch_images(batch, 'zeromean')
            valid = self.discriminator.predict(batch)
            for i in range(valid.shape[0]):
                print('Patch {:d}: {:d}'.format(i, int(valid[i])))



