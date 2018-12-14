from __future__ import print_function, division

from .model import ModelGAVAE
import tensorflow as tf

import os.path

from matplotlib import pyplot as plt

import numpy as np

from data_loader import TexDAT
from data_loader import resize_batch_images, normalize_batch_images
import sklearn.preprocessing as prep

def new_custom_loss(y_true, y_pred, sigma, kernel):
    return 0

class GAVAE_SIM(ModelGAVAE):
    def __init__(self, data_path, w, h, c, layer_depth, batch_size=32, lr=0.00001):
        super(GAVAE_SIM, self).__init__(w, h, c, layer_depth, batch_size)
        self.patch_size = (w,h,c)

        # self.texdat = TexDAT(data_path, self.batch_size)
        # self.texdat.load_data(only_paths=True)

        self.texdat = TexDAT(data_path, self.batch_size)
        self.texdat.load_images(False)

        # Init TODO: research the middle layer
        self.m = self.batch_size #1 #50
        self.n_z = 128

        self.dropout = 0.3

        self.k_reg = None #regularizers.l2(0.01)
        self.reg = None #regularizers.l2(0.01)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9)

        self.input, self.output = self.get_autoencoder()

        self.autoencoder = tf.keras.models.Model(self.input, self.output, name='autoencoder')

        self.autoencoder.compile(loss=self.ae_loss, optimizer=self.optimizer, metrics=['accuracy'])

        log_path = './logs'
        # TODO : how to using tensorflow implementation of keras
        self.callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0,
                                    write_graph=True, write_images=False)
        self.callback.set_model(self.autoencoder)

    def sample_z(self, args):

        # slice tensor up

        d_4 = args

        a = tf.keras.layers.Flatten()(tf.slice(d_4, [0, 0, 0, 0], [self.batch_size, 160, 1, 1]))
        a_all = tf.keras.layers.Dense(1)(a)
        b = tf.keras.layers.Flatten()(tf.slice(d_4, [0, 0, 0, 0], [self.batch_size, 1, 160, 1]))
        b_all = tf.keras.layers.Dense(1)(b)

        for i in range(1, 160):
            # a is column
            a = tf.keras.layers.Flatten()(tf.slice(d_4, [0, 0, i, 0], [self.batch_size, 160, 1, 1]))
            d_a = tf.keras.layers.Dense(1)(a)
            a_all = tf.keras.layers.concatenate([a_all, d_a], axis=1)
            b = tf.keras.layers.Flatten()(tf.slice(d_4, [0, i, 0, 0], [self.batch_size, 1, 160, 1]))
            d_b = tf.keras.layers.Dense(1)(b)
            b_all = tf.keras.layers.concatenate([b_all, d_b], axis=1)
            # b_all = tf.keras.layers.Dense(1)(b)

        a_all = tf.reshape(a_all, [self.batch_size, 160, 1])
        b_all = tf.reshape(b_all, [self.batch_size, 1, 160])

        # TODO: MATRIX MULTIPLICATION WITH TRANSPOSE MATRIX
        # TODO: DO SOMETHING WITH YOUR LIFE
        c_all = tf.matmul(a_all, b_all)
        c_all = tf.reshape(c_all, [self.batch_size, 160, 160, 1])

        return c_all

    def ae_loss(self, y_true, y_pred):

        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))

        return reconstruction_loss

    def get_autoencoder(self):

        #input = Input(shape=self.img_shape)
        input = tf.keras.layers.Input(batch_shape=self.batch_img_shape)

        l_1 = tf.keras.layers.Conv2D(filters=8, kernel_size=5, strides=1, padding='same', data_format='channels_last',
               dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
               activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(input)
        a_1 = tf.keras.layers.LeakyReLU()(l_1)
        b_1 = tf.keras.layers.BatchNormalization()(a_1)
        #d_1 = tf.keras.layers.Dropout(self.dropout)(b_1)

        l_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_1)
        a_2 = tf.keras.layers.LeakyReLU()(l_2)
        b_2 = tf.keras.layers.BatchNormalization()(a_2)
        #d_2 = tf.keras.layers.Dropout(self.dropout)(b_2)

        l_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_2)
        a_3 = tf.keras.layers.LeakyReLU()(l_3)
        b_3 = tf.keras.layers.BatchNormalization()(a_3)
        #d_3 = tf.keras.layers.Dropout(self.dropout)(b_3)

        l_4 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_3)
        a_4 = tf.keras.layers.LeakyReLU()(l_4)
        b_4 = tf.keras.layers.BatchNormalization()(a_4)
        d_4 = tf.keras.layers.Dropout(self.dropout)(b_4)

        z = tf.keras.layers.Lambda(self.sample_z)(d_4)



        l_5 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                                     dilation_rate=(1, 1), activation=None, use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(z)
        a_5 = tf.keras.layers.LeakyReLU()(l_5)
        b_5 = tf.keras.layers.BatchNormalization()(a_5)

        l_6 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                                     dilation_rate=(1, 1), activation=None, use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_5)
        a_6 = tf.keras.layers.LeakyReLU()(l_6)
        b_6 = tf.keras.layers.BatchNormalization()(a_6)

        l_7 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                                     dilation_rate=(1, 1), activation=None, use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_6)
        a_7 = tf.keras.layers.LeakyReLU()(l_7)
        b_7 = tf.keras.layers.BatchNormalization()(a_7)

        l_8 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                                     dilation_rate=(1, 1), activation=None, use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_7)
        a_8 = tf.keras.layers.LeakyReLU()(l_8)


        return input, a_8

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

        # sorted_list = list(self.texdat.train.objectsPaths.items())
        # sorted_list.sort()
        sorted_list = self.texdat.train.images

        batch_test = []
        # indices = [46, 103, 137, 195, 15, 69, 125, 180]
        indices = [46]
        # for ind in indices:
        #     for i in range(int(self.batch_size/len(indices))):
        #         batch_test.append(self.texdat.read_segment(sorted_list[ind][1].paths[0]))
        # batch_test = resize_batch_images(batch_test, self.patch_size)

        for ind in indices:
            for i in range(int(self.batch_size / len(indices))):
                batch_test.append(
                    self.texdat.load_image_patch(sorted_list[ind], patch_size=self.patch_size))
        batch_test = resize_batch_images(batch_test, self.patch_size)
        batch_test = normalize_batch_images(batch_test, 'zeromean')

        if os.path.exists(model_file):
            #self.vae_complete = load_model(model_file)
            self.autoencoder.load_weights(model_file)

        # Epochs
        for epoch in range(epochs):
            # TODO: code here

            # batch = self.texdat.next_classic_batch_from_paths(self.texdat.train.objectsPaths, self.batch_size,
            #                                                   self.patch_size, normalize='zeromean')

            batch = []
            for ind in indices:
                for i in range(int(self.batch_size / len(indices))):
                    batch.append(
                        self.texdat.load_image_patch(sorted_list[ind], patch_size=self.patch_size))
            batch = resize_batch_images(batch_test, self.patch_size)
            batch = normalize_batch_images(batch, 'zeromean')

            loss = self.autoencoder.train_on_batch(batch, batch)
            print("Epoch: %d loss: %f, acc.: %.2f%%]" % (epoch, loss[0], 100 * loss[1]))

            # Save interval
            if epoch % save_interval == 0:
                if epoch == 0:
                    for i in range(len(indices)):
                        ims = np.reshape(batch_test[(i)*int(self.batch_size / len(indices))], (160, 160))
                        plt.imsave('./images/'+str(i)+'/0_baseline.png', ims, cmap='gray')

                # TODO: logging
                ims = self.autoencoder.predict(batch_test)
                for i in range(len(indices)):
                    imss = np.reshape(ims[(i)*int(self.batch_size / len(indices))], (160, 160))
                    plt.imsave('./images/' + str(i) +'/'+ str(epoch)+'.png', imss, cmap='gray')

                self.autoencoder.save(model_file)
