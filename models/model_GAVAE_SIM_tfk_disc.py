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
        self.n_z = 512

        self.dropout = 0.3

        self.k_reg = None #regularizers.l2(0.01)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9)
        self.optimizer_disc = tf.keras.optimizers.Adam(lr=lr , beta_1=0.9) #SGD(lr=lr / 100.) #
        # self.optimizer = SGD(lr=lr, momentum=0.9, decay=1e-6, nesterov=True)
        # self.optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        self.reg = None #regularizers.l2(1e-3)

        # BASIC AUTOENCODER
        self.input = tf.keras.layers.Input(batch_shape=self.batch_img_shape)
        self.input_d1 = tf.keras.layers.Input(batch_shape=self.batch_img_shape)
        self.input_d2 = tf.keras.layers.Input(batch_shape=self.batch_img_shape)

        self.input_dummy = tf.keras.layers.Input(batch_shape=self.batch_img_shape)

        self.ae_a = self.get_autoencoder_a(self.input_d1)
        self.ae_b = self.get_autoencoder_b(self.ae_a)

        self.generator = tf.keras.models.Model(self.input_d1, self.ae_b, name='aec')

        self.generator.compile(loss=self.gen_loss,
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])

        # DISTRIBUTED DISCRIMINATOR

        self.disc_first = self.get_disc_first(self.input_d1)
        self.disc_second = self.get_disc_first(self.input_d2)

        self.disc_first_model = tf.keras.models.Model(self.input_d1, self.disc_first, name='disc_first_model')
        self.disc_second_model = tf.keras.models.Model(self.input_d2, self.disc_second, name='disc_second_model')

        self.disc_a = self.disc_first_model(self.input_d1)
        self.disc_b = self.disc_second_model(self.input_d2)

        # self.disc_last, self.mu, self.sigma = self.get_disc_second(self.disc_a, self.disc_b)
        self.disc_last, self.mu, self.sigma = self.get_disc_second(self.disc_a, self.disc_b)

        #self.disc_all_model = tf.keras.models.Model([self.disc_a, self.disc_b], self.disc_last, name='disc_first_model')


        self.discriminator = tf.keras.models.Model([self.input_d1, self.input_d2], self.disc_last, name='discriminator')

        self.discriminator.compile(loss=self.disc_loss,
                                   optimizer=self.optimizer_disc,
                                   metrics=['accuracy'])

        # combined model
        self.discriminator.trainable = False

        self.comb = self.discriminator([self.generator(self.input_d1), self.input_d2])

        self.combined = tf.keras.models.Model([self.input_d1, self.input_d2], self.comb, name='combined')
        self.combined.compile(loss=self.comb_loss,
                                    optimizer=self.optimizer_disc,
                                    metrics=['accuracy'])


        log_path = './logs'
        # TODO : how to using tensorflow implementation of keras
        self.callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0,
                                    write_graph=True, write_images=False)
        self.callback.set_model(self.combined)


    def comb_loss(self, y_true, y_pred):
        # MSE
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))
        # Goodness-of-fit - R2 score
        # SS_res = tf.keras.backend.sum(tf.keras.backend.square(self.ae_a - self.mu), axis=0)
        # SS_tot = tf.keras.backend.sum(tf.keras.backend.square(self.ae_a - tf.keras.backend.mean(self.ae_a, axis=0)),axis=0)
        # goodness_fit_rows =  (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))
        # goodness_fit = tf.keras.backend.mean(goodness_fit_rows)


        chi = tf.keras.backend.mean(tf.keras.backend.sum(tf.keras.backend.abs(tf.keras.backend.square(self.ae_a - self.mu) / self.mu), axis = 0))

        # reconstruction
        reconstruction_loss2 = tf.keras.backend.mean(tf.keras.backend.square(self.ae_b - self.input_d2))
        return reconstruction_loss + chi + reconstruction_loss2

    def gen_loss(self, y_true, y_pred):
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))

        return reconstruction_loss

    def disc_loss(self, y_true, y_pred):
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))

        reconstruction_loss2 = 1. / tf.keras.backend.abs(tf.keras.backend.log(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)) + 0.00001) / tf.keras.backend.log(10.))
        return reconstruction_loss


    def sample_z(self, args):
        mu, log_sigma = args
        eps = tf.keras.backend.random_normal(shape=(self.m, self.n_z), mean=0., stddev=1.)
        #sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
        return mu + tf.keras.backend.exp(log_sigma / 2) * eps


    def get_autoencoder_a(self, input):

        l_1 = tf.keras.layers.Conv2D(filters=8, kernel_size=5, strides=2, padding='same', data_format='channels_last',
                                     dilation_rate=(1, 1), activation=None, use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(input)
        a_1 = tf.keras.layers.LeakyReLU()(l_1)
        b_1 = tf.keras.layers.BatchNormalization()(a_1)
        # d_1 = tf.keras.layers.Dropout(self.dropout)(b_1)

        l_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=2, padding='same', data_format='channels_last',
                                     dilation_rate=(1, 1), activation=None, use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_1)
        a_2 = tf.keras.layers.LeakyReLU()(l_2)
        b_2 = tf.keras.layers.BatchNormalization()(a_2)
        # d_2 = tf.keras.layers.Dropout(self.dropout)(b_2)

        l_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same', data_format='channels_last',
                                     dilation_rate=(1, 1), activation=None, use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_2)
        a_3 = tf.keras.layers.LeakyReLU()(l_3)
        b_3 = tf.keras.layers.BatchNormalization()(a_3)
        # d_3 = tf.keras.layers.Dropout(self.dropout)(b_3)

        l_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                                     dilation_rate=(1, 1), activation=None, use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_3)
        a_4 = tf.keras.layers.LeakyReLU()(l_4)
        b_4 = tf.keras.layers.BatchNormalization()(a_4)
        d_4 = tf.keras.layers.Dropout(self.dropout)(b_4)

        # For now we can flatten it TODO: make it all convolutional. Flattening is BAD PRACTICE
        f = tf.keras.layers.Flatten()(d_4)

        dense = tf.keras.layers.Dense(self.n_z)(f)
        flat = tf.keras.layers.LeakyReLU()(dense)

        return flat

    def get_autoencoder_b(self, input):

        first = tf.keras.layers.Dense(64 * 10 * 10)(input)
        act = tf.keras.layers.LeakyReLU()(first)
        res = tf.keras.layers.Reshape(target_shape=(10, 10, 64))(act)

        l_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', data_format='channels_last',
                                     dilation_rate=(1, 1), activation=None, use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(res)
        a_1 = tf.keras.layers.LeakyReLU()(l_1)
        u_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(a_1)
        b_1 = tf.keras.layers.BatchNormalization()(u_1)
        # d_1 = tf.keras.layers.Dropout(self.dropout)(b_1)

        l_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', data_format='channels_last',
                                     dilation_rate=(1, 1), activation=None, use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_1)
        a_2 = tf.keras.layers.LeakyReLU()(l_2)
        u_2 = tf.keras.layers.UpSampling2D(size=(2, 2))(a_2)
        b_2 = tf.keras.layers.BatchNormalization()(u_2)
        # d_2 = tf.keras.layers.Dropout(self.dropout)(b_2)

        l_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', data_format='channels_last',
                                     dilation_rate=(1, 1), activation=None, use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_2)
        a_3 = tf.keras.layers.LeakyReLU()(l_3)
        u_3 = tf.keras.layers.UpSampling2D(size=(2, 2))(a_3)
        b_3 = tf.keras.layers.BatchNormalization()(u_3)
        # d_3 = tf.keras.layers.Dropout(self.dropout)(b_3)

        l_4 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                                     dilation_rate=(1, 1), activation=None, use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_3)
        a_4 = tf.keras.layers.LeakyReLU()(l_4)
        u_4 = tf.keras.layers.UpSampling2D(size=(2, 2))(a_4)
        b_4 = tf.keras.layers.BatchNormalization()(u_4)
        # d_4 = tf.keras.layers.Dropout(self.dropout)(b_4)
        l_4 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same', data_format='channels_last',
                                     dilation_rate=(1, 1), activation=None, use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_4)
        return l_4

    def get_disc_first(self, input):
        #input = tf.keras.layers.Input(batch_shape=self.batch_img_shape)
        l_1 = tf.keras.layers.Conv2D(filters=8, kernel_size=5, strides=2, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(input)
        a_1 = tf.keras.layers.LeakyReLU()(l_1)
        b_1 = tf.keras.layers.BatchNormalization()(a_1)
        #d_1 = tf.keras.layers.Dropout(self.dropout)(a_1)

        l_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=2, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_1)
        a_2 = tf.keras.layers.LeakyReLU()(l_2)
        b_2 = tf.keras.layers.BatchNormalization()(a_2)
        #d_2 = tf.keras.layers.Dropout(self.dropout)(a_2)

        l_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_2)
        a_3 = tf.keras.layers.LeakyReLU()(l_3)
        b_3 = tf.keras.layers.BatchNormalization()(a_3)
        #d_3 = tf.keras.layers.Dropout(self.dropout)(a_3)

        l_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
                     activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(b_3)
        a_4 = tf.keras.layers.LeakyReLU()(l_4)
        b_4 = tf.keras.layers.BatchNormalization()(a_4)
        #d_4 = tf.keras.layers.Dropout(self.dropout)(a_4)

        # l_4 = Conv2D(filters=16, kernel_size=1, strides=1, padding='same', data_format='channels_last',
        #              dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        #              bias_initializer='zeros', kernel_regularizer=self.k_reg, bias_regularizer=None,
        #              activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None)(d_3)
        # a_4 = LeakyReLU()(l_4)
        # b_4 = BatchNormalization(momentum=0.8)(a_4)

        # For now we can flatten it TODO: make it all convolutional. Flattening is BAD PRACTICE

        flat = tf.keras.layers.Flatten()(b_4)
        drop = tf.keras.layers.Dropout(self.dropout)(flat)
        dense = tf.keras.layers.Dense(512)(drop)
        fin = tf.keras.layers.LeakyReLU()(dense)


        return fin

    def get_disc_second(self, layer1, layer2):
        input = tf.keras.layers.concatenate([layer1, layer2], axis=1)

        mu = tf.keras.layers.Dense(self.n_z, activation='linear')(input)
        log_sigma = tf.keras.layers.Dense(self.n_z, activation='linear')(input)

        z = tf.keras.layers.Lambda(self.sample_z)([mu, log_sigma])
        dense = tf.keras.layers.Dense(1, activation='sigmoid')(z)
        fin = tf.keras.layers.LeakyReLU()(dense)

        return fin, mu, log_sigma

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
        # indices = np.array([46, 103, 137, 195, 15, 69, 125, 180])
        # indices = np.array([46, 103, 137, 195])
        indices = np.array([305, 309, 314, 345])
        # indices = [46, 195]
        # indices = [46]
        #indices = [46, 195]

        for ind in indices:
            for i in range(int(self.batch_size/len(indices))):
                #batch_test.append(self.texdat.read_segment(sorted_list[ind][1].paths[0]))
                batch_test.append(self.texdat.load_image_patch(sorted_list[ind], patch_size=self.patch_size))
        batch_test = resize_batch_images(batch_test, self.patch_size)

        batch_test = normalize_batch_images(batch_test, 'zeromean')

        train_disc = True
        train_gen = True
        train_gen_count = 1
        train_disc_count = 2

        if os.path.exists(model_file):
            #self.vae_complete = load_model(model_file)
            self.generator_combined.load_weights(model_file)

        # Epochs
        for epoch in range(epochs):
            # TODO: code here

            # batch = self.texdat.next_classic_batch_from_paths(self.texdat.train.objectsPaths, self.batch_size,
            #                                                   self.patch_size, normalize='zeromean')

            permutation = np.random.permutation(len(indices))
            shuffled_indices = np.empty(indices.shape, dtype=indices.dtype)
            for old_index, new_index in enumerate(permutation):
                shuffled_indices[new_index] = indices[old_index]

            indices = shuffled_indices

            batch = []
            batch_disc = []
            for ind in indices:
                for i in range(int(self.batch_size / len(indices))):
                    # batch.append(self.texdat.read_image_patch(sorted_list[ind][1].paths[0], patch_size=self.patch_size))
                    batch.append(self.texdat.load_image_patch(sorted_list[ind], patch_size=self.patch_size))
                    batch_disc.append(self.texdat.load_image_patch(sorted_list[ind], patch_size=self.patch_size))
                    # batch.append(self.texdat.read_segment(sorted_list[ind][1].paths[0]))
                    # batch_disc.append(self.texdat.read_segment(sorted_list[ind][1].paths[0]))
            batch = resize_batch_images(batch, self.patch_size)
            batch_disc = resize_batch_images(batch_disc, self.patch_size)
            batch = normalize_batch_images(batch, 'zeromean')
            batch_disc = normalize_batch_images(batch_disc, 'zeromean')

            # SHIFT for additional training
            batch_disc_roll = np.roll(batch_disc, int(self.batch_size / len(indices)))

            # IMPROVE 2. use noisy labels
            # IMPORVEMENT other way around -> real =0, fake = 1, better for gradient flow
            # IMPROVEMENT noise not required when training generator
            labels_fake = np.ones(self.batch_size, np.float32) + np.subtract(np.multiply(np.random.rand(self.batch_size), 0.2 ), 0.1)
            labels_fake_g = np.ones(self.batch_size, np.float32)
            labels_real = np.zeros(self.batch_size, np.float32) + np.multiply(np.random.rand(self.batch_size), 0.2 )
            labels_real_g = np.zeros(self.batch_size, np.float32)

            # IMPROVE 3. train gen/disc depending on each other:

            if train_disc:
                #
                generated = self.generator.predict(batch)

                # SPLIT BATCHES
                loss_disc_a = self.discriminator.train_on_batch([batch, batch], labels_real)
                loss_disc_b = self.discriminator.train_on_batch([batch, batch_disc], labels_real)
                loss_disc_c = self.discriminator.train_on_batch([batch, batch_disc_roll], labels_fake)
                loss_disc_d = self.discriminator.train_on_batch([batch, generated], labels_fake)

                print("Epoch: %d [Disc. loss_a: %f, loss_b: %f, loss_c: %f, loss_d: %f %%]" % (epoch, loss_disc_a[0], loss_disc_b[0], loss_disc_c[0], loss_disc_d[0]))

            if train_gen:
                # loss_gen = self.combined.test_on_batch([batch, batch_disc], [labels_real_g, labels_fake_g])
                # print("Epoch: %d [Gen ->real test. loss: %f, acc.: %.2f%%]" % (epoch, loss_gen[0], 100 * loss_gen[1]))

                loss_gen = self.combined.train_on_batch([batch, batch], labels_real_g)
                print("Epoch: %d [Gen_same. loss: %f, acc.: %.2f%%]" % (epoch, loss_gen[0], 100 * loss_gen[1]))

                loss_gen = self.combined.train_on_batch([batch, batch_disc], labels_real_g)
                print("Epoch: %d [Gen_sim. loss: %f, acc.: %.2f%%]" % (epoch, loss_gen[0], 100 * loss_gen[1]))


            # if loss_disc[0] < 0.25:
            #     train_gen = True
            # if loss_gen[0] < 0.25:
            #     train_disc = True
            #
            # if loss_disc[0] > 0.4:
            #     train_gen = False
            # if loss_gen[0] > 0.4:
            #     train_disc = False
            #
            # if train_gen == False and train_disc == False:
            #     train_disc = True



            # Save interval
            if epoch % save_interval == 0:
                if epoch == 0:
                    for i in range(len(indices)):
                        ims = np.reshape(batch_test[(i)*int(self.batch_size / len(indices))], (160, 160))
                        plt.imsave('./images/'+str(i)+'/0_baseline.png', ims, cmap='gray')

                    # ims = np.reshape(batch_test[0], (160, 160))
                    # plt.imsave('./images/0/0_baseline.png',ims, cmap='gray')
                    # ims = np.reshape(batch_test[4], (160, 160))
                    # plt.imsave('./images/1/0_baseline.png', ims, cmap='gray')
                    # ims = np.reshape(batch_test[8], (160, 160))
                    # plt.imsave('./images/2/0_baseline.png', ims, cmap='gray')
                    # ims = np.reshape(batch_test[12], (160, 160))
                    # plt.imsave('./images/3/0_baseline.png', ims, cmap='gray')
                    # ims = np.reshape(batch_test[16], (160, 160))
                    # plt.imsave('./images/4/0_baseline.png', ims, cmap='gray')
                    # ims = np.reshape(batch_test[20], (160, 160))
                    # plt.imsave('./images/5/0_baseline.png', ims, cmap='gray')
                    # ims = np.reshape(batch_test[24], (160, 160))
                    # plt.imsave('./images/6/0_baseline.png', ims, cmap='gray')
                    # ims = np.reshape(batch_test[28], (160, 160))
                    # plt.imsave('./images/7/0_baseline.png', ims, cmap='gray')
                # TODO: logging
                ims = self.generator.predict(batch_test)
                for i in range(len(indices)):
                    imss = np.reshape(ims[(i)*int(self.batch_size / len(indices))], (160, 160))
                    plt.imsave('./images/' + str(i) +'/'+ str(epoch)+'.png', imss, cmap='gray')


                # imss = np.reshape(ims[0],(160,160))
                # plt.imsave('./images/0/'+str(epoch)+'.png', imss, cmap='gray')
                # imss = np.reshape(ims[4], (160, 160))
                # plt.imsave('./images/1/' + str(epoch) + '.png', imss, cmap='gray')
                # imss = np.reshape(ims[8], (160, 160))
                # plt.imsave('./images/2/' + str(epoch) + '.png', imss, cmap='gray')
                # imss = np.reshape(ims[12], (160, 160))
                # plt.imsave('./images/3/' + str(epoch) + '.png', imss, cmap='gray')
                # imss = np.reshape(ims[16], (160, 160))
                # plt.imsave('./images/4/' + str(epoch) + '.png', imss, cmap='gray')
                # imss = np.reshape(ims[20], (160, 160))
                # plt.imsave('./images/5/' + str(epoch) + '.png', imss, cmap='gray')
                # imss = np.reshape(ims[24], (160, 160))
                # plt.imsave('./images/6/' + str(epoch) + '.png', imss, cmap='gray')
                # imss = np.reshape(ims[28], (160, 160))
                # plt.imsave('./images/7/' + str(epoch) + '.png', imss, cmap='gray')

                self.generator.save(model_file)
