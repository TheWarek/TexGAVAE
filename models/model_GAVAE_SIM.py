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

        self.texdat = TexDAT(data_path, self.batch_size)
        self.texdat.load_data(only_paths=True)

        # Init TODO: research the middle layer
        self.m = self.batch_size #1 #50
        self.n_z = 128

        self.dropout = 0.3

        self.k_reg = None #regularizers.l2(0.01)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9)
        self.optimizer_disc = tf.keras.optimizers.Adam(lr=lr , beta_1=0.9) #SGD(lr=lr / 100.) #
        # self.optimizer = SGD(lr=lr, momentum=0.9, decay=1e-6, nesterov=True)
        # self.optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        self.loss = self.vae_loss
        self.reg = None #regularizers.l2(1e-3)

        # Example of custom loss
        custom_loss = self.new_custom_loss(0.5, 400)

        # Additional settings (VAE, GAN, generator, discriminator etc..)

        # VAE encoder part
        self.input = self.get_autoencoder()
        # # Encoder model, to encode input into latent variable
        # # We use the mean as the output as it is the center point, the representative of the gaussian
        # self.vae_enc = tf.keras.models.Model(self.input, self.mu, name='vae_enc')
        #
        # # VAE decoder part
        # # Generator model, generate new data given latent variable z
        # # 1. define input for decoder
        # input_dec = tf.keras.layers.Input(shape=(self.n_z,))
        # layer_list = self.get_vae_decoder_part()
        # for i, l in enumerate(layer_list):
        #     if i == 0:
        #         l_1 = l(input_dec)
        #     else:
        #         l_1 = l(l_1)
        #
        # self.vae_dec = tf.keras.models.Model(input_dec, l_1, name='vae_dec')
        #
        # # Complete VAE model
        # # we need to connect it into one model
        # # combined encoder + decoder (so we can backpropagate with some loss)
        # # TODO: Merge with GAN
        # for i, l in enumerate(layer_list):
        #     if i == 0:
        #         l_1 = l(self.z)
        #     else:
        #         l_1 = l(l_1)
        #
        # with tf.variable_scope('vae_complete'):
        #     self.vae_complete = tf.keras.models.Model(self.input, l_1, name='vae_complete')
        #
        # #160x160
        #
        # # self.vae_complete.compile(loss=self.r2_keras,
        # #                            optimizer=self.optimizer,
        # #                            metrics=['accuracy'])
        #
        # # Implementation with GAN:
        #
        # # Discriminator part (whole model to train)
        # disc_input, disc_output = self.get_discriminator()
        # # create discriminator model
        # self.discriminator = tf.keras.models.Model(disc_input, disc_output, name='discriminator')
        # self.discriminator.compile(loss=self.disc_loss,
        #                            optimizer=self.optimizer_disc,
        #                            metrics=['accuracy'])
        #
        # # Combined model = generator part which train on discriminator loss
        # # For the combined model we will only train the generator
        # #self.discriminator.trainable = False
        #
        # #another discriminator just to set trainable false without discrepancy
        # # create discriminator model
        # self.discriminator2 = tf.keras.models.Model(disc_input, disc_output, name='discriminator2')
        # self.discriminator2.trainable = False
        # self.discriminator2.compile(loss=self.disc_loss,
        #                            optimizer=self.optimizer_disc,
        #                            metrics=['accuracy'])
        #
        # # self.discriminator.compile(loss=self.disc_loss,
        # #                            optimizer=self.optimizer_disc,
        # #                            metrics=['accuracy'])
        #
        # # The valid takes generated images as input and determines validity
        # self.input2 = tf.keras.layers.Input(batch_shape=self.batch_img_shape)
        # # img = self.vae_complete(self.input2)
        # img = self.vae_complete(self.input)
        # input2 = tf.keras.layers.Concatenate(axis=-1)([self.input, img])
        # # input2 = K.concatenate([self.input, img], axis=-1)
        #
        # valid = self.discriminator2(input2)
        #
        # # self.generator_combined = Model([self.input, self.input2], valid, name='generator_combined')
        # self.generator_combined = tf.keras.models.Model(self.input, valid, name='generator_combined')
        # self.generator_combined.compile(loss=self.gen_loss,
        #                            optimizer=self.optimizer,
        #                            metrics=['accuracy'])
        #
        # #self.discriminator.summary()
        # #self.generator_combined.summary()
        # #self.vae_complete.summary()
        # #self.vae_enc.summary()
        #
        # log_path = './logs'
        # # TODO : how to using tensorflow implementation of keras
        # self.callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0,
        #                             write_graph=True, write_images=False)
        # self.callback.set_model(self.generator_combined)


    def vae_loss(self, y_true, y_pred):
        # compute the average MSE error, then scale it up, ie. simply sum on all axes
        # reconstruction_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))
        # compute the KL loss
        kl_loss = 1 / (- 0.5 * tf.keras.backend.mean(1 + self.log_sigma - tf.keras.backend.square(self.mu) -
                                                     tf.keras.backend.square(tf.keras.backend.exp(self.log_sigma)), axis=-1))
        # return the average loss over all images in batch
        total_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
        return total_loss

    def gen_loss(self, y_true, y_pred):
        # compute the binary crossentropy
        # reconstruction_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))
        reconstruction_loss2 = 1. / tf.keras.backend.abs(tf.keras.backend.log(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)) + 0.00001) / tf.keras.backend.log(10.))

        # compute the KL loss
        kl_loss = 1 / (- 0.5 * tf.keras.backend.mean(1 + self.log_sigma - tf.keras.backend.square(self.mu) - tf.keras.backend.square(tf.keras.backend.exp(self.log_sigma)), axis=-1))
        # return the average loss over all images in batch
        total_loss = tf.keras.backend.mean(reconstruction_loss2 + kl_loss * 0.)
        # total_loss_log = - K.log(1 - total_loss)
        return reconstruction_loss

    def disc_loss(self, y_true, y_pred):
        # reconstruction_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))
        # reconstruction_loss_log = - K.log(1 - reconstruction_loss)
        # 1 / abs(log)
        reconstruction_loss2 = 1. / tf.keras.backend.abs(tf.keras.backend.log(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)) + 0.00001) / tf.keras.backend.log(10.))
        return reconstruction_loss

    def r2_keras(self, y_true, y_pred):
        SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
        SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
        # return (1 - SS_res / (SS_tot + K.epsilon()))
        return tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true) - y_pred * 0.))

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

        # slice tensor up

        a = tf.keras.layers.Flatten()(tf.slice(d_4, [0, 0, 0, 0], [self.batch_size, 160, 1, 1]))
        a_all = tf.keras.layers.Dense(1)(a)
        b = tf.keras.layers.Flatten()(tf.slice(d_4, [0, 0, 0, 0], [self.batch_size, 1, 160, 1]))
        b_all = tf.keras.layers.Dense(1)(b)

        for i in range(1,160):
            a = tf.keras.layers.Flatten()(tf.slice(d_4, [0, 0, i, 0], [self.batch_size, 160, 1, 1]))
            d_a = tf.keras.layers.Dense(1)(a)
            a_all = tf.keras.layers.concatenate([a_all, d_a], axis=1)
            b = tf.keras.layers.Flatten()(tf.slice(d_4, [0, i, 0, 0], [self.batch_size, 1, 160, 1]))
            d_b = tf.keras.layers.Dense(1)(b)
            b_all = tf.keras.layers.concatenate([b_all, d_b], axis=1)
            #b_all = tf.keras.layers.Dense(1)(b)

        a_all = tf.reshape(a_all,[self.batch_size, 160, 1])
        b_all = tf.reshape(b_all,[self.batch_size, 160, 1])
        b_all = tf.transpose(b_all)

        # TODO: MATRIX MULTIPLICATION WITH TRANSPOSE MATRIX
        # TODO: DO SOMETHING WITH YOUR LIFE
        test = tf.linalg.matmul(b_all, a_all)

        return input

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
        indices = [46, 103, 137, 195, 15, 69, 125, 180]
        # indices = [46, 103, 137, 195]
        # indices = [46, 195]
        # indices = [46]
        #indices = [46, 195]
        for ind in indices:
            for i in range(int(self.batch_size/len(indices))):
                batch_test.append(self.texdat.read_segment(sorted_list[ind][1].paths[0]))
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

            batch = []
            batch_disc = []
            for ind in indices:
                for i in range(int(self.batch_size / len(indices))):
                    # batch.append(self.texdat.read_image_patch(sorted_list[ind][1].paths[0], patch_size=self.patch_size))
                    batch.append(self.texdat.read_segment(sorted_list[ind][1].paths[0]))
                    batch_disc.append(self.texdat.read_segment(sorted_list[ind][1].paths[0]))
            batch = resize_batch_images(batch, self.patch_size)
            batch_disc = resize_batch_images(batch_disc, self.patch_size)
            batch = normalize_batch_images(batch, 'zeromean')
            batch_disc = normalize_batch_images(batch_disc, 'zeromean')

            batch_cat_real = np.concatenate([batch, batch_disc], axis=-1)

            generated = self.vae_complete.predict(batch)

            batch_cat_gen = np.concatenate([batch, generated], axis=-1)

            # batch_discriminator = np.concatenate((batch, generated))
            # labels = np.concatenate((np.ones(self.batch_size, np.float32), (np.zeros(self.batch_size, np.float32))))
            # loss_disc = self.discriminator.train_on_batch(batch_discriminator,labels)
            # print("Epoch: %d [Disc. loss: %f, acc.: %.2f%%]" % (epoch, loss_disc[0], 100 * loss_disc[1]))

            # IMPROVE 2. use noisy labels
            # IMPORVEMENT other way around -> real =0, fake = 1, better for gradient flow
            # IMPROVEMENT noise not required when training generator
            labels_fake = np.ones(self.batch_size, np.float32) + np.subtract(np.multiply(np.random.rand(self.batch_size), 0.2 ), 0.1)
            labels_fake_g = np.ones(self.batch_size, np.float32)
            labels_real = np.zeros(self.batch_size, np.float32) + np.multiply(np.random.rand(self.batch_size), 0.2 )
            labels_real_g = np.zeros(self.batch_size, np.float32)

            # IMPROVE 3. train gen/disc depending on each other:

            if train_disc:
                # IMPROVE 1. mini-batches of REAL / FAKE
                # loss_real = self.discriminator.train_on_batch(batch_cat_real, labels_real)
                # loss_fake = self.discriminator.train_on_batch(batch_cat_gen, labels_fake)
                # loss_disc = 0.5 * np.add(loss_real, loss_fake)
                # print("Epoch: %d [Disc real. loss: %f, acc.: %.2f%%]" % (epoch, loss_real[0], 100 * loss_real[1]))
                # print("Epoch: %d [Disc fake. loss: %f, acc.: %.2f%%]" % (epoch, loss_fake[0], 100 * loss_fake[1]))

                # improve 2 - MERGE

                batch_cat_all = np.concatenate([batch_cat_real, batch_cat_gen], axis=0)
                labels_all = np.concatenate([labels_real, labels_fake], axis=0)
                loss_disc = self.discriminator.train_on_batch(batch_cat_all, labels_all)
                print("Epoch: %d [Disc. loss: %f, acc.: %.2f%%]" % (epoch, loss_disc[0], 100 * loss_disc[1]))

                # loss_disc = self.discriminator.test_on_batch(batch_cat_gen, labels_real)
                # print("Epoch: %d [Disc gen->real test. loss: %f, acc.: %.2f%%]" % (epoch, loss_disc[0], 100 * loss_disc[1]))
                # loss_disc = self.discriminator.test_on_batch(batch_cat_gen, labels_fake)
                # print("Epoch: %d [Disc gen->fake test. loss: %f, acc.: %.2f%%]" % (
                # epoch, loss_disc[0], 100 * loss_disc[1]))


            if train_gen:
                loss_gen = self.generator_combined.test_on_batch(batch, labels_real_g)
                print("Epoch: %d [Gen ->real test. loss: %f, acc.: %.2f%%]" % (epoch, loss_gen[0], 100 * loss_gen[1]))
                # imgs, test = self.generator_combined.predict_on_batch(batch)
                # #print(test)
                loss_gen = self.generator_combined.train_on_batch(batch, labels_real_g)
                print("Epoch: %d [Gen. loss: %f, acc.: %.2f%%]" % (epoch, loss_gen[0], 100 * loss_gen[1]))

            if loss_disc[0] < 0.25:
                train_gen = True
            if loss_gen[0] < 0.25:
                train_disc = True

            if loss_disc[0] > 0.4:
                train_gen = False
            if loss_gen[0] > 0.4:
                train_disc = False

            if train_gen == False and train_disc == False:
                train_disc = True

            # we will train generator until it will overcome 1.5* of disc loss
            # as generator takes usually more time to train
            # if loss_gen[0] <= 2.5 * loss_disc[0]:
            #     train_disc = True
            # else:
            #     train_disc = False


            # Save interval
            if epoch % save_interval == 0:
                mu = self.vae_enc.predict(batch_test)
                if epoch == 0:
                    for i in range(len(indices)):
                        ims = np.reshape(batch_test[(i)*int(self.batch_size / len(indices))], (160, 160))
                        plt.imsave('./images/'+str(i)+'/0_baseline.png', ims, cmap='gray')


                # TODO: logging
                ims = self.vae_complete.predict(batch_test)
                for i in range(len(indices)):
                    imss = np.reshape(ims[(i)*int(self.batch_size / len(indices))], (160, 160))
                    plt.imsave('./images/' + str(i) +'/'+ str(epoch)+'.png', imss, cmap='gray')

                self.generator_combined.save(model_file)
