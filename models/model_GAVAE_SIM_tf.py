from .model import ModelGAVAE
import tensorflow as tf

from scipy.misc import imread
import os.path

from matplotlib import pyplot as plt

import numpy as np

from data_loader import TexDAT
from data_loader import resize_batch_images, normalize_batch_images

class GAVAE_SIM(ModelGAVAE):
    def __init__(self, data_path, w, h, c, layer_depth, batch_size=32, lr=0.0002, margin=4.5):
        super(GAVAE_SIM, self).__init__(w, h, c, layer_depth, batch_size)
        self.patch_size = (w,h,c)

        self.texdat = TexDAT(data_path, self.batch_size)
        self.texdat.load_images(False)

        self.m = batch_size
        self.margin = margin
        self.n_z = 400
        self.drop_rate = tf.placeholder(tf.float32, None, name='dropout_rate')

        # self.disc_gt = tf.placeholder(tf.float32, [None, ], name="disc_gt")
        self.vae_input = tf.placeholder(dtype=tf.float32, shape=[None, w, h, c], name="vae_input")
        self.dec_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_z], name="dec_input")

        self.disc_summaries = []
        self.vae_summaries = []

        with tf.variable_scope('encoders') as scope:
            self.mu_1, self.log_sigma_1, self.z_1 = self.get_vae_encoder_part(self.vae_input, True)  # encoder at the beginning
            with tf.variable_scope('decoder') as dec_scope:
                self.vae_output = self.get_vae_decoder_part(self.z_1)
                dec_scope.reuse_variables()
                self.dec_output = self.get_vae_decoder_part(self.dec_input)
            scope.reuse_variables()
            self.mu_2, self.log_sigma_2, self.z_2 = self.get_vae_encoder_part(self.vae_output, False)

        # with tf.variable_scope('discriminator') as scope:
        #     self.discriminator = self.get_discriminator(self.vae_input, True)
        #     scope.reuse_variables()
        #     self.gavae = self.get_discriminator(self.vae_output, False)

        # self.gen_loss = self.gen_loss(self.disc_gt, self.gavae)
        # self.disc_loss = self.disc_loss(self.disc_gt, self.discriminator)

        self.vae_loss = self.__vae_loss()

        with tf.variable_scope('optimizers'):
            # self.disc_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9)
            self.vae_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            self.vae_global_step = tf.Variable(initial_value=0, name='vae_global_step', trainable=False)
            self.vae_train_step = self.vae_optimizer.minimize(
                loss=self.vae_loss,
                global_step=self.vae_global_step
            )
            self.disc_global_step = tf.Variable(initial_value=0, name='disc_global_step', trainable=False)
            self.disc_train_step = self.disc_optimizer.minimize(
                loss=self.disc_loss,
                global_step=self.disc_global_step
            )


    def __vae_loss(self):
        # compute the average MSE error, then scale it up, ie. simply sum on all axes
        # mse_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        with tf.variable_scope('vae_loss'):
            reconstruction_loss = tf.sqrt(tf.reduce_sum(tf.square(self.vae_input - self.vae_output)))
            self.vae_summaries.append(tf.summary.scalar('recon_loss', reconstruction_loss))
            # compute the KL loss
            kl_loss = - 0.5 * tf.reduce_mean(1 + self.log_sigma_1 - tf.square(self.mu_1) - tf.square(tf.exp(self.log_sigma_1)), axis=-1)
            self.vae_summaries.append(tf.summary.scalar('kl_loss', tf.reduce_mean(kl_loss)))
            # compute encoder loss
            encoder_loss = tf.sqrt(tf.reduce_sum(tf.square(self.z_1 - self.z_2)))
            self.vae_summaries.append(tf.summary.scalar('encoder_loss', encoder_loss))
            # return the average loss over all images in batch
            # kokot neotacaj ho, nechaj +, lebo uz je vynasobeny -1 on sam
            total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

            self.vae_summaries.append(tf.summary.scalar('total_loss', total_loss))
        return total_loss

    def __gen_loss(self, y_true, y_pred):
        # encoder loss
        with tf.variable_scope('generator_loss'):
            z_2 = self.sample_z([self.mu_2, self.log_sigma_2])
            # with tf.variable_scope('margin_loss'):
            #     self.margin_loss = tf.reduce_mean(tf.minimum(0., tf.sqrt(tf.reduce_sum(tf.square(z_2 - self.z_1),axis=1)) - self.margin)) # better name for margin loss
            #     self.vae_summaries.append(tf.summary.scalar('margin_loss', self.margin_loss))
            with tf.variable_scope('meansquare_loss'):
                # compute the binary crossentropy
                # trueseems_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
                self.meansquare_loss = tf.reduce_mean(tf.square(y_pred - y_true))
                self.vae_summaries.append(tf.summary.scalar('true_loss', self.meansquare_loss))
            with tf.variable_scope('reconstruction_loss'):
                reconstruction_loss = tf.sqrt(tf.reduce_sum(tf.square(self.vae_input - self.vae_output)))
                self.reconstruction_loss = tf.reduce_mean(reconstruction_loss) / (self.patch_size[0]*self.patch_size[1])
                self.vae_summaries.append(tf.summary.scalar('reconstruction_loss', self.reconstruction_loss))
            with tf.variable_scope('kl_divergence_loss'):
                # compute the KL loss - reduce_sum
                self.kl_loss = - 0.5 * tf.reduce_mean(1 + self.log_sigma_1 - tf.square(self.mu_1) - tf.square(tf.exp(self.log_sigma_1)), axis=-1)
                # self.vae_summaries.append(tf.summary.scalar('kl_loss', self.kl_loss))
            with tf.variable_scope('total_mean_loss'):
                # return the average loss over all images in batch
                total_loss = tf.reduce_mean(self.reconstruction_loss + self.kl_loss) #+ 0.02 * self.kl_loss + 0.7 * self.margin_loss + self.reconstruction_loss)
                self.vae_summaries.append(tf.summary.scalar('total_loss', total_loss))
        return total_loss

    def __disc_loss(self, y_true, y_pred):
        with tf.variable_scope('discriminant_loss'):
            # reconstruction_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
            reconstruction_loss = tf.reduce_mean(tf.square(y_pred - y_true))
            self.disc_summaries.append(tf.summary.scalar('discriminator_loss',reconstruction_loss))
        return reconstruction_loss

    def sample_z(self, args) -> tf.Tensor:
        with tf.variable_scope('sampling_z'):
            mu, log_sigma = args
            eps = tf.random_normal(shape=(self.m , self.n_z), mean=0., stddev=1.)
            # sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
            return mu + tf.exp(log_sigma / tf.constant(2, tf.float32)) * eps

    # vae encoder mu, log_sigma, sampled_z
    def get_vae_encoder_part(self, input, trainable) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.variable_scope('layer_1'):
            net_1 = tf.layers.conv2d(input, filters=16, kernel_size=5, strides=1,
                                     padding='same', data_format='channels_last', trainable=trainable,
                                     reuse=tf.AUTO_REUSE, name='conv_1')
            net_1 = tf.nn.leaky_relu(net_1, name='relu_1')
            net_1 = tf.layers.batch_normalization(net_1, momentum=0.8, trainable=trainable, reuse=tf.AUTO_REUSE, name='bn_1')
            net_1 = tf.layers.dropout(net_1, self.drop_rate, name='dropout_1')

        with tf.variable_scope('layer_2'):
            net_2 = tf.layers.conv2d(net_1, filters=32, kernel_size=5, strides=2,
                                     padding='same', data_format='channels_last', trainable=trainable,
                                     reuse=tf.AUTO_REUSE, name='conv_2')
            net_2 = tf.nn.leaky_relu(net_2, name='relu_2')
            net_2 = tf.layers.batch_normalization(net_2, momentum=0.8, trainable=trainable, reuse=tf.AUTO_REUSE, name='bn_2')
            net_2 = tf.layers.dropout(net_2, self.drop_rate, name='dropout_2')

        with tf.variable_scope('layer_3'):
            net_3 = tf.layers.conv2d(net_2, filters=64, kernel_size=5, strides=2,
                                     padding='same', data_format='channels_last', trainable=trainable,
                                     reuse=tf.AUTO_REUSE, name='conv_3')
            net_3 = tf.nn.leaky_relu(net_3, name='relu_3')
            net_3 = tf.layers.batch_normalization(net_3, momentum=0.8, trainable=trainable, reuse=tf.AUTO_REUSE, name='bn_3')
            net_3 = tf.layers.dropout(net_3, self.drop_rate, name='dropout_3')

        with tf.variable_scope('layer_4'):
            net_4 = tf.layers.conv2d(net_3, filters=128, kernel_size=5, strides=2,
                                     padding='same', data_format='channels_last', trainable=trainable,
                                     reuse=tf.AUTO_REUSE, name='conv_4')
            net_4 = tf.nn.leaky_relu(net_4, name='relu_4')
            net_4 = tf.layers.batch_normalization(net_4, momentum=0.8, trainable=trainable, reuse=tf.AUTO_REUSE, name='bn_4')
            net_4 = tf.layers.dropout(net_4, self.drop_rate, name='dropout_5')

        # with tf.variable_scope('layer_5'):
        #     net_5 = tf.layers.conv2d(net_4, filters=64, kernel_size=1, strides=1,
        #                              padding='same', data_format='channels_last', trainable=trainable,
        #                              reuse=tf.AUTO_REUSE, name='conv_5')
        #     net_5 = tf.nn.leaky_relu(net_5, name='relu_5')
        #     net_5 = tf.layers.batch_normalization(net_5, momentum=0.8, trainable=trainable, reuse=tf.AUTO_REUSE, name='bn_5')
        #     net_5 = tf.layers.dropout(net_5, self.drop_rate, name='dropout_6')
        #
        # with tf.variable_scope('layer_6'):
        #     net_6 = tf.layers.conv2d(net_5, filters=32, kernel_size=1, strides=1,
        #                              padding='same', data_format='channels_last', trainable=trainable,
        #                              reuse=tf.AUTO_REUSE, name='conv_6')
        #     net_6 = tf.nn.leaky_relu(net_6, name='relu_6')
        #     net_6 = tf.layers.batch_normalization(net_6, momentum=0.8, trainable=trainable, reuse=tf.AUTO_REUSE, name='bn_6')
        #     net_6 = tf.layers.dropout(net_6, self.drop_rate, name='dropout_7')
        #
        # with tf.variable_scope('layer_7'):
        #     net_7 = tf.layers.conv2d(net_6, filters=16, kernel_size=1, strides=1,
        #                              padding='same', data_format='channels_last', trainable=trainable,
        #                              reuse=tf.AUTO_REUSE, name='conv_7')
        #     net_7 = tf.nn.leaky_relu(net_7, name='relu_7')
        #     net_7 = tf.layers.batch_normalization(net_7, momentum=0.8, trainable=trainable, reuse=tf.AUTO_REUSE, name='bn_7')
        #     net_7 = tf.layers.dropout(net_7, self.drop_rate, name='dropout_7')

        flat = tf.layers.flatten(net_4, name='flatten')

        mu = tf.layers.dense(flat, self.n_z, activation=None, trainable=trainable,
                                 reuse=tf.AUTO_REUSE, name='mu')
        log_sigma = tf.layers.dense(flat, self.n_z, activation=None, trainable=trainable,
                                 reuse=tf.AUTO_REUSE, name='log_sigma')

        z = self.sample_z([mu, log_sigma])

        return mu, log_sigma, z

    def get_vae_decoder_part(self, input) -> tf.Tensor:
        with tf.variable_scope('layer_1'):
            net_1 = tf.layers.dense(input, self.mid_shape[0] * self.mid_shape[1], reuse=tf.AUTO_REUSE)
            net_1 = tf.nn.leaky_relu(net_1)
            net_1 = tf.layers.batch_normalization(net_1, momentum=0.8,reuse=tf.AUTO_REUSE)

        reshape = tf.reshape(net_1, (self.m, self.mid_shape[0], self.mid_shape[1], self.mid_shape[2]))

        with tf.variable_scope('layer_2'):
            net_2 = tf.layers.conv2d(reshape, filters=16, kernel_size=5, strides=1,
                                     padding='same', data_format='channels_last', reuse=tf.AUTO_REUSE)
            net_2 = tf.nn.leaky_relu(net_2)
            net_2 = tf.layers.batch_normalization(net_2, momentum=0.8, reuse=tf.AUTO_REUSE)
            net_2 = tf.layers.dropout(net_2, self.drop_rate)

        with tf.variable_scope('layer_3'):
            net_3 = tf.image.resize_images(net_2, size=(2 * self.mid_shape_16[0], 2 * self.mid_shape_16[1]))
            net_3 = tf.layers.conv2d(net_3, filters=32, kernel_size=5, strides=1,
                                     padding='same', data_format='channels_last', reuse=tf.AUTO_REUSE)
            net_3 = tf.nn.leaky_relu(net_3)
            net_3 = tf.layers.batch_normalization(net_3, momentum=0.8, reuse=tf.AUTO_REUSE)
            net_3 = tf.layers.dropout(net_3, self.drop_rate)

        with tf.variable_scope('layer_4'):

            net_4 = tf.image.resize_images(net_3, size=(4 * self.mid_shape_16[0], 4 * self.mid_shape_16[1]))
            net_4 = tf.layers.conv2d(net_4, filters=64, kernel_size=5, strides=1,
                                     padding='same', data_format='channels_last', reuse=tf.AUTO_REUSE)
            net_4 = tf.nn.leaky_relu(net_4)
            net_4 = tf.layers.batch_normalization(net_4, momentum=0.8)
            net_4 = tf.layers.dropout(net_4, self.drop_rate)

        with tf.variable_scope('layer_5'):
            net_5 = tf.image.resize_images(net_4, size=(8 * self.mid_shape_16[0], 8 * self.mid_shape_16[1]))
            net_5 = tf.layers.conv2d(net_5, filters=128, kernel_size=5, strides=1,
                                     padding='same', data_format='channels_last', reuse=tf.AUTO_REUSE)
            net_5 = tf.nn.leaky_relu(net_5)
            net_5 = tf.layers.batch_normalization(net_5, momentum=0.8, reuse=tf.AUTO_REUSE)
            net_5 = tf.layers.dropout(net_5, self.drop_rate)

        with tf.variable_scope('layer_6'):
            # net_6 = tf.image.resize_images(net_5, size=(8 * self.mid_shape_16[0], 8 * self.mid_shape_16[1]))
            net_6 = tf.layers.conv2d(net_5, filters=1, kernel_size=1, strides=1,
                                               padding='same', data_format='channels_last', activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE)
        #     net_6 = tf.nn.leaky_relu(net_6)
        #     net_6 = tf.layers.batch_normalization(net_6, momentum=0.8)
        #     net_6 = tf.layers.dropout(net_6, self.drop_rate)
        #
        # with tf.variable_scope('layer_7'):
        #     net_7 = tf.layers.conv2d(net_6, filters=64, kernel_size=3, strides=1,
        #                                        padding='same', data_format='channels_last')
        #     net_7 = tf.nn.leaky_relu(net_7)
        #     net_7 = tf.layers.batch_normalization(net_7, momentum=0.8)
        #
        # with tf.variable_scope('layer_8'):
        #     net_8 = tf.layers.conv2d(net_7, filters=32, kernel_size=3, strides=1,
        #                                        padding='same', data_format='channels_last')
        #     net_8 = tf.nn.leaky_relu(net_8)
        #     net_8 = tf.layers.batch_normalization(net_8, momentum=0.8)
        #
        # with tf.variable_scope('out_layer'):
        #     out = tf.layers.conv2d(net_8, filters=1, kernel_size=1, strides=1,
        #                            padding='same', data_format='channels_last', activation=tf.nn.sigmoid)
        out = net_6
        return out

    def get_discriminator(self, input, trainable):
        with tf.variable_scope('layer_1'):
            net_1 = tf.layers.conv2d(input, filters=32, kernel_size=3, strides=2,
                                     padding='same', data_format='channels_last',
                                     reuse=tf.AUTO_REUSE, trainable=trainable, name='conv_1')
            net_1 = tf.nn.leaky_relu(net_1, name='l_relu_1')
            net_1 = tf.layers.batch_normalization(net_1, momentum=0.8, reuse=tf.AUTO_REUSE, trainable=trainable, name="bn_1")
            net_1 = tf.layers.dropout(net_1, self.drop_rate, name='dr_1')

        with tf.variable_scope('layer_2'):
            net_2 = tf.layers.conv2d(net_1, filters=48, kernel_size=5, strides=2,
                                     padding='same', data_format='channels_last',
                                     reuse=tf.AUTO_REUSE, trainable=trainable, name='conv_2')
            net_2 = tf.nn.leaky_relu(net_2, name='l_relu_2')
            net_2 = tf.layers.batch_normalization(net_2, momentum=0.8, reuse=tf.AUTO_REUSE, trainable=trainable, name="bn_2")
            net_2 = tf.layers.dropout(net_2, self.drop_rate, name='dr_2')

        with tf.variable_scope('layer_3'):
            net_3 = tf.layers.conv2d(net_2, filters=64, kernel_size=5, strides=2,
                                     padding='same', data_format='channels_last',
                                     reuse=tf.AUTO_REUSE, trainable=trainable, name='conv_3')
            net_3 = tf.nn.leaky_relu(net_3, name='l_relu_3')
            net_3 = tf.layers.batch_normalization(net_3, momentum=0.8, reuse=tf.AUTO_REUSE, trainable=trainable, name="bn_3")
            net_3 = tf.layers.dropout(net_3, self.drop_rate, name='dr_3')

        with tf.variable_scope('layer_4'):
            net_4 = tf.layers.conv2d(net_3, filters=32, kernel_size=1, strides=1,
                                     padding='same', data_format='channels_last',
                                     reuse=tf.AUTO_REUSE, trainable=trainable, name='conv_4')
            net_4 = tf.nn.leaky_relu(net_4, name='l_relu_4')
            net_4 = tf.layers.batch_normalization(net_4, momentum=0.8, reuse=tf.AUTO_REUSE, trainable=trainable, name="bn_4")
            net_4 = tf.layers.dropout(net_4, self.drop_rate, name='dr_4')

        flat = tf.layers.flatten(net_4, 'flat')
        dense = tf.layers.dense(flat, 1, activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, trainable=trainable, name="dense_1")

        return dense

    def train(self, epochs, model_file, save_interval=50, log_interval=20):
        sorted_list = self.texdat.train.images

        # batch_test = []
        # indices = [31, 100, 120, 198]
        # for ind in indices:
        #     for i in range(int(self.batch_size / 4)):
        #         batch_test.append(
        #             self.texdat.load_image_patch(sorted_list[ind], patch_size=self.patch_size))
        # batch_test = resize_batch_images(batch_test, self.patch_size)

        save_path = 'model/' + model_file + '/'
        saver = tf.train.Saver(max_to_keep=5)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_ckpt = os.path.join(save_path, 'checkpoint')

        with tf.Session() as sess:
            if os.path.exists(model_ckpt):
                # restore checkpoint if it exists
                try:
                    print("Trying to restore last checkpoint ...")
                    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path)
                    saver.restore(sess, save_path=last_chk_path)
                    print("Restored checkpoint from:", last_chk_path)
                except:
                    print("Failed to restore checkpoint. Initializing variables instead.")
                    sess.run(tf.global_variables_initializer())
            else:
                sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter('logs/' + model_file + '/', graph=sess.graph)

            # summary_disc_1 = tf.summary.merge(self.disc_summaries)
            summary_vae = tf.summary.merge(self.vae_summaries)

            early_stop_counter = 0
            for epoch in range(epochs):
                print('Epoch {:d}'.format(epoch))
                batch = []
                indices = np.random.choice(np.arange(len(sorted_list)), size=4, replace=False)
                for ind in indices:
                    for i in range(int(self.batch_size / 4)):
                        batch.append(
                            self.texdat.load_image_patch(sorted_list[ind], patch_size=self.patch_size))
                batch = resize_batch_images(batch, self.patch_size)

                # labels_real = np.ones(self.batch_size, np.float32) + np.subtract(np.multiply(np.random.rand(self.batch_size), 0.3), 0.15)
                # labels_fake = np.zeros(self.batch_size, np.float32) + np.multiply(np.random.rand(self.batch_size), 0.3)
                #
                # vae_output = self.vae_output.eval({self.vae_input: batch})

                # training the discriminant
                #                 _, discriminant_loss, merged = sess.run([self.disc_train_step, self.disc_loss, summary_disc_1], feed_dict={
                #                     self.vae_input : batch,
                #                     self.disc_gt : labels_real,
                #                     self.drop_rate: 0.6
                #                 })
                #                 if epoch % save_interval == 0:
                #                     writer.add_summary(merged, global_step=epoch)
                #                 print('Disc. on real {:.6f}'.format(discriminant_loss))
                #                 _, discriminant_loss, merged = sess.run([self.disc_train_step, self.disc_loss, summary_disc_1], feed_dict={
                #                     self.vae_input : vae_output,
                #                     self.disc_gt : labels_fake,
                #                     self.drop_rate: 0.6
                #                 })
                #                 if epoch % save_interval == 0:
                #                     writer.add_summary(merged, global_step=epoch)
                #                 print('Disc. on fake {:.6f}'.format(discriminant_loss))

                # training the GAVAE
                _, gavae_loss, merged = sess.run([self.vae_train_step, self.vae_loss, summary_vae], feed_dict={
                    self.vae_input : batch,
                    self.drop_rate : 0
                })
                print('Gavae {:.6f}'.format(gavae_loss))

                # regularization for early stopping
                if gavae_loss > 1000 or gavae_loss < -1000:
                    if early_stop_counter > 5:
                        print("Stopping the training due to high loss.")
                        print("Unable to change gradient for the 5th time")
                        break
                    early_stop_counter += 1
                    try:
                        print(["VAE loss to high", "Trying to restore last checkpoint ..."], sep='\n')
                        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path)
                        idx = last_chk_path.rfind('-')
                        restore_point = last_chk_path[:idx+1] + str(int(last_chk_path[idx+1:])-200)
                        saver.restore(sess, save_path=restore_point)
                        print("Restored checkpoint from:", restore_point)
                    except:
                        print("Stopping the training due to high loss.")
                        break

# log + test for results
                if epoch % save_interval == 0:
                    save_path_local = saver.save(sess=sess, save_path=model_ckpt, global_step=epoch)
                    print('Model saved to file as {:s}'.format(save_path_local))

                if epoch % log_interval == 0:
                    writer.add_summary(merged, global_step = epoch)
                if epoch % log_interval == 0:
                    if epoch == 0:
                        if not os.path.exists('./images/0'):
                            os.makedirs('./images/0')
                        if not os.path.exists('./images/1'):
                            os.makedirs('./images/1')
                        if not os.path.exists('./images/2'):
                            os.makedirs('./images/2')
                        if not os.path.exists('./images/3'):
                            os.makedirs('./images/3')

                    ims = np.reshape(batch[0], (160, 160))
                    plt.imsave('./images/0/'+str(epoch)+'_0_baseline.png', ims, cmap='gray')
                    ims = np.reshape(batch[8], (160, 160))
                    plt.imsave('./images/1/'+str(epoch)+'_8_baseline.png', ims, cmap='gray')
                    ims = np.reshape(batch[16], (160, 160))
                    plt.imsave('./images/2/'+str(epoch)+'_16_baseline.png', ims, cmap='gray')
                    ims = np.reshape(batch[24], (160, 160))
                    plt.imsave('./images/3/'+str(epoch)+'_24_baseline.png', ims, cmap='gray')
                    # TODO: logging
                    latent = self.z_1.eval(feed_dict={
                        self.vae_input: batch
                    })
                    latent += 1.3
                    generated = self.dec_output.eval(feed_dict={
                        self.dec_input: latent
                    })
                    ims = self.vae_output.eval(feed_dict={
                        self.vae_input : batch,
                        self.drop_rate: 1
                    })
                    imss = np.reshape(ims[0], (160, 160))
                    plt.imsave('./images/0/' + str(epoch) + '.png', imss, cmap='gray')
                    imss = np.reshape(ims[8], (160, 160))
                    plt.imsave('./images/1/' + str(epoch) + '.png', imss, cmap='gray')
                    imss = np.reshape(ims[16], (160, 160))
                    plt.imsave('./images/2/' + str(epoch) + '.png', imss, cmap='gray')
                    imss = np.reshape(ims[24], (160, 160))
                    plt.imsave('./images/3/' + str(epoch) + '.png', imss, cmap='gray')

                    imss = np.reshape(generated[0], (160, 160))
                    plt.imsave('./images/0/' + str(epoch) + '_z.png', imss, cmap='gray')
                    imss = np.reshape(generated[8], (160, 160))
                    plt.imsave('./images/1/' + str(epoch) + '_z.png', imss, cmap='gray')
                    imss = np.reshape(generated[16], (160, 160))
                    plt.imsave('./images/2/' + str(epoch) + '_z.png', imss, cmap='gray')
                    imss = np.reshape(generated[24], (160, 160))
                    plt.imsave('./images/3/' + str(epoch) + '_z.png', imss, cmap='gray')

    def test(self, model_file):
        sorted_list = self.texdat.train.images

        restore_path = 'model/' + model_file + '/'
        saver = tf.train.Saver(max_to_keep=5)
        if not os.path.exists(restore_path):
            print("Unable to find restore path, ending test")
            return
        model_ckpt = os.path.join(restore_path, 'checkpoint')

        with tf.Session() as sess:
            if os.path.exists(model_ckpt):
                # restore checkpoint if it exists
                try:
                    print("Trying to restore last checkpoint ...")
                    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=restore_path)
                    saver.restore(sess, save_path=last_chk_path)
                    print("Restored checkpoint from:", last_chk_path)
                except:
                    print("Failed to restore model checkpoint. Ending test.")
                    return
            else:
                print("Failed to restore model. Ending test")
                return
            if not os.path.exists('./test-images/'):
                os.makedirs('./test-images/')


            for j in range(int(len(sorted_list) / 32)):
                batch_1 = []
                for i in range(j * 32, j * 32 + 32):
                    batch_1.append(self.texdat.load_image_patch(sorted_list[i], patch_size=self.patch_size))
                batch_1 = resize_batch_images(batch_1, self.patch_size)

                batch_2 = []
                for i in range(j * 32, j * 32 + 32):
                    batch_2.append(self.texdat.load_image_patch(sorted_list[i], patch_size=self.patch_size))
                batch_2 = resize_batch_images(batch_2, self.patch_size)

                latent_1 = self.z_1.eval(feed_dict={
                    self.vae_input: batch_1,
                    self.drop_rate: 0
                })

                latent_2 = self.z_1.eval(feed_dict={
                    self.vae_input: batch_2,
                    self.drop_rate: 0
                })
                B = (latent_2 - latent_1) / 6
                for t in range(7):
                    latent = latent_1+B*t

                    ims = self.dec_output.eval(feed_dict={
                        self.dec_input: latent,
                        self.drop_rate: 0
                    })
                    c = 0
                    for i in range(j * 32, j * 32 + 32):
                        batchs = np.reshape(batch_1[c], (160, 160))
                        plt.imsave('./test-images/index_' + str(i) + '_o_1.png', batchs, cmap='gray')
                        batchs = np.reshape(batch_2[c], (160, 160))
                        plt.imsave('./test-images/index_' + str(i) + '_o_2.png', batchs, cmap='gray')
                        imss = np.reshape(ims[c],(160,160))
                        plt.imsave('./test-images/index_' + str(i) + '_t_' + str(t) + '.png', imss, cmap='gray')
                        c = (c+1) % 32
