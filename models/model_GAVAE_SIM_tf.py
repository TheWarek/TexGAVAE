from .model import ModelGAVAE
import tensorflow as tf

from scipy import ndimage
import os.path

from matplotlib import pyplot as plt

import numpy as np

from data_loader import TexDAT
from data_loader import resize_batch_images, normalize_batch_images

class GAVAE_SIM(ModelGAVAE):
    def __init__(self, data_path, w, h, c, layer_depth, batch_size=32, lr=0.0002, margin=4.5):
        super(GAVAE_SIM, self).__init__(w, h, c, layer_depth, batch_size)
        self.patch_size = (w,h,c)

        self.new_batch = False
        self.test_phase = False

        if 'mnist' in str(data_path).lower():
            self.is_mnist = True
            self.is_textdat = False
            self.train_mnist, self.test_mnist = tf.keras.datasets.mnist.load_data(data_path)
            self.mnist_x = np.asarray(np.reshape(self.train_mnist[0], (-1, 28, 28, 1)) / 255, dtype=np.float32)
            self.mnist_x_l = np.asarray(np.reshape(self.train_mnist[1], (-1, 1)), dtype=np.float32)
            print("Loaded MNIST from: ", data_path)
        else:
            self.is_mnist = False
            self.is_textdat = True
            self.texdat = TexDAT(data_path, self.batch_size)
            self.texdat.load_images(False)

        self.dataset = tf.data.Dataset.from_generator(self.dataset_generator,output_types=tf.float32, output_shapes=([None,w,h,c]))
        self.dataset_iterator = self.dataset.make_initializable_iterator()
        self.dataset_batch = self.dataset_iterator.get_next()

        self.batch_size = batch_size
        self.margin = margin
        self.n_z = 400
        self.drop_rate = tf.placeholder(tf.float32, None, name='dropout_rate')

        self.disc_gt = tf.placeholder(tf.float32, [None, 1], name="disc_gt")
        self.disc_input = tf.placeholder(dtype=tf.float32, shape=[None, w, h, c], name="disc_input")

        self.dec_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_z], name="dec_input")

        self.disc_summaries = []
        self.vae_summaries = []

        with tf.variable_scope('encoders') as scope:
            self.mu_1, self.log_sigma_1, self.z_1 = self.get_vae_encoder_part(self.dataset_batch, True)  # encoder at the beginning
        with tf.variable_scope('decoder') as dec_scope:
            self.vae_output = self.get_vae_decoder_part(self.z_1)
            dec_scope.reuse_variables()
            self.dec_output = self.get_vae_decoder_part(self.dec_input)
            # scope.reuse_variables()
            # self.mu_2, self.log_sigma_2, self.z_2 = self.get_vae_encoder_part(self.vae_output, False)

        self.gen_variables_to_train = self.variables_to_restore = tf.global_variables()
        self.dec_variables_to_train = list(filter(lambda v: 'decoder' in v.name, self.gen_variables_to_train))

        # with tf.variable_scope('discriminator') as scope:
        #     self.discriminator_original = self.get_discriminator(self.disc_input, True)
        #     scope.reuse_variables()
        #     self.gavae = self.get_discriminator(self.vae_output, False)

        self.sess = tf.Session()

        # self.gen_loss = self.__gen_loss(self.disc_gt, self.gavae, self.dataset_batch)
        # self.disc_loss = self.__disc_loss(self.disc_gt, self.discriminator_original)

        self.vae_I_loss = self.__vae_I_loss(self.dataset_batch)
        # self.vae_II_loss = self.__vae_II_loss(self.dataset_batch)

        with tf.variable_scope('optimizers'):
            # self.disc_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.002)
            # self.gen_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            self.vae_I_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            # self.vae_II_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            self.vae_I_global_step = tf.Variable(initial_value=0, name='vae_I_global_step', trainable=False)
            self.vae_I_train_step = self.vae_I_optimizer.minimize(
                loss=self.vae_I_loss,
                global_step=self.vae_I_global_step
            )
            # self.vae_II_global_step = tf.Variable(initial_value=0, name='vae_II_global_step', trainable=False)
            # self.vae_II_train_step = self.vae_II_optimizer.minimize(
            #     loss=self.vae_II_loss,
            #     global_step=self.vae_II_global_step,
            #     var_list=self.dec_variables_to_train
            # )
            # self.disc_global_step = tf.Variable(initial_value=0, name='disc_global_step', trainable=False)
            # self.disc_train_step = self.disc_optimizer.minimize(
            #     loss=self.disc_loss,
            #     global_step=self.disc_global_step
            # )
            # self.gen_global_step = tf.Variable(initial_value=0, name='gen_global_step', trainable=False)
            # self.gen_train_step = self.gen_optimizer.minimize(
            #     loss=self.gen_loss,
            #     global_step=self.gen_global_step,
            #     var_list=self.gen_variables_to_train
            # )

    def __vae_I_loss(self, vae_input):
        # compute the average MSE error, then scale it up, ie. simply sum on all axes
        # mse_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        with tf.variable_scope('vae_loss'):
            reconstruction_loss = tf.sqrt(tf.reduce_sum(tf.square(vae_input - self.vae_output)))
            self.vae_summaries.append(tf.summary.scalar('recon_loss', reconstruction_loss))
            # compute the KL loss
            kl_loss = - 0.5 * tf.reduce_mean(1 + self.log_sigma_1 - tf.square(self.mu_1) - tf.square(tf.exp(self.log_sigma_1)), axis=-1)
            self.vae_summaries.append(tf.summary.scalar('kl_loss', tf.reduce_mean(kl_loss)))
            # return the average loss over all images in batch
            # kokot neotacaj ho, nechaj +, lebo uz je vynasobeny -1 on sam
            total_loss = tf.reduce_mean(reconstruction_loss + 2.5*kl_loss)
            self.vae_summaries.append(tf.summary.scalar('total_loss', total_loss))
        return total_loss

    def __vae_II_loss(self, vae_input):
        # compute the average MSE error, then scale it up, ie. simply sum on all axes
        # mse_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        with tf.variable_scope('vae_loss'):
            reconstruction_loss = tf.sqrt(tf.reduce_sum(tf.square(vae_input - self.vae_output)))
            self.vae_summaries.append(tf.summary.scalar('recon_loss', reconstruction_loss))
            # compute the KL loss
            kl_loss = - 0.5 * tf.reduce_mean(1 + self.log_sigma_1 - tf.square(self.mu_1) - tf.square(tf.exp(self.log_sigma_1)), axis=-1)
            self.vae_summaries.append(tf.summary.scalar('kl_loss', tf.reduce_mean(kl_loss)))
            # compute encoder loss
            encoder_loss = tf.sqrt(tf.reduce_sum(tf.square(self.mu_1 - self.mu_2) + tf.square(self.log_sigma_1 - self.log_sigma_2)))
            self.vae_summaries.append(tf.summary.scalar('encoder_loss', encoder_loss))
            # return the average loss over all images in batch
            # kokot neotacaj ho, nechaj +, lebo uz je vynasobeny -1 on sam
            total_loss = tf.reduce_mean(reconstruction_loss + 3*kl_loss + encoder_loss)
            self.vae_summaries.append(tf.summary.scalar('total_loss', total_loss))
        return total_loss

    def __gen_loss(self, y_true, y_pred, vae_input):
        # encoder loss
        with tf.variable_scope('generator_loss'):
            with tf.variable_scope('meansquare_loss'):
                reconstruction_loss = tf.sqrt(tf.reduce_sum(tf.square(vae_input - self.vae_output)))
                sampling_loss = tf.sqrt(tf.reduce_mean(tf.square(self.mu_1 - self.mu_2)))
                meansquare_loss = tf.reduce_mean(tf.square(y_pred-y_true))
                self.disc_summaries.append(tf.summary.scalar('recon_loss', reconstruction_loss))
                self.disc_summaries.append(tf.summary.scalar('meansq_loss', meansquare_loss))
                self.disc_summaries.append(tf.summary.scalar('sampling_loss', sampling_loss))
            # with tf.variable_scope('kl_divergence_loss'):
                # compute the KL loss - reduce_sum
                self.kl_loss = - 0.5 * tf.reduce_mean(1 + self.log_sigma_1 - tf.square(self.mu_1) - tf.square(tf.exp(self.log_sigma_1)), axis=-1)
                self.disc_summaries.append(tf.summary.scalar('kl_loss', tf.reduce_mean(self.kl_loss)))
            with tf.variable_scope('total_mean_loss'):
                # return the average loss over all images in batch
                total_loss = tf.reduce_mean(10*meansquare_loss + sampling_loss + self.kl_loss)
                self.disc_summaries.append(tf.summary.scalar('total_loss', total_loss))
        return total_loss

    def __disc_loss(self, y_true, y_pred):
        with tf.variable_scope('discriminant_loss'):
            meansquare_loss = tf.reduce_mean(tf.square(y_pred - y_true))
            self.disc_summaries.append(tf.summary.text('y_pred', tf.as_string(tf.reshape(y_pred,(2, self.batch_size >> 1)))))
            # cross_entropy = tf.reduce_mean(y_pred - y_pred * y_true + tf.log(1 + tf.exp(-y_pred)))
            self.disc_summaries.append(tf.summary.scalar('discriminator_loss', meansquare_loss))
        return meansquare_loss

    def sample_z(self, args) -> tf.Tensor:
        with tf.variable_scope('sampling_z'):
            mu, log_sigma = args
            eps = tf.random_normal(shape=(self.batch_size , self.n_z), mean=0., stddev=1.)
            # sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
            return mu + tf.exp(log_sigma / tf.constant(2, tf.float32)) * eps

    # vae encoder mu, log_sigma, sampled_z
    def get_vae_encoder_part(self, input, trainable) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.variable_scope('layer_1'):
            net_1 = tf.layers.conv2d(input, filters=128, kernel_size=5, strides=1,
                                     padding='same', data_format='channels_last', trainable=trainable,
                                     reuse=tf.AUTO_REUSE, name='conv_1')
            net_1 = tf.nn.leaky_relu(net_1, name='relu_1')
            # net_1 = tf.layers.batch_normalization(net_1, momentum=0.8, trainable=trainable, reuse=tf.AUTO_REUSE, name='bn_1')
            net_1 = tf.layers.dropout(net_1, self.drop_rate, name='dropout_1')

        with tf.variable_scope('layer_2'):
            net_2 = tf.layers.conv2d(net_1, filters=128, kernel_size=5, strides=2,
                                     padding='same', data_format='channels_last', trainable=trainable,
                                     reuse=tf.AUTO_REUSE, name='conv_2')
            net_2 = tf.nn.leaky_relu(net_2, name='relu_2')
            # net_2 = tf.layers.batch_normalization(net_2, momentum=0.8, trainable=trainable, reuse=tf.AUTO_REUSE, name='bn_2')
            net_2 = tf.layers.dropout(net_2, self.drop_rate, name='dropout_2')

        with tf.variable_scope('layer_3'):
            net_3 = tf.layers.conv2d(net_2, filters=256, kernel_size=5, strides=2,
                                     padding='same', data_format='channels_last', trainable=trainable,
                                     reuse=tf.AUTO_REUSE, name='conv_3')
            net_3 = tf.nn.leaky_relu(net_3, name='relu_3')
            # net_3 = tf.layers.batch_normalization(net_3, momentum=0.8, trainable=trainable, reuse=tf.AUTO_REUSE, name='bn_3')
            net_3 = tf.layers.dropout(net_3, self.drop_rate, name='dropout_3')

        with tf.variable_scope('layer_4'):
            net_4 = tf.layers.conv2d(net_3, filters=256, kernel_size=3, strides=2,
                                     padding='same', data_format='channels_last', trainable=trainable,
                                     reuse=tf.AUTO_REUSE, name='conv_4')
            net_4 = tf.nn.leaky_relu(net_4, name='relu_4')
            # net_4 = tf.layers.batch_normalization(net_4, momentum=0.8, trainable=trainable, reuse=tf.AUTO_REUSE, name='bn_4')
            net_4 = tf.layers.dropout(net_4, self.drop_rate, name='dropout_5')

        self.mid_shape = [int(net_4.shape.dims[1]),int(net_4.shape.dims[2]),int(net_4.shape.dims[3])]
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
            # net_1 = tf.layers.batch_normalization(net_1, momentum=0.8,reuse=tf.AUTO_REUSE)

        reshape = tf.reshape(net_1, (self.batch_size, self.mid_shape[0], self.mid_shape[1], 1))

        with tf.variable_scope('layer_2'):
            net_2 = tf.layers.conv2d(reshape, filters=256, kernel_size=3, strides=1,
                                     padding='same', data_format='channels_last', reuse=tf.AUTO_REUSE)
            net_2 = tf.nn.leaky_relu(net_2)
            # net_2 = tf.layers.batch_normalization(net_2, momentum=0.8, reuse=tf.AUTO_REUSE)
            net_2 = tf.layers.dropout(net_2, self.drop_rate)

        with tf.variable_scope('layer_3'):
            net_3 = tf.image.resize_images(net_2, size=(2 * self.mid_shape[0], 2 * self.mid_shape[1]))
            net_3 = tf.layers.conv2d(net_3, filters=256, kernel_size=3, strides=1,
                                     padding='same', data_format='channels_last', reuse=tf.AUTO_REUSE)
            net_3 = tf.nn.leaky_relu(net_3)
            # net_3 = tf.layers.batch_normalization(net_3, momentum=0.8, reuse=tf.AUTO_REUSE)
            net_3 = tf.layers.dropout(net_3, self.drop_rate)

        with tf.variable_scope('layer_4'):

            net_4 = tf.image.resize_images(net_3, size=(4 * self.mid_shape[0], 4 * self.mid_shape[1]))
            net_4 = tf.layers.conv2d(net_4, filters=128, kernel_size=5, strides=1,
                                     padding='same', data_format='channels_last', reuse=tf.AUTO_REUSE)
            net_4 = tf.nn.leaky_relu(net_4)
            # net_4 = tf.layers.batch_normalization(net_4, momentum=0.8)
            net_4 = tf.layers.dropout(net_4, self.drop_rate)

        with tf.variable_scope('layer_5'):
            net_5 = tf.image.resize_images(net_4, size=(8 * self.mid_shape[0], 8 * self.mid_shape[1]))
            net_5 = tf.layers.conv2d(net_5, filters=128, kernel_size=5, strides=1,
                                     padding='same', data_format='channels_last', reuse=tf.AUTO_REUSE)
            net_5 = tf.nn.leaky_relu(net_5)
            # net_5 = tf.layers.batch_normalization(net_5, momentum=0.8, reuse=tf.AUTO_REUSE)
            net_5 = tf.layers.dropout(net_5, self.drop_rate)

        def relu1(features, name=None):
            return tf.nn.relu(tf.minimum(tf.maximum(features,0),1), name)

        with tf.variable_scope('layer_6'):
            # net_6 = tf.image.resize_images(net_5, size=(8 * self.mid_shape_16[0], 8 * self.mid_shape_16[1]))
            net_6 = tf.layers.conv2d(net_5, filters=1, kernel_size=1, strides=1,
                                     padding='same', data_format='channels_last', activation=relu1, reuse=tf.AUTO_REUSE)
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
            net_1 = tf.layers.conv2d(input, filters=128, kernel_size=3, strides=1,
                                     padding='same', data_format='channels_last',
                                     reuse=tf.AUTO_REUSE, trainable=trainable, name='conv_1')
            net_1 = tf.nn.leaky_relu(net_1, name='l_relu_1')
            # net_1 = tf.layers.batch_normalization(net_1, momentum=0.8, reuse=tf.AUTO_REUSE, trainable=trainable, name="bn_1")
            net_1 = tf.layers.dropout(net_1, self.drop_rate, name='dr_1')

        with tf.variable_scope('layer_2'):
            net_2 = tf.layers.conv2d(net_1, filters=128, kernel_size=3, strides=1,
                                     padding='same', data_format='channels_last',
                                     reuse=tf.AUTO_REUSE, trainable=trainable, name='conv_2')
            net_2 = tf.nn.leaky_relu(net_2, name='l_relu_2')
            # net_2 = tf.layers.batch_normalization(net_2, momentum=0.8, reuse=tf.AUTO_REUSE, trainable=trainable, name="bn_2")
            net_2 = tf.layers.dropout(net_2, self.drop_rate, name='dr_2')

        with tf.variable_scope('layer_3'):
            net_3 = tf.layers.conv2d(net_2, filters=256, kernel_size=5, strides=1,
                                     padding='same', data_format='channels_last',
                                     reuse=tf.AUTO_REUSE, trainable=trainable, name='conv_3')
            net_3 = tf.nn.leaky_relu(net_3, name='l_relu_3')
            # net_3 = tf.layers.batch_normalization(net_3, momentum=0.8, reuse=tf.AUTO_REUSE, trainable=trainable, name="bn_3")
            net_3 = tf.layers.dropout(net_3, self.drop_rate, name='dr_3')

        with tf.variable_scope('layer_4'):
            net_4 = tf.layers.conv2d(net_3, filters=256, kernel_size=5, strides=1,
                                     padding='same', data_format='channels_last',
                                     reuse=tf.AUTO_REUSE, trainable=trainable, name='conv_4')
            net_4 = tf.nn.leaky_relu(net_4, name='l_relu_4')

        flat = tf.layers.flatten(net_4, 'flat')

        dense = tf.layers.dense(flat, 1, activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, trainable=trainable, name="dense_1")

        return dense

    def dataset_generator(self):
        if self.is_textdat:
            sorted_list = self.texdat.train.images
            # indices = np.random.choice(np.arange(len(sorted_list)), size=4, replace=False)
            # indices = [294, 309, 314, 248]
            # indices = [55, 53]
            indices = [0]
            if self.test_phase:
                indices = np.arange(len(sorted_list))
            div = int(self.batch_size / len(indices))
            if div < 1:
                div = 1
            while True:
                batch = []
                for ind in indices:
                    for i in range(div):
                        batch.append(
                            self.texdat.load_image_patch(sorted_list[ind], patch_size=self.patch_size))
                batch = resize_batch_images(batch, self.patch_size)
                self.new_batch = False
                while not self.new_batch:
                    yield batch
        elif self.is_mnist:
            start = 0
            end = self.batch_size
            epochs = 0
            while True:
                batch = self.mnist_x[start:end]
                start = end
                end = end + self.batch_size
                batch = resize_batch_images(batch, self.patch_size)
                self.new_batch = False
                while not self.new_batch:
                    yield batch
                if end > len(self.mnist_x):
                    start = 0
                    end = self.batch_size
                    epochs += 1
                    print("End of epoch: ", epochs)

    def make_some_noise(self, batch, t):
        if t < 0 or t > 1:
            return [ndimage.gaussian_filter(i, sigma=10*(1.01-np.random.rand())) for i in batch]
        return [ndimage.gaussian_filter(i, sigma=25*(1-t)) for i in batch]

    def transfer_train(self, iterations, model_file, save_interval=50, log_interval=20):
        restorer = tf.train.Saver(var_list=self.variables_to_restore)
        restore_path = 'model/' + model_file + '/'
        saver = tf.train.Saver(max_to_keep=3)
        save_path_gan = 'model/' + model_file + '_gan/'
        gen_model_ckpt = os.path.join(save_path_gan, 'checkpoint')
        vae_model_ckpt = os.path.join(restore_path, 'checkpoint')
        with self.sess:
            flag_load_VAE = True
            if os.path.exists(save_path_gan):
                try:
                    print("Trying to restore GAVAE checkpoint ...")
                    last_gan_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path_gan)
                    saver.restore(self.sess, save_path=last_gan_chk_path)
                    print("Restored GAN from:", last_gan_chk_path)
                    flag_load_VAE = False
                except:
                    print("Failed to restore GAN weights. Reinitializing from VAE.")
                    self.sess.run(tf.global_variables_initializer())
            else:
                os.makedirs(save_path_gan)
                self.sess.run(tf.global_variables_initializer())

            if os.path.exists(vae_model_ckpt) and flag_load_VAE:
                # restore checkpoint if it exists
                try:
                    print("Trying to restore VAE checkpoint ...")
                    last_vae_chk_path = tf.train.latest_checkpoint(checkpoint_dir=restore_path)
                    restorer.restore(self.sess, save_path=last_vae_chk_path)
                    print("Restored VAE weights from:", last_vae_chk_path)
                except:
                    print("Failed to restore VAE weights. Error")
                    return 0xfff

            self.sess.run(self.dataset_iterator.initializer)
            disc_sum_merge = tf.summary.merge(self.disc_summaries)

            writer = tf.summary.FileWriter('logs_gen/' + model_file+'/', graph=self.sess.graph)

            disc_ones = np.ones((self.batch_size, 1), dtype=np.float32)
            disc_zeros = np.zeros((self.batch_size, 1), dtype=np.float32)

            early_stop_counter = 0
            unwanted = [0.0, 0.5, 1.0]
            for iteration in range(iterations):
                print('Iteration {:d}.'.format(iteration))

                self.new_batch = True
                # validation batch
                originals = np.asarray(self.sess.run(self.dataset_batch))
                fakes_vae = np.asarray(self.sess.run(self.vae_output))

                disc_predict_fake_v = self.sess.run([self.discriminator_original], feed_dict={
                    self.disc_input: fakes_vae,
                    self.drop_rate: 0
                })

                disc_predict_orig = self.sess.run([self.discriminator_original], feed_dict={
                    self.disc_input: originals,
                    self.drop_rate: 0
                })

                if iteration % log_interval == 0:
                    print("Disc. validation originals: ")
                    print(np.reshape(disc_predict_orig, (2, self.batch_size >> 1)))
                    print("Disc. validation vae: ")
                    print(np.reshape(disc_predict_fake_v, (2, self.batch_size >> 1)))
                disc_val_ori = np.mean(np.square(disc_predict_orig - disc_ones))
                disc_val_fakv = np.mean(np.square(disc_predict_fake_v))

                print("Disc. validation originals error: ", disc_val_ori)
                print("Disc. validation fakes vae error: ", disc_val_fakv)

                # train batch
                self.new_batch = True
                originals = np.asarray(self.sess.run(self.dataset_batch))
                fakes_vae = np.asarray(self.sess.run(self.vae_output))

                _, disc_loss, disc_sum = self.sess.run([self.disc_train_step, self.disc_loss, disc_sum_merge], feed_dict={
                    self.disc_gt: disc_ones,
                    self.disc_input: originals,
                    self.drop_rate: 0.5
                })
                print("Disc. orig_1. loss: ", disc_loss)

                _, disc_loss_v, disc_sum = self.sess.run([self.disc_train_step, self.disc_loss, disc_sum_merge], feed_dict={
                    self.disc_gt: disc_zeros,
                    self.disc_input: fakes_vae,
                    self.drop_rate: 0.5
                })
                print("Disc. fake vae loss: ", disc_loss_v)

                # self.new_batch = True

                # _, disc_loss_v, disc_sum = self.sess.run([self.disc_train_step, self.disc_loss, disc_sum_merge], feed_dict={
                #     self.disc_gt: disc_zeros,
                #     self.disc_input: fakes_vae,
                #     self.drop_rate: 0.5
                # })
                # print("Disc. fake vae loss: ", disc_loss_v)

                # if disc_loss < 0.2 and disc_loss_v < 0.2:
                #     print("Disc. loss < 0.05 --> trying validation again")
                #     disc_predict_orig = self.sess.run([self.discriminator_original], feed_dict={
                #         self.disc_input: np.concatenate((originals, fakes_vae)),
                #         self.drop_rate: 0
                #     })
                #     disc_val_ori = np.mean(np.square(disc_predict_orig - np.concatenate((disc_ones, disc_zeros))))
                #     print("Disc. validation classification error: ", disc_val_ori)

                _, vae_loss = self.sess.run([self.vae_I_train_step, self.vae_I_loss])
                print("VAE loss: ", vae_loss)

                # self.new_batch = True
                _, gen_loss = self.sess.run([self.gen_train_step, self.gen_loss], feed_dict={
                    self.disc_gt: disc_ones,
                    self.drop_rate: 0.5
                })
                print("Gen loss: ", gen_loss)

                # regularization for early stopping
                if iteration > 200 and (disc_loss in unwanted or vae_loss > 10000):
                    if early_stop_counter > 5:
                        print("Stopping the training due to unable to learn details.")
                        print("Unable to change gradient for the 5th time")
                        break
                    early_stop_counter += 1
                    try:
                        print("VAE loss too high. \nTrying to restore last checkpoint ...")
                        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path_gan)
                        idx = last_chk_path.rfind('-')
                        restore_point = last_chk_path[:idx+1] + str(int(last_chk_path[idx+1:])-save_interval)
                        saver.restore(self.sess, save_path=restore_point)
                        print("Restored checkpoint from:", restore_point)
                    except:
                        if early_stop_counter < 5:
                            print("Did not restored, continue training.")
                            continue
                        else:
                            print("Stopping the training due to unable to find suitable checkpoint.")
                            return 1488

                if iteration % save_interval == 0 and iteration > 0:
                    save_path_local = saver.save(sess=self.sess, save_path=gen_model_ckpt, global_step=iteration)
                    print('Model saved to file as {:s}'.format(save_path_local))
                    save_path_vae = restorer.save(sess=self.sess, save_path=vae_model_ckpt, global_step=iteration)
                    print('VAE saved to file as {:s}'.format(save_path_vae))

                if iteration % log_interval == 0:
                    writer.add_summary(disc_sum, global_step=iteration)
                    if iteration == 0:
                        if not os.path.exists('./images/0'):
                            os.makedirs('./images/0')
                        if not os.path.exists('./images/1'):
                            os.makedirs('./images/1')
                        if not os.path.exists('./images/2'):
                            os.makedirs('./images/2')
                        if not os.path.exists('./images/3'):
                            os.makedirs('./images/3')

                    batch = self.sess.run(self.dataset_batch)
                    # w = self.patch_size[0]
                    # h = self.patch_size[1]
                    # dimensions = (w, h)

                    ims = batch[0][:,:,0]
                    plt.imsave('./images/0/'+str(iteration)+'_0_baseline.png', ims, cmap='gray')
                    ims = batch[int(self.batch_size*1/4)][:,:,0]
                    plt.imsave('./images/1/'+str(iteration)+'_1_baseline.png', ims, cmap='gray')
                    ims = batch[int(self.batch_size*2/4)][:,:,0]
                    plt.imsave('./images/2/'+str(iteration)+'_2_baseline.png', ims, cmap='gray')
                    ims = batch[int(self.batch_size*3/4)][:,:,0]
                    plt.imsave('./images/3/'+str(iteration)+'_3_baseline.png', ims, cmap='gray')

                    ims = self.vae_output.eval()

                    imss = ims[0][:,:,0]
                    plt.imsave('./images/0/' + str(iteration) + '.png', imss, cmap='gray')
                    imss = ims[int(self.batch_size*1/4)][:,:,0]
                    plt.imsave('./images/1/' + str(iteration) + '.png', imss, cmap='gray')
                    imss = ims[int(self.batch_size*2/4)][:,:,0]
                    plt.imsave('./images/2/' + str(iteration) + '.png', imss, cmap='gray')
                    imss = ims[int(self.batch_size*3/4)][:,:,0]
                    plt.imsave('./images/3/' + str(iteration) + '.png', imss, cmap='gray')
        return 0

    def test_discriminant(self, model_file):
        saver = tf.train.Saver()
        save_path_gan = 'model/' + model_file + '_gan/'

        with self.sess:
            if os.path.exists(save_path_gan):
                try:
                    print("Trying to restore GAVAE checkpoint ...")
                    last_gan_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path_gan)
                    saver.restore(self.sess, save_path=last_gan_chk_path)
                    print("Restored GAN from:", last_gan_chk_path)
                except:
                    print("Failed to restore GAN weights.")
                    return -1
            else:
                print("GAN save directory {:s} does not exist.".format(save_path_gan))
                return -1

            self.sess.run(self.dataset_iterator.initializer)
            disc_ones = np.ones((self.batch_size, 1), dtype=np.float32)
            disc_zeros = np.zeros((self.batch_size >> 1, 1), dtype=np.float32)
            disc_labels = np.concatenate((disc_ones[0:(self.batch_size >> 1)], disc_zeros))

            class_err = 0
            avg = np.zeros((self.batch_size, 1))
            for epoch in range(100):
                self.new_batch = True
                # validation batch
                originals = np.asarray(self.sess.run(self.dataset_batch)[0:self.batch_size:2])
                fakes = np.asarray(self.make_some_noise(self.sess.run(self.vae_output)[0:self.batch_size:2], 0.94))
                batch_validation = np.concatenate((originals, fakes))

                disc_predict = self.sess.run(self.discriminator_original, feed_dict={
                    self.disc_input: batch_validation,
                    self.drop_rate: 0
                })
                class_err += np.mean(np.square(disc_predict - disc_labels))
                class_err /= 2
                avg += disc_predict
            print('Class error: ', class_err)
            avg /= 100
            print('Average prediction: ')
            print(avg.reshape((2,16)))
        return 0

    def train(self, iterations, model_file, save_interval=50, log_interval=20):
        # sorted_list = self.texdat.train.images
        save_path = 'model/' + model_file + '/'

        saver = tf.train.Saver(max_to_keep=3)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_ckpt = os.path.join(save_path, 'checkpoint')

        with self.sess:
            if os.path.exists(model_ckpt):
                # restore checkpoint if it exists
                try:
                    print("Trying to restore last checkpoint ...")
                    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path)
                    saver.restore(self.sess, save_path=last_chk_path)
                    print("Restored checkpoint from:", last_chk_path)
                except:
                    print("Failed to restore checkpoint. Initializing variables instead.")
                    self.sess.run(tf.global_variables_initializer())
            else:
                self.sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter('logs/' + model_file + '/', graph=self.sess.graph)

            # summary_disc_1 = tf.summary.merge(self.disc_summaries)
            summary_vae = tf.summary.merge(self.vae_summaries)

            self.sess.run(self.dataset_iterator.initializer)

            early_stop_counter = 0
            for iteration in range(iterations):
                self.new_batch = True
                # training the VAE
                # _, vae_loss = self.sess.run([self.vae_I_train_step, self.vae_I_loss])
                # print('I {:d}. VAE_I {:.6f}'.format(iteration, vae_loss))

                _, vae_loss, merged = self.sess.run([self.vae_I_train_step, self.vae_I_loss, summary_vae])
                print('I {:d}. VAE_I {:.6f}'.format(iteration, vae_loss))

                # regularization for early stopping
                if vae_loss > 500 or vae_loss < -500:
                    # if early_stop_counter > 5:
                    #     print("Stopping the training due to high loss.")
                    #     print("Unable to change gradient for the 5th time")
                    #     break
                    early_stop_counter += 1
                    try:
                        print(["VAE loss to high", "Trying to restore last checkpoint ..."], sep='\n')
                        print("Stopped for the {:d}. time".format(early_stop_counter))
                        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path)
                        idx = last_chk_path.rfind('-')
                        restore_point = last_chk_path[:idx+1] + str(int(last_chk_path[idx+1:])-save_interval)
                        saver.restore(self.sess, save_path=last_chk_path)
                        print("Restored checkpoint from:", last_chk_path)
                    except:
                        print("Stopping the training due to high loss.")
                        break

# log + test for results
                if iteration % save_interval == 0 and iteration > 0:
                    save_path_local = saver.save(sess=self.sess, save_path=model_ckpt, global_step=iteration)
                    print('Model saved to file as {:s}'.format(save_path_local))

                if iteration % log_interval == 0:
                    writer.add_summary(merged, global_step = iteration)
                if iteration % log_interval == 0:
                    if iteration == 0:
                        if not os.path.exists('./images/0'):
                            os.makedirs('./images/0')
                        if not os.path.exists('./images/1'):
                            os.makedirs('./images/1')
                        if not os.path.exists('./images/2'):
                            os.makedirs('./images/2')
                        if not os.path.exists('./images/3'):
                            os.makedirs('./images/3')

                    batch = self.sess.run(self.dataset_batch)
                    # w = self.patch_size[0]
                    # h = self.patch_size[1]
                    # dimensions = (w,h)
                    ims = batch[0][:, :, 0]
                    plt.imsave('./images/0/' + str(iteration) + '_0_baseline.png', ims, cmap='gray')
                    ims = batch[int(self.batch_size * 1 / 4)][:, :, 0]
                    plt.imsave('./images/1/' + str(iteration) + '_1_baseline.png', ims, cmap='gray')
                    ims = batch[int(self.batch_size * 2 / 4)][:, :, 0]
                    plt.imsave('./images/2/' + str(iteration) + '_2_baseline.png', ims, cmap='gray')
                    ims = batch[int(self.batch_size * 3 / 4)][:, :, 0]
                    plt.imsave('./images/3/' + str(iteration) + '_3_baseline.png', ims, cmap='gray')

                    # TODO: logging
                    # latent = self.z_1.eval()
                    # latent += 1.3
                    # generated = self.dec_output.eval(feed_dict={
                    #     self.dec_input: latent
                    # })

                    ims = self.vae_output.eval()

                    imss = ims[0][:, :, 0]
                    plt.imsave('./images/0/' + str(iteration) + '.png', imss, cmap='gray')
                    imss = ims[int(self.batch_size * 1 / 4)][:, :, 0]
                    plt.imsave('./images/1/' + str(iteration) + '.png', imss, cmap='gray')
                    imss = ims[int(self.batch_size * 2 / 4)][:, :, 0]
                    plt.imsave('./images/2/' + str(iteration) + '.png', imss, cmap='gray')
                    imss = ims[int(self.batch_size * 3 / 4)][:, :, 0]
                    plt.imsave('./images/3/' + str(iteration) + '.png', imss, cmap='gray')

                    # imss = np.reshape(generated[0], (160, 160))
                    # plt.imsave('./images/0/' + str(epoch) + '_z.png', imss, cmap='gray')
                    # imss = np.reshape(generated[8], (160, 160))
                    # plt.imsave('./images/1/' + str(epoch) + '_z.png', imss, cmap='gray')
                    # imss = np.reshape(generated[16], (160, 160))
                    # plt.imsave('./images/2/' + str(epoch) + '_z.png', imss, cmap='gray')
                    # imss = np.reshape(generated[24], (160, 160))
                    # plt.imsave('./images/3/' + str(epoch) + '_z.png', imss, cmap='gray')


    def test(self, model_file):
        sorted_list = self.texdat.train.images

        restore_path = 'model/' + model_file + '/'
        saver = tf.train.Saver(max_to_keep=5)
        # if not os.path.exists(restore_path):
        #     print("Unable to find restore path, ending test")
        #     return
        model_ckpt = os.path.join(restore_path, 'checkpoint')

        with self.sess:
            if os.path.exists(model_ckpt):
                # restore checkpoint if it exists
                try:
                    print("Trying to restore last checkpoint ...")
                    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=restore_path)
                    saver.restore(self.sess, save_path=last_chk_path)
                    print("Restored checkpoint from:", last_chk_path)
                except:
                    print("Failed to restore model checkpoint. Ending test.")
                    return
            else:
                print("Failed to restore model. Ending test")
                return
            if not os.path.exists('./test-images/'):
                os.makedirs('./test-images/')

            self.sess.run(self.dataset_iterator.initializer)

            for j in range(10):
                batch_1 = self.sess.run(self.dataset_batch)
                latent_1 = self.z_1.eval()

                self.new_batch = True
                batch_2 = self.sess.run(self.dataset_batch)
                latent_2 = self.z_1.eval()

                B = (latent_2 - latent_1) / 6
                for t in range(5):
                    latent = latent_1+B*t

                    ims = self.dec_output.eval(feed_dict={
                        self.dec_input: latent,
                        self.drop_rate: 0
                    })
                    c = 0
                    for i in range(j * 10, j * 10 + 10):
                        batchs = np.reshape(batch_1[c], (160, 160))
                        plt.imsave('./test-images/' + str(i) + '_o_1.png', batchs, cmap='gray')
                        batchs = np.reshape(batch_2[c], (160, 160))
                        plt.imsave('./test-images/' + str(i) + '_o_2.png', batchs, cmap='gray')
                        imss = np.reshape(ims[c],(160,160))
                        plt.imsave('./test-images/' + str(i) + '_t_' + str(t) + '.png', imss, cmap='gray')
                        c = (c+1) % 10
