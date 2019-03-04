from .model import ModelGAVAE
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from scipy import ndimage
import os.path

from matplotlib import pyplot as plt
from skimage.transform import resize

import numpy as np

from data_loader import TexDAT
from data_loader import resize_batch_images, normalize_batch_images

class vae_next_gen:
    def __init__(self, data_path, w, h, c, batch_size=32, lr=0.0001):
        self.patch_size = (w, h, c)
        self.batch_size = batch_size
        self.new_batch = False
        self.test_phase = False
        self.learning_rate = lr

        self.n_z = 10
        self.z_1_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_z], name="z_1_input")
        self.z_2_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_z], name="z_2_input")
        self.z_3_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_z], name="z_3_input")
        self.z_4_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_z], name="z_4_input")

        self.drop_rate = tf.placeholder(dtype=np.float32,shape=[1], name='drop_rate')

        if 'mnist' in str(data_path).lower():
            self.is_mnist = True
            self.is_textdat = False
            data = input_data.read_data_sets(data_path, one_hot=False)
            # self.train_mnist, self.test_mnist = tf.keras.datasets.mnist.load_data(data_path)
            self.train_mnist = [data.train.images, data.train.labels]
            self.test_mnist = [data.test.images, data.test.labels]
            self.mnist_x = np.asarray(np.reshape(self.train_mnist[0], (-1, 28, 28)), dtype=np.float32)
            self.mnist_x = np.asarray([resize(b, (32, 32),mode='reflect') for b in self.mnist_x]).reshape((-1, 32, 32, 1))
            self.mnist_x = normalize_batch_images(self.mnist_x, normalize='zeromean')
            self.mnist_x_l = np.asarray(np.reshape(self.train_mnist[1], (-1, 1)), dtype=np.float32)
            print("Loaded MNIST from: ", data_path)
        else:
            self.is_mnist = False
            self.is_textdat = True
            self.texdat = TexDAT(data_path, self.batch_size)
            self.texdat.load_images(False)

        self.dataset = tf.data.Dataset.from_generator(self.dataset_generator, output_types=tf.float32, output_shapes=([None, w, h, c]), args=[2048])
        self.dataset_iterator = self.dataset.make_initializable_iterator()
        self.dataset_batch = self.dataset_iterator.get_next()

        self.vae_summaries = []
        self.sess = tf.Session()

        with tf.variable_scope('encoder'):
            self.smp1, self.smp2, self.smp3, self.smp4 = self.generate_encoder(self.dataset_batch)
        with tf.variable_scope('decoder') as scope:
            self.vae_out = self.generate_decoder(self.smp1[2],self.smp2[2],self.smp3[2],self.smp4[2], save_variables=True)
            scope.reuse_variables()
            self.dec_out = self.generate_decoder(self.z_1_input,self.z_2_input,self.z_3_input,self.z_4_input, save_variables=False)

        self.vae_loss = self.__vae_loss(self.dataset_batch)
        with tf.variable_scope('optimizers'):
            self.vae_optimizer_1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.vae_global_step_1 = tf.Variable(initial_value=0, name='vae_global_step_1', trainable=False)
            self.vae_train_step_1 = self.vae_optimizer_1.minimize(
                loss=self.vae_loss,
                global_step=self.vae_global_step_1,
                var_list=self.enc_l1+self.dec_l1
            )
            self.vae_optimizer_2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.vae_global_step_2 = tf.Variable(initial_value=0, name='vae_global_step_2', trainable=False)
            self.vae_train_step_2 = self.vae_optimizer_2.minimize(
                loss=self.vae_loss,
                global_step=self.vae_global_step_2,
                var_list=self.enc_l2+self.dec_l2
            )
            self.vae_optimizer_3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.vae_global_step_3 = tf.Variable(initial_value=0, name='vae_global_step_3', trainable=False)
            self.vae_train_step_3 = self.vae_optimizer_3.minimize(
                loss=self.vae_loss,
                global_step=self.vae_global_step_3,
                var_list=self.enc_l3+self.dec_l3
            )
            self.vae_optimizer_4 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.vae_global_step_4 = tf.Variable(initial_value=0, name='vae_global_step_4', trainable=False)
            self.vae_train_step_4 = self.vae_optimizer_4.minimize(
                loss=self.vae_loss,
                global_step=self.vae_global_step_4,
                var_list=self.enc_l4+self.dec_l4
            )
            self.vae_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.vae_global_step = tf.Variable(initial_value=0, name='vae_global_step_4', trainable=False)
            self.vae_train_step = self.vae_optimizer.minimize(
                loss=self.vae_loss,
                global_step=self.vae_global_step
            )


    def __vae_loss(self, vae_input):
        # compute the average MSE error, then scale it up, ie. simply sum on all axes
        # mse_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        with tf.variable_scope('vae_loss'):
            reconstruction_loss = tf.sqrt(tf.reduce_sum(tf.square(vae_input - self.vae_out)))
            self.vae_summaries.append(tf.summary.scalar('recon_loss', reconstruction_loss))
            # compute the KL loss
            kl_loss_1 = - 0.5 * tf.reduce_mean(1 + self.smp1[1] - tf.square(self.smp1[0]) - tf.square(tf.exp(self.smp1[1])), axis=-1)
            self.vae_summaries.append(tf.summary.scalar('kl_loss_1', tf.reduce_mean(kl_loss_1)))
            kl_loss_2 = - 0.5 * tf.reduce_mean(1 + self.smp2[1] - tf.square(self.smp2[0]) - tf.square(tf.exp(self.smp2[1])), axis=-1)
            self.vae_summaries.append(tf.summary.scalar('kl_loss_2', tf.reduce_mean(kl_loss_2)))
            kl_loss_3 = - 0.5 * tf.reduce_mean(1 + self.smp3[1] - tf.square(self.smp3[0]) - tf.square(tf.exp(self.smp3[1])), axis=-1)
            self.vae_summaries.append(tf.summary.scalar('kl_loss_3', tf.reduce_mean(kl_loss_3)))
            kl_loss_4 = - 0.5 * tf.reduce_mean(1 + self.smp4[1] - tf.square(self.smp4[0]) - tf.square(tf.exp(self.smp4[1])), axis=-1)
            self.vae_summaries.append(tf.summary.scalar('kl_loss_4', tf.reduce_mean(kl_loss_4)))
            # return the average loss over all images in batch
            # kokot neotacaj ho, nechaj +, lebo uz je vynasobeny -1 on sam
            total_loss = tf.reduce_mean(reconstruction_loss + 4.*kl_loss_1+3.*kl_loss_2+2.*kl_loss_3+1.2*kl_loss_4)
            self.vae_summaries.append(tf.summary.scalar('total_loss', total_loss))
        return total_loss

    def dataset_generator(self, samples):
        if self.is_textdat:
            sorted_list = self.texdat.train.images
            # indices = np.random.choice(np.arange(len(sorted_list)), size=4, replace=False)
            # indices = [294, 309, 314, 248]
            # indices = [55, 53]
            indices = np.arange(len(sorted_list))

            if samples > 0:
                dataset = []
                for ind in indices:
                    for i in range(samples):
                        dataset.append(self.texdat.load_image_patch(sorted_list[ind], patch_size=self.patch_size).reshape(self.patch_size))
                dataset = normalize_batch_images(dataset, normalize='zeromean')
                start = 0
                end = self.batch_size
                epochs = 0
                while True:
                    batch = dataset[start:end]
                    batch = resize_batch_images(batch, self.patch_size)
                    start = end
                    end = end + self.batch_size
                    self.new_batch = False
                    while not self.new_batch:
                        yield batch
                    if end > len(dataset):
                        start = 0
                        end = self.batch_size
                        epochs+=1
                        print("End of epoch: ", epochs)
                        print("Shuffling dataset")
                        np.random.shuffle(dataset)

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
                    print("Shuffling dataset")
                    np.random.shuffle(self.mnist_x)

    def adain(self, baseline, details, epsilon=1e-5, data_format='channels_first'):
        axes = [2, 3] if data_format == 'channels_first' else [1, 2]

        mean_b, var_b = tf.nn.moments(baseline, axes=axes, keep_dims=True)
        mean_d, var_d = tf.nn.moments(details, axes=axes, keep_dims=True)
        b_std, d_std = tf.sqrt(var_b + epsilon), tf.sqrt(var_d + epsilon)

        return d_std * (baseline - mean_b) / b_std + mean_d

    def make_some_noise(self, batch, t):
        if t < 0 or t > 1:
            return [ndimage.gaussian_filter(i, sigma=10*(1.01-np.random.rand())) for i in batch]
        return [ndimage.gaussian_filter(i, sigma=25*(1-t)) for i in batch]


    def sample_z(self, args) -> tf.Tensor:
        with tf.variable_scope('sampling_z'):
            mu, log_sigma = args
            eps = tf.random_normal(shape=(self.batch_size , self.n_z), mean=0., stddev=1.)
            # sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
            return mu + tf.exp(log_sigma / tf.constant(2, tf.float32)) * eps


    def generate_encoder(self, input):
        with tf.variable_scope('layer_1'):
            net_1 = tf.layers.conv2d(input, filters=32, kernel_size=5, strides=1,
                                     padding='same', data_format='channels_last',
                                     reuse=tf.AUTO_REUSE, name='conv_1')
            net_1 = tf.nn.leaky_relu(net_1, name='relu_1')
            net_1 = tf.layers.dropout(net_1, self.drop_rate, name='dropout_1')
            to_dense_1 = tf.layers.conv2d(net_1, filters=2, kernel_size=1, strides=1,
                                     padding='same', reuse=tf.AUTO_REUSE, name='conv1x1_1')
            net_1 = tf.layers.batch_normalization(net_1, name='batchnorm_1')
            net_1 = tf.layers.max_pooling2d(net_1, 2, 2, padding='same', name='maxpool_1')

        self.enc_l1 = tf.global_variables()

        with tf.variable_scope('layer_2'):
            net_2 = tf.layers.conv2d(net_1, filters=32, kernel_size=5, strides=1,
                                     padding='same', data_format='channels_last',
                                     reuse=tf.AUTO_REUSE, name='conv_2')
            net_2 = tf.nn.leaky_relu(net_2, name='relu_2')
            net_2 = tf.layers.dropout(net_2, self.drop_rate, name='dropout_2')
            to_dense_2 = tf.layers.conv2d(net_2, filters=2, kernel_size=1, strides=1,
                                     padding='same', reuse=tf.AUTO_REUSE, name='conv1x1_2')
            net_2 = tf.layers.batch_normalization(net_2, name='batchnorm_2')
            net_2 = tf.layers.max_pooling2d(net_2, 2, 2, padding='same', name='maxpool_2')

        self.enc_l2 = tf.global_variables()

        with tf.variable_scope('layer_3'):
            net_3 = tf.layers.conv2d(net_2, filters=64, kernel_size=3, strides=1,
                                     padding='same', data_format='channels_last',
                                     reuse=tf.AUTO_REUSE, name='conv_3')
            net_3 = tf.nn.leaky_relu(net_3, name='relu_3')
            net_3 = tf.layers.dropout(net_3, self.drop_rate, name='dropout_3')
            to_dense_3 = tf.layers.conv2d(net_3, filters=4, kernel_size=1, strides=1,
                                          padding='same', reuse=tf.AUTO_REUSE, name='conv1x1_3')
            net_3 = tf.layers.batch_normalization(net_3, name='batchnorm_3')
            net_3 = tf.layers.max_pooling2d(net_3, 2, 2, padding='same', name='maxpool_3')

        self.enc_l3 = tf.global_variables()

        with tf.variable_scope('layer_4'):
            net_4 = tf.layers.conv2d(net_3, filters=64, kernel_size=3, strides=1,
                                     padding='same', data_format='channels_last',
                                     reuse=tf.AUTO_REUSE, name='conv_2')
            net_4 = tf.nn.leaky_relu(net_4, name='relu_2')
            net_4 = tf.layers.dropout(net_4, self.drop_rate, name='dropout_2')
            to_dense_4 = tf.layers.conv2d(net_4, filters=4, kernel_size=1, strides=1,
                                          padding='same', reuse=tf.AUTO_REUSE, name='conv1x1_4')

        self.enc_l4 = tf.global_variables()

        self.enc_l4 = [i for i in self.enc_l4 if i not in self.enc_l3]
        self.enc_l3 = [i for i in self.enc_l3 if i not in self.enc_l2]
        self.enc_l2 = [i for i in self.enc_l2 if i not in self.enc_l1]

        self.mid_shape = [net_4.shape[0].value, net_4.shape[1].value, net_4.shape[2].value, net_4.shape[3].value]

        temp = tf.global_variables()
        with tf.variable_scope('sampling_layers'):
            f1 = tf.layers.flatten(to_dense_1)
            f2 = tf.layers.flatten(to_dense_2)
            f3 = tf.layers.flatten(to_dense_3)
            f4 = tf.layers.flatten(to_dense_4)

            mu1 = tf.layers.dense(f1, self.n_z, activation=None, reuse=tf.AUTO_REUSE, name='mu_1')
            sigma1 = tf.layers.dense(f1, self.n_z, activation=None, reuse=tf.AUTO_REUSE, name='log_sigma_1')
            self.enc_l1 += [i for i in tf.global_variables() if i not in temp]
            z1 = self.sample_z([mu1, sigma1])
            temp = tf.global_variables()

            mu2 = tf.layers.dense(f2, self.n_z, activation=None, reuse=tf.AUTO_REUSE, name='mu_2')
            sigma2 = tf.layers.dense(f2, self.n_z, activation=None, reuse=tf.AUTO_REUSE, name='log_sigma_2')
            self.enc_l2 += [i for i in tf.global_variables() if i not in temp]
            z2 = self.sample_z([mu2, sigma2])
            temp = tf.global_variables()

            mu3 = tf.layers.dense(f3, self.n_z, activation=None, reuse=tf.AUTO_REUSE, name='mu_3')
            sigma3 = tf.layers.dense(f3, self.n_z, activation=None, reuse=tf.AUTO_REUSE, name='log_sigma_3')
            self.enc_l3 += [i for i in tf.global_variables() if i not in temp]
            z3 = self.sample_z([mu3, sigma3])
            temp = tf.global_variables()

            mu4 = tf.layers.dense(f4, self.n_z, activation=None, reuse=tf.AUTO_REUSE, name='mu_4')
            sigma4 = tf.layers.dense(f4, self.n_z, activation=None, reuse=tf.AUTO_REUSE, name='log_sigma_4')
            self.enc_l4 += [i for i in tf.global_variables() if i not in temp]
            z4 = self.sample_z([mu4, sigma4])

        self.enc_all = tf.global_variables()

        return (mu1,sigma1,z1), (mu2,sigma2,z2), (mu3,sigma3,z3), (mu4,sigma4,z4)

    def generate_decoder(self, z1, z2, z3, z4, save_variables:bool=False):
        def relu1(features, name=None):
            return tf.nn.relu(tf.minimum(tf.maximum(features, 0), 1), name)

        with tf.variable_scope('lowest_layer'):
            out_4 = tf.layers.dense(z4, self.mid_shape[1]*self.mid_shape[2],reuse=tf.AUTO_REUSE, name='squaring_dense_4')
            out_4 = tf.reshape(out_4, (-1, self.mid_shape[1], self.mid_shape[2], 1))
            net_4 = tf.layers.conv2d(out_4, 32, 3, 1, padding='same', reuse=tf.AUTO_REUSE, name='out_conv_4')
            net_4 = tf.nn.leaky_relu(net_4, name='relu_4')
            net_4 = tf.layers.batch_normalization(net_4, name='batchnorm_4')
            net_4 = tf.layers.dropout(net_4, 'dropout_4')

        if save_variables:
            self.dec_l4 = tf.global_variables()

        with tf.variable_scope('second_lowest_layer'):
            out_3 = tf.layers.dense(z3, 2*self.mid_shape[1] * 2*self.mid_shape[2], reuse=tf.AUTO_REUSE, name='squaring_dense_3')
            out_3 = tf.reshape(out_3, (-1, 2*self.mid_shape[1], 2*self.mid_shape[2], 1))
            out_3 = tf.layers.conv2d(out_3, 32, 3, 1, padding='same',reuse=tf.AUTO_REUSE, name='out_conv_3')
            out_3 = tf.nn.leaky_relu(out_3, name='out_relu_3')

            net_3 = tf.image.resize_images(net_4, size=(2 * self.mid_shape[1], 2 * self.mid_shape[2]))
            net_3 = tf.add(net_3, out_3, name='merge_out_net_3')
            net_3 = tf.layers.conv2d(net_3, 64, 3, 1, padding='same', reuse=tf.AUTO_REUSE, name='conv_3')
            net_3 = tf.nn.leaky_relu(net_3, name='relu_3')
            net_3 = tf.layers.batch_normalization(net_3, name='batchnorm_3')
            net_3 = tf.layers.dropout(net_3, 'dropout_3')

        if save_variables:
            self.dec_l3 = tf.global_variables()

        with tf.variable_scope('second_highest_layer'):
            out_2 = tf.layers.dense(z2, 4*self.mid_shape[1] * 4*self.mid_shape[2], reuse=tf.AUTO_REUSE, name='squaring_dense_2')
            out_2 = tf.reshape(out_2, (-1, 4*self.mid_shape[1], 4*self.mid_shape[2], 1))
            out_2 = tf.layers.conv2d(out_2, 64, 5, 1, padding='same',reuse=tf.AUTO_REUSE, name='out_conv_2')
            out_2 = tf.nn.leaky_relu(out_2, name='out_relu_2')

            net_2 = tf.image.resize_images(net_3, size=(4 * self.mid_shape[1], 4 * self.mid_shape[2]))
            net_2 = tf.add(net_2, out_2, name='merge_out_net_2')
            net_2 = tf.layers.conv2d(net_2, 64, 5, 1, padding='same', reuse=tf.AUTO_REUSE, name='conv_2')
            net_2 = tf.nn.leaky_relu(net_2, name='relu_2')
            net_2 = tf.layers.batch_normalization(net_2, name='batchnorm_2')
            net_2 = tf.layers.dropout(net_2, 'dropout_2')

        if save_variables:
            self.dec_l2 = tf.global_variables()

        with tf.variable_scope('top_layer'):
            out_1 = tf.layers.dense(z1, 8*self.mid_shape[1] * 8*self.mid_shape[2], reuse=tf.AUTO_REUSE, name='squaring_dense_1')
            out_1 = tf.reshape(out_1, (-1, 8*self.mid_shape[1], 8*self.mid_shape[2], 1))
            out_1 = tf.layers.conv2d(out_1, 64, 5, 1, padding='same',reuse=tf.AUTO_REUSE, name='out_conv_1')
            out_1 = tf.nn.leaky_relu(out_1, name='out_relu_1')

            net_1 = tf.image.resize_images(net_2, size=(8 * self.mid_shape[1], 8 * self.mid_shape[2]))
            net_1 = tf.add(net_1, out_1, name='merge_out_net_1')
            net_1 = tf.layers.conv2d(net_1, 64, 5, 1, padding='same', reuse=tf.AUTO_REUSE, name='conv_1')
            net_1 = tf.nn.leaky_relu(net_1, alpha=0.5, name='relu_1')
            net_1 = tf.layers.batch_normalization(net_1, name='batchnorm_1')
            # net_1 = tf.layers.dropout(net_1, 'dropout_1')
            net_1 = tf.layers.conv2d(net_1, 64, 3, 1, padding='same', reuse=tf.AUTO_REUSE, name='conv_1_top')
            net_1 = tf.nn.leaky_relu(net_1, alpha=0.5, name='relu_1top')
            out = tf.layers.conv2d(net_1, 1, 1, 1, padding='same', reuse=tf.AUTO_REUSE, name='top')

        if save_variables:
            self.dec_l1 = tf.global_variables()

        if save_variables:
            self.dec_l4 = [i for i in self.dec_l4 if i not in self.enc_all]
            self.dec_l1 = [i for i in self.dec_l1 if i not in self.enc_all]
            self.dec_l1 = [i for i in self.dec_l1 if i not in self.dec_l2]
            self.dec_l2 = [i for i in self.dec_l2 if i not in self.enc_all]
            self.dec_l2 = [i for i in self.dec_l2 if i not in self.dec_l3]
            self.dec_l3 = [i for i in self.dec_l3 if i not in self.enc_all]
            self.dec_l3 = [i for i in self.dec_l3 if i not in self.dec_l4]

        return out


    def train(self, iterations, save_interval, log_interval, model_file):
        save_path = 'D:/VAE_model/' + model_file + '/'

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

            all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
            print("Model has {0} number of trainable parameters".format(self.sess.run(all_trainable_vars)))
            writer_1 = tf.summary.FileWriter('logs/' + model_file + '/stack_1/', graph=self.sess.graph, filename_suffix='stack_1')
            writer_2 = tf.summary.FileWriter('logs/' + model_file + '/stack_2/', graph=self.sess.graph, filename_suffix='stack_2')
            writer_3 = tf.summary.FileWriter('logs/' + model_file + '/stack_3/', graph=self.sess.graph, filename_suffix='stack_3')
            writer_4 = tf.summary.FileWriter('logs/' + model_file + '/stack_4/', graph=self.sess.graph, filename_suffix='stack_4')
            writer = tf.summary.FileWriter('logs/' + model_file + '/full_stack/', graph=self.sess.graph, filename_suffix='full_stack')


            # summary_disc_1 = tf.summary.merge(self.disc_summaries)
            summary_vae = tf.summary.merge(self.vae_summaries)

            self.sess.run(self.dataset_iterator.initializer)

            early_stop_counter = 0
            for iteration in range(0,iterations+1):
                self.new_batch = True

                _, vae_loss, merged = self.sess.run([self.vae_train_step, self.vae_loss, summary_vae])
                print('I {:d}. VAE full-stack {:.6f}'.format(iteration, vae_loss))
                _, vae_loss_4, merged_4 = self.sess.run([self.vae_train_step_4, self.vae_loss, summary_vae])
                print('I {:d}. VAE stack 4 {:.6f}'.format(iteration, vae_loss_4))
                _, vae_loss_3, merged_3 = self.sess.run([self.vae_train_step_3, self.vae_loss, summary_vae])
                print('I {:d}. VAE stack 3 {:.6f}'.format(iteration, vae_loss_3))
                _, vae_loss_2, merged_2 = self.sess.run([self.vae_train_step_2, self.vae_loss, summary_vae])
                print('I {:d}. VAE stack 2 {:.6f}'.format(iteration, vae_loss_2))
                _, vae_loss_1, merged_1 = self.sess.run([self.vae_train_step_1, self.vae_loss, summary_vae])
                print('I {:d}. VAE stack 1 {:.6f}'.format(iteration, vae_loss_1))
                # regularization for early stopping
                if (vae_loss > 10000 or vae_loss < -10000) and \
                        (vae_loss_1 > 10000 or vae_loss_1 < -10000) and \
                        (vae_loss_2 > 10000 or vae_loss_2 < -10000) and \
                        (vae_loss_3 > 10000 or vae_loss_3 < -10000):
                    early_stop_counter += 1
                    try:
                        print(["VAE loss to high", "Trying to restore last checkpoint ..."], sep='\n')
                        print("Stopped for the {:d}. time".format(early_stop_counter))
                        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path)
                        idx = last_chk_path.rfind('-')
                        restore_point = last_chk_path[:idx + 1] + str(int(last_chk_path[idx + 1:]) - save_interval)
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
                    writer.add_summary(merged, global_step=iteration)
                    writer_1.add_summary(merged_1, global_step=iteration)
                    writer_2.add_summary(merged_2, global_step=iteration)
                    writer_3.add_summary(merged_3, global_step=iteration)
                    writer_4.add_summary(merged_4, global_step=iteration)
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
                        if not os.path.exists('./images/test_z'):
                            os.makedirs('./images/test_z')

                    batch = self.sess.run(self.dataset_batch)
                    ims = batch[0][:, :, 0]
                    plt.imsave('./images/0/' + str(iteration) + '_0_baseline.png', ims, cmap='gray')
                    ims = batch[int(self.batch_size * 1 / 4)][:, :, 0]
                    plt.imsave('./images/1/' + str(iteration) + '_1_baseline.png', ims, cmap='gray')
                    ims = batch[int(self.batch_size * 2 / 4)][:, :, 0]
                    plt.imsave('./images/2/' + str(iteration) + '_2_baseline.png', ims, cmap='gray')
                    ims = batch[int(self.batch_size * 3 / 4)][:, :, 0]
                    plt.imsave('./images/3/' + str(iteration) + '_3_baseline.png', ims, cmap='gray')

                    ims = self.vae_out.eval()

                    imss = ims[0][:, :, 0]
                    plt.imsave('./images/0/' + str(iteration) + '.png', imss, cmap='gray')
                    imss = ims[int(self.batch_size * 1 / 4)][:, :, 0]
                    plt.imsave('./images/1/' + str(iteration) + '.png', imss, cmap='gray')
                    imss = ims[int(self.batch_size * 2 / 4)][:, :, 0]
                    plt.imsave('./images/2/' + str(iteration) + '.png', imss, cmap='gray')
                    imss = ims[int(self.batch_size * 3 / 4)][:, :, 0]
                    plt.imsave('./images/3/' + str(iteration) + '.png', imss, cmap='gray')

                    a = self.sess.run((self.smp1,self.smp2,self.smp3,self.smp4))
                    zz1 = a[0][2]
                    zz2 = a[1][2]
                    zz3 = a[2][2]
                    zz4 = a[3][2]

                    z1 = np.random.normal(loc=0,scale=1,size=[4*self.n_z])
                    z4=np.asarray(z1[0:self.n_z]).reshape(1,self.n_z)
                    z2=np.asarray(z1[self.n_z:2*self.n_z]).reshape(1,self.n_z)
                    z3=np.asarray(z1[2*self.n_z:3*self.n_z]).reshape(1,self.n_z)
                    z1=np.asarray(z1[3*self.n_z:4*self.n_z]).reshape(1,self.n_z)
                    result = self.sess.run(self.dec_out, feed_dict={self.z_1_input:z1,self.z_2_input:z2,self.z_3_input:z3,self.z_4_input:z4,})

                    plt.imsave('./images/test_z/' + str(iteration) + '_random.png',result.reshape(64,64), cmap='gray')
                    i = np.random.randint(0,self.batch_size)
                    result = self.sess.run(self.dec_out, feed_dict={self.z_1_input: zz1[i].reshape(1, self.n_z),
                                                                    self.z_2_input: zz2[i].reshape(1, self.n_z),
                                                                    self.z_3_input: zz3[i].reshape(1, self.n_z),
                                                                    self.z_4_input: zz4[i].reshape(1, self.n_z), })
                    plt.imsave('./images/test_z/' + str(iteration) + '_reconstructed.png', result.reshape(64,64), cmap='gray')
                    result = self.sess.run(self.dec_out, feed_dict={self.z_1_input: z1.reshape(1, self.n_z),
                                                                    self.z_2_input: zz2[i].reshape(1, self.n_z),
                                                                    self.z_3_input: zz3[i].reshape(1, self.n_z),
                                                                    self.z_4_input: zz4[i].reshape(1, self.n_z), })
                    plt.imsave('./images/test_z/' + str(iteration) + '_reconstructed_z1.png', result.reshape(64, 64), cmap='gray')
                    result = self.sess.run(self.dec_out, feed_dict={self.z_1_input: zz1[i].reshape(1, self.n_z),
                                                                    self.z_2_input: z2.reshape(1, self.n_z),
                                                                    self.z_3_input: zz3[i].reshape(1, self.n_z),
                                                                    self.z_4_input: zz4[i].reshape(1, self.n_z), })
                    plt.imsave('./images/test_z/' + str(iteration) + '_reconstructed_z2.png', result.reshape(64, 64), cmap='gray')
                    result = self.sess.run(self.dec_out, feed_dict={self.z_1_input: zz1[i].reshape(1, self.n_z),
                                                                    self.z_2_input: zz2[i].reshape(1, self.n_z),
                                                                    self.z_3_input: z3.reshape(1, self.n_z),
                                                                    self.z_4_input: zz4[i].reshape(1, self.n_z), })
                    plt.imsave('./images/test_z/' + str(iteration) + '_reconstructed_z3.png', result.reshape(64, 64), cmap='gray')
                    result = self.sess.run(self.dec_out, feed_dict={self.z_1_input: zz1[i].reshape(1, self.n_z),
                                                                    self.z_2_input: zz2[i].reshape(1, self.n_z),
                                                                    self.z_3_input: zz3[i].reshape(1, self.n_z),
                                                                    self.z_4_input: z4.reshape(1, self.n_z), })
                    plt.imsave('./images/test_z/' + str(iteration) + '_reconstructed_z4.png', result.reshape(64, 64), cmap='gray')

# a = self.sess.run((self.smp1,self.smp2,self.smp3,self.smp4))
# zz1 = a[0][2]
# zz2 = a[1][2]
# zz3 = a[2][2]
# zz4 = a[3][2]

# z1 = np.random.normal(loc=0,scale=1,size=[80])
# z4=np.asarray(z1[0:20]).reshape(1,20)
# z2=np.asarray(z1[20:40]).reshape(1,20)
# z3=np.asarray(z1[40:60]).reshape(1,20)
# z1=np.asarray(z1[60:80]).reshape(1,20)

# result = self.sess.run(self.dec_out, feed_dict={self.z_1_input:z1,self.z_2_input:z2,self.z_3_input:z3,self.z_4_input:z4,})
# plt.imshow(result.reshape(64,64),cmap='gray')
# plt.show()

# i = 0
#
# result = self.sess.run(self.dec_out, feed_dict={self.z_1_input:zz1[i].reshape(1,20),self.z_2_input:zz2[i].reshape(1,20),self.z_3_input:zz3[i].reshape(1,20),self.z_4_input:zz4[i].reshape(1,20),})
# plt.imshow(result.reshape(64,64),cmap='gray')
# plt.show()