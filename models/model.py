
import numpy as np

class ModelGAVAE(object):
    """
        Basic model for later model implementations
        GAN models = Generative Adversarial Networks
        VAE models = Variatonal Autoencoders

        TODO:
         - Support for all models
        """

    def __init__(self, input_width, input_height, input_channels, layer_depth, batch_size=32):
        """
        Init

        :param input_width: image width
        :param input_height: image height
        :param input_channels: image channels
        :param batch_size: batch size
        """

        self.img_rows = input_height
        self.img_cols = input_width
        self.channels = input_channels
        self.batch_size = batch_size

        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.batch_img_shape = (batch_size, self.img_rows, self.img_cols, self.channels)
        self.img_shape_disc = (self.img_rows, self.img_cols, 2)
        self.mid_shape = (int(self.img_rows / (2 ** layer_depth)), int(self.img_cols / (2 ** layer_depth)), self.channels)
        self.mid_shape_16 = (int(self.img_rows / (2 ** layer_depth)), int(self.img_cols / (2 ** layer_depth)), 16)

        # subbands for spectral sharpening

        self.band_1 = np.ones(shape=self.img_shape)
        self.band_2 = np.ones(shape=self.img_shape)
        self.band_3 = np.ones(shape=self.img_shape)
        self.band_4 = np.ones(shape=self.img_shape)
        self.band_5 = np.zeros(shape=self.img_shape)

        half_w, half_h = int(self.img_rows / 2), int(self.img_cols / 2)
        width = int(self.img_rows / 10)

        self.band_1[half_w - width:half_w + width + 1, half_h - width:half_h + width + 1,0] = 0

        self.band_2[half_w - 2 * width:half_w + 2 * width + 1, half_h - 2 * width:half_h + 2 * width + 1,0] = 0
        self.band_2[half_w - width:half_w + width + 1, half_h - width:half_h + width + 1,0] = 1

        self.band_3[half_w - 3 * width:half_w + 3 * width + 1, half_h - 3 * width:half_h + 3 * width + 1,0] = 0
        self.band_3[half_w - 2 * width:half_w + 2 * width + 1, half_h - 2 * width:half_h + 2 * width + 1,0] = 1

        self.band_4[half_w - 4 * width:half_w + 4 * width + 1, half_h - 4 * width:half_h + 4 * width + 1,0] = 0
        self.band_4[half_w - 3 * width:half_w + 3 * width + 1, half_h - 3 * width:half_h + 3 * width + 1,0] = 1

        self.band_5[half_w - 4 * width:half_w + 4 * width + 1, half_h - 4 * width:half_h + 4 * width + 1,0] = 1