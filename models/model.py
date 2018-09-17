
class ModelGAVAE(object):
    """
        Basic model for later model implementations
        GAN models = Generative Adversarial Networks
        VAE models = Variatonal Autoencoders

        TODO:
         - Support for all models
        """

    def __init__(self, input_width, input_height, input_channels, input_glob_channels, batch_size=32):
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
        self.glob_channels = input_glob_channels

        self.img_shape = (self.img_rows, self.img_cols, self.channels)