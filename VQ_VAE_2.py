from __future__ import absolute_import, division, print_function
import tensorflow as tf
import sonnet as snt


class Encoder(tf.keras.Model):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name=None):
        """
        Encoder convolution model for initial latent representation in VQ_VAE model

        :param num_hiddens: Number of filters in convolutional layers
        :param num_residual_layers: Number of residual layers in residual stack
        :param num_residual_hiddens: Number of filters in residual layers
        :param name: Name of encoder model
        """
        super(Encoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self.residual_stack = ResidualStack(num_hiddens,num_residual_layers,num_residual_hiddens)


class Encoder8Stride(Encoder):
    """ Encoder for creating quarter of a quarter embedding of image data along both x and y axis """
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, name=None):
        super().__init__(num_hiddens, num_residual_layers, num_residual_hiddens)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_hiddens // 2,
            kernel_size=4,
            strides=2,
            padding='same',
            name='encoder_conv_1'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=num_hiddens,
            kernel_size=4,
            strides=2,
            padding='same',
            name='encoder_conv_2'
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=num_hiddens,
            kernel_size=4,
            strides=2,
            padding='same',
            name='encoder_conv_3'
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=num_hiddens,
            kernel_size=3,
            strides=1,
            padding='same',
            name="encoder_conv_4"
        )

    def __call__(self, input):
        x = input
        x = tf.nn.relu(self.conv1(x))
        x = tf.nn.relu(self.conv2(x))
        x = tf.nn.relu(self.conv3(x))
        x = tf.nn.relu(self.conv4(x))
        return self.residual_stack(x)


class Encoder4Stride(Encoder):
    """ Encoder for creating quarter embedding of image data along both x and y axis """
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, name=None):
        super().__init__(num_hiddens, num_residual_layers, num_residual_hiddens)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_hiddens//2,
            kernel_size=4,
            strides=2,
            padding='same',
            name = 'encoder_conv_1'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=num_hiddens,
            kernel_size=4,
            strides=2,
            padding='same',
            name='encoder_conv_2'
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=num_hiddens,
            kernel_size=3,
            strides=1,
            padding='same',
            name='encoder_conv_3'
        )

    def __call__(self, input):
        x = input
        x = tf.nn.relu(self.conv1(x))
        x = tf.nn.relu(self.conv2(x))
        x = tf.nn.relu(self.conv3(x))
        return self.residual_stack(x)


class Encoder2Stride(Encoder):
    """ Encoder for creating half size embedding of image data along both x and y axis """
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, name=None):
        super().__init__(num_hiddens, num_residual_layers, num_residual_hiddens)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_hiddens // 2,
            kernel_size=4,
            strides=2,
            padding='same',
            name='encoder_conv_1'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=num_hiddens,
            kernel_size=3,
            strides=1,
            padding='same',
            name='encoder_conv_2'
        )

    def __call__(self, input):
        x = input
        x = tf.nn.relu(self.conv1(x))
        x = tf.nn.relu(self.conv2(x))
        return self.residual_stack(x)


class ResidualStack(tf.keras.Model):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name=None):
        super(ResidualStack, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens


        self.conv3 = tf.keras.layers.Conv2D(
            filters=num_residual_hiddens,
            kernel_size=3,
            strides=1,
            padding='same',
            name="residual_3x3"
        )
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_hiddens,
            kernel_size=3,
            strides=1,
            padding='same',
            name='residual_1x1'
        )

    def __call__(self, input):
        x = input
        conv3_out = self.conv3(tf.nn.relu(x))
        conv1_out = self.conv1(tf.nn.relu(conv3_out))
        x += conv1_out
        if self._num_residual_layers == 1:
            conv3_out2 = self.conv3(tf.nn.relu(x))
            conv1_out2 = self.conv1(tf.nn.relu(conv3_out2))
            x += conv1_out2
        elif self._num_residual_layers > 2:
            print('Only 2 residual layers are supported')
        return tf.nn.relu(x)


class Decoder8(Encoder):
    """ Decoder for upscaling 8 times input size """
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name=None):
        super().__init__(num_hiddens, num_residual_layers, num_residual_hiddens, name)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_hiddens,
            kernel_size=3,
            strides=1,
            padding='same',
            name='decoder_conv_1'
        )
        self.conv_2 = tf.keras.layers.Conv2DTranspose(
                filters=num_hiddens//2,
                kernel_size=4,
                strides=2,
                padding='same'
        )
        self.conv_3 = tf.keras.layers.Conv2DTranspose(
            filters=num_hiddens // 2,
            kernel_size=4,
            strides=2,
            padding='same'
        )
        self.conv_t_final = tf.keras.layers.Conv2DTranspose(
                filters=3,
                kernel_size=4,
                strides=2,
                padding='same'
        )

    def __call__(self, input):
        x = input
        x = self.conv1(x)
        x = self.residual_stack(x)
        x = tf.nn.relu(self.conv_2(x))
        x = tf.nn.relu(self.conv_3(x))
        x = self.conv_t_final(x)
        return x


class Decoder4(Encoder):
    """ Decoder for upscaling 4 times input size """
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name=None):
        super().__init__(num_hiddens, num_residual_layers, num_residual_hiddens, name)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_hiddens,
            kernel_size=3,
            strides=1,
            padding='same',
            name='decoder_conv_1'
        )
        self.conv_2 = tf.keras.layers.Conv2DTranspose(
                filters=num_hiddens//2,
                kernel_size=4,
                strides=2,
                padding='same'
        )
        self.conv_t_final = tf.keras.layers.Conv2DTranspose(
                filters=3,
                kernel_size=4,
                strides=2,
                padding='same'
        )

    def __call__(self, input):
        x = input
        x = self.conv1(x)
        x = self.residual_stack(x)
        x = tf.nn.relu(self.conv_2(x))
        x = self.conv_t_final(x)
        return x


class Decoder2(Encoder):
    """ Decoder for upscaling 2 times input size """
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, output_channel,
                 name=None):
        super().__init__(num_hiddens, num_residual_layers, num_residual_hiddens, name)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_hiddens,
            kernel_size=3,
            strides=1,
            padding='same',
            name='decoder_conv_1'
        )
        self.conv_t_final = tf.keras.layers.Conv2DTranspose(
                filters=output_channel,
                kernel_size=4,
                strides=2,
                padding='same'
        )

    def __call__(self, input):
        x = input
        x = self.conv1(x)
        x = self.residual_stack(x)
        x = self.conv_t_final(x)
        return x


class VQVAEModel(tf.keras.Model):
    def __init__(self, image_size, hierarchy_level=None, num_hiddens=128, num_residual_hiddens=32, num_residual_layers=2,
                 embedding_dim=64, num_embeddings=512, commitment_cost=0.25, decay=0.99):
        """
        VQ_VAE_2 Object for training and evaluating a VQ_VAE_2 Model.
        :param image_size: Size of input images - even number between 28 and 1024
        :param hierarchy_level: Determines the number of hierarchy levels in the model - if =None it is determined by image_size
        :param num_hiddens: Control number of filter in convolutional layers
        :param num_residual_hiddens: Control number of filter in residual convo layers
        :param num_residual_layers: Number of residual layers
        :param embedding_dim: Embedding dimension
        :param num_embeddings: Number of embeddings in shared codebook
        :param commitment_cost: Controls the level on influence from the encoder output
        :param decay: Controls the speed of the EMA for vector quantizer
        """
        super().__init__()
        self.image_size = image_size
        if image_size <= 512 or hierarchy_level == 2:
            self.encoder_bottom = Encoder4Stride(num_hiddens,num_residual_layers,num_residual_hiddens,
                                                 name='encoder_bottom_4')
            self.encoder_top = Encoder2Stride(num_hiddens, num_residual_layers, num_residual_hiddens,
                                              name='encoder_top_2')
            self.pre_vq_conv_top = tf.keras.layers.Conv2D(
                filters=embedding_dim,
                kernel_size=1,
                strides=1,
                padding='same',
                name='top_to_vq_conv'
            )
            self.quantize_top = snt.nets.VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
            self.decoder_top_embed = Decoder2(num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)
            self.pre_vq_conv_bottom = tf.keras.layers.Conv2D(
                filters=embedding_dim,
                kernel_size=1,
                strides=1,
                padding='same',
                name='bottom_to_vq_conv'
            )
            self.quantize_bottom = snt.nets.VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
            self.upsample_t = tf.keras.layers.Conv2DTranspose(
                filters= embedding_dim,
                kernel_size=4,
                strides=2,
                padding='same',
                name='upsample_top_conv'
            )
            self.decoder = Decoder4(
                num_hiddens,
                num_residual_layers,
                num_residual_hiddens,
                3
            )

        elif image_size > 512 or hierarchy_level == 3:
            self.encoder_bottom = Encoder8Stride(num_hiddens, num_residual_layers, num_residual_hiddens,
                                                 name='encoder_bottom_8')
            self.encoder_middle = Encoder2Stride(num_hiddens, num_residual_layers, num_residual_hiddens,
                                                 name='encoder_middle_2')
            self.encoder_top = Encoder2Stride(num_hiddens, num_residual_layers, num_residual_hiddens,
                                              name='encoder_top_2')

            self.pre_vq_conv_top = tf.keras.layers.Conv2D(
                filters=embedding_dim,
                kernel_size=1,
                strides=1,
                padding='same',
                name='top_to_vq_conv'
            )
            self.quantize_top = snt.nets.VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
            self.decoder_top_embed = Decoder2(num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)

            self.pre_vq_conv_middle = tf.keras.layers.Conv2D(
                filters=embedding_dim,
                kernel_size=1,
                strides=1,
                padding='same',
                name='middle_to_vq_conv'
            )

            self.quantize_middle = snt.nets.VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
            self.decoder_middle_embed = Decoder2(num_hiddens, num_residual_layers, num_residual_hiddens,embedding_dim)
            self.upsample_m = tf.keras.layers.Conv2DTranspose(
                filters=embedding_dim,
                kernel_size=4,
                strides=2,
                padding='same',
                name='upsample_mid_conv'
             )

            self.pre_vq_conv_bottom = tf.keras.layers.Conv2D(
                filters=embedding_dim,
                kernel_size=1,
                strides=1,
                padding='same',
                name='bottom_to_vq_conv'
            )
            self.quantize_bottom = snt.nets.VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
            self.upsample_t = tf.keras.layers.Conv2DTranspose(
                filters=embedding_dim,
                kernel_size=4,
                strides=2,
                padding='same',
                name='upsample_top_conv'
            )
            self.decoder = Decoder8(
                num_hiddens,
                num_residual_layers,
                num_residual_hiddens,
                3
            )


        else:
            print('Unsupported hierarchy level: Current support for 2 or 3 levels')

    def encode2layer(self, input):
        encoded_bottom = self.encoder_bottom(input)
        encoded_top = self.encoder_top(encoded_bottom)

        prep_encoded_top = self.pre_vq_conv_top(encoded_top)
        quantized_top = self.quantize_top(prep_encoded_top, is_training=True)
        decoded_top = self.decoder_top_embed(quantized_top['quantize'])
        encoded_bottom = tf.concat([decoded_top, encoded_bottom], 3)

        prep_encoded_bottom = self.pre_vq_conv_bottom(encoded_bottom)
        quantized_bottom = self.quantize_bottom(prep_encoded_bottom,is_training=True)


        return quantized_top, quantized_bottom

    def encode3layer(self, input):
        # Encoding chain from bottom to top
        encoded_bottom = self.encoder_bottom(input)
        encoded_middle = self.encoder_middle(encoded_bottom)
        encoded_top = self.encoder_top(encoded_middle)

        # quantize top layer for coarse features
        prep_encoded_top = self.pre_vq_conv_top(encoded_top)
        quantized_top = self.quantize_top(prep_encoded_top, is_training=True)

        # Decode top layer quantized for encoding of middle layer (dependancy)
        decoded_top = self.decoder_top_embed(quantized_top['quantize'])
        encoded_middle = tf.concat([decoded_top, encoded_middle], 3)

        # quantize middle layer for medium features
        prep_encoded_middle = self.pre_vq_conv_middle(encoded_middle)
        quantized_middle = self.quantize_middle(prep_encoded_middle, is_training=True)

        # decode middle layer quantized for encoding of bottom layer (dependancy)
        decoded_middle = self.decoder_middle_embed(quantized_middle['quantize'])
        encoded_bottom = tf.concat([decoded_middle, encoded_bottom], 3)

        # quantize bottom layer for finest features
        prep_encoded_bottom = self.pre_vq_conv_bottom(encoded_bottom)
        quantized_bottom = self.quantize_bottom(prep_encoded_bottom, is_training=True)

        return quantized_top, quantized_middle, quantized_bottom


    def decode2layer(self, quantized_top, quantized_bottom):
        upsample_top = self.upsample_t(quantized_top['quantize'])
        quantized_embed = tf.concat([upsample_top, quantized_bottom['quantize']], 3)
        decoding = self.decoder(quantized_embed)

        return decoding

    def decode3layer(self, quantized_top, quantized_middle, quantized_bottom):
        upsample_top = self.upsample_t(quantized_top['quantize'])
        quantized_mid = tf.concat([upsample_top, quantized_middle['quantize']], 3)
        upsample_mid = self.upsample_m(quantized_mid)
        quantized_embed = tf.concat([upsample_mid, quantized_bottom['quantize']], 3)
        decoding = self.decoder(quantized_embed)
        return decoding

    def decode_code(self, code_top, code_bottom):
        quantized_top = self.quantize_top.embeddings(code_top)
        quantized_bottom = self.quantize_bottom.embeddings(code_bottom)

        dec = self.decoder(quantized_top, quantized_bottom)

        return dec

    def __call__(self, input, is_training):
        if self.image_size > 512:
            quantized_top, quantized_middle, quantized_bottom = self.encode3layer(input)
            dec = self.decode3layer(quantized_top, quantized_middle, quantized_bottom)
            recon_error = tf.reduce_mean((dec - input) ** 2)
            mean_latent_loss = (quantized_bottom['loss'] + quantized_middle['loss'] + quantized_top['loss']) / 2
            loss = recon_error + mean_latent_loss
            return {
                'x_recon': dec,
                'loss': loss,
                'recon_error': recon_error,
                'mean_latent_loss': mean_latent_loss
            }
        else:
            quantized_top, quantized_bottom = self.encode2layer(input)
            dec = self.decode2layer(quantized_top, quantized_bottom)
            recon_error = tf.reduce_mean((dec - input) ** 2)
            mean_latent_loss = (quantized_bottom['loss'] + quantized_top['loss']) / 2
            loss = recon_error + mean_latent_loss
            return {
                'x_recon': dec,
                'loss': loss,
                'recon_error': recon_error,
                'mean_latent_loss': mean_latent_loss
            }