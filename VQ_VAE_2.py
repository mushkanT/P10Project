from __future__ import absolute_import, division, print_function
import tensorflow as tf
import sonnet as snt


class Encoder(tf.keras.Model):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, stride,
                 name=None):
        """
        Encoder convolution model for initial latent representation in VQ_VAE model

        :param num_hiddens: Number of filters in convolutional layers
        :param num_residual_layers: Number of residual layers in residual stack
        :param num_residual_hiddens: Number of filters in residual layers
        :param stride: Stride 8, 4 or 2, depending on encoding level (top,middel,bottom)
        :param name: Name of encoder model
        """
        super(Encoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        if stride == 8:
             self.layers = [
                 tf.keras.layers.Conv2D(
                     filters=num_hiddens // 2,
                     kernel_size=4,
                     strides=8,
                     padding='same',
                     name='encoder_conv_1'
                 ),
                 tf.keras.layers.Conv2D(
                    filters=num_hiddens,
                    kernel_size=4,
                    strides=2,
                    padding='same',
                    name='encoder_conv_2'
                ),
                tf.keras.layers.Conv2D(
                    filters=num_hiddens,
                    kernel_size=4,
                    strides=2,
                    padding='same',
                    name='encoder_conv_3'
                ),
                tf.keras.layers.Conv2D(
                    filters=num_hiddens,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    name="encoder_conv_4"
                )
             ]
        elif stride == 4:
            self.layers = [
                tf.keras.layers.Conv2D(
                    filters=num_hiddens//2,
                    kernel_size=4,
                    strides=2,
                    padding='same'
                ),
                tf.keras.layers.Conv2D(
                    filters=num_hiddens,
                    kernel_size=4,
                    strides=2,
                    padding='same'
                ),
                tf.keras.layers.Conv2D(
                    filters=num_hiddens,
                    kernel_size=3,
                    strides=1,
                    padding='same'
                )
            ]
        elif stride == 2:
            self.layers=[
                tf.keras.layers.Conv2D(
                    filters=num_hiddens // 2,
                    kernel_size=4,
                    strides=2,
                    padding='same'
                ),
                tf.keras.layers.Conv2D(
                    filters=num_hiddens,
                    kernel_size=3,
                    strides=1,
                    padding='same'
                )
            ]
        self.residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens
        )


    def __call__(self, inputs):
        x = inputs
        for i in self.layers:
            x = tf.nn.relu(self.layers[i](x))
        return self._residual_stack(x)



class ResidualStack(tf.keras.Model):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name=None):
        super(ResidualStack, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self.layers = []

        for i in range(num_residual_layers):
            conv3 = tf.keras.layers.Conv2D(
                filters=num_residual_hiddens,
                kernel_size=3,
                strides=1,
                padding='same',
                name="residual_3x3_%d" % i
            )
            conv1 = tf.keras.layers.Conv2D(
                filters=num_hiddens,
                kernel_size=3,
                strides=1,
                padding='same',
                name='residual_1x1_%d' % i
            )
            self.layers.append((conv3, conv1))


    def __call__(self, inputs):
        x = inputs
        for conv3, conv1 in self.layers:
            conv3_out = conv3(tf.nn.relu(x))
            conv1_out = conv1(tf.nn.relu(conv3_out))
            x += conv1_out
        return tf.nn.relu(x)


class Decoder(tf.keras.Model):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,stride,
                 name=None):
        super(Decoder,self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self.residual_stack = ResidualStack(
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens
        )
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_hiddens,
            kernel_size=3,
            strides=1,
            padding='same',
            name='decoder_conv_1'
        )

        self.layers = [
            tf.keras.layers.Conv2DTranspose(
                filters=num_hiddens,
                kernel_size=4,
                strides=2,
                padding='same'
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=num_hiddens//2,
                kernel_size=4,
                strides=2,
                padding='same'
            )
        ]

        self.conv_t_final = tf.keras.layers.Conv2DTranspose(
                filters=3,
                kernel_size=4,
                strides=2,
                padding='same'
        )

    def __call__(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.residual_stack(x)
        for i in self.layers:
            x = tf.nn.relu(self.layers[i](x))
        x = self.conv_t_final(x)
        return x


class VQVAEModel(tf.keras.Model):
    def __init__(self, encoder, decoder, vqvae, pre_vq_conv1,
                 data_variance, name=None):
        super(VQVAEModel, self).__init__(name=name)
        self._encoder = encoder
        self._decoder = decoder
        self._vqvae = vqvae
        self._pre_vq_conv1 = pre_vq_conv1
        self._data_variance = data_variance

    def __call__(self, inputs, is_training):
        z = self._pre_vq_conv1(self._encoder(inputs))
        vq_output = self._vqvae(z, is_training=is_training)
        x_recon = self._decoder(vq_output['quantize'])
        recon_error = tf.reduce_mean((x_recon - inputs) ** 2) / self._data_variance
        loss = recon_error + vq_output['loss']
        return {
            'z': z,
            'x_recon': x_recon,
            'loss': loss,
            'recon_error': recon_error,
            'vq_output': vq_output,
        }