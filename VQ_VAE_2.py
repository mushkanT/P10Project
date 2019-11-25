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
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, name=None):
        super().__init__(num_hiddens, num_residual_layers, num_residual_hiddens)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_hiddens // 2,
            kernel_size=4,
            strides=8,
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

    def __call__(self, inputs):
        x = inputs
        x = tf.nn.relu(self.conv1(x))
        x = tf.nn.relu(self.conv2(x))
        x = tf.nn.relu(self.conv3(x))
        x = tf.nn.relu(self.conv4(x))
        return self.residual_stack(x)


class Encoder4Stride(Encoder):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, name=None):
        super().__init__(num_hiddens, num_residual_layers, num_residual_hiddens)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_hiddens//2,
            kernel_size=4,
            strides=2,
            padding='same'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=num_hiddens,
            kernel_size=4,
            strides=2,
            padding='same'
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=num_hiddens,
            kernel_size=3,
            strides=1,
            padding='same'
        )

    def __call__(self, inputs):
        x = inputs
        x = tf.nn.relu(self.conv1(x))
        x = tf.nn.relu(self.conv2(x))
        x = tf.nn.relu(self.conv3(x))
        return self.residual_stack(x)


class Encoder2Stride(Encoder):
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

    def __call__(self, inputs):
        x = inputs
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
        """
        self._layers = []
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
            self._layers.append((conv3, conv1))
        """

    def __call__(self, inputs):
        x = inputs
        conv3_out = self.conv3(tf.nn.relu(x))
        conv1_out = self.conv1(tf.nn.relu(conv3_out))
        x += conv1_out
        if(self._num_residual_layers > 1):
            print('You only get 2 residual layers boy')
            conv3_out2 = self.conv3(tf.nn.relu(x))
            conv1_out2 = self.conv1(tf.nn.relu(conv3_out2))
            x += conv1_out2
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

    def __call__(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.residual_stack(x)
        x = tf.nn.relu(self.conv_2(x))
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