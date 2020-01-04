from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.keras import layers
# import sonnet as snt


class VQVAEModel:
    def __init__(self, image_size, num_channels, num_hiddens=128, num_residual_hiddens=32, num_residual_layers=2,
                 embedding_dim=64, num_embeddings=512, commitment_cost=0.25, decay=0.99):
        """
        VQ_VAE class which contains encoder/decoder models for training a 2-tier VQ_VAE Model.
        :param image_size: Size of input images - even number divisble by 4 between 32 and 1024
        :param num_hiddens: Control number of filters in convolutional layers
        :param num_residual_hiddens: Control number of filters in residual convo layers
        :param num_residual_layers: Number of residual layers
        :param embedding_dim: Embedding dimension
        :param num_embeddings: Number of embeddings in shared codebook
        :param commitment_cost: Controls the level on influence from the encoder output
        :param decay: Controls the speed of the EMA for vector quantizer
        """

        self.num_hiddens = num_hiddens
        self.num_residual_hiddens = num_residual_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.num_channels = num_channels
        self.decay = decay
        self.image_size = image_size

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def residual_stack_builder(self, in_shape, num_residual_layers, num_residual_hiddens, name):
        res_input = layers.Input(shape=in_shape)
        encoded = layers.Activation('relu')(res_input)

        res3 = layers.Conv2D(filters=num_residual_hiddens,
                             kernel_size=3,
                             strides=1,
                             padding='same',
                             name="residual_3x3",
                             activation='relu'
                             )

        res1 = layers.Conv2D(filters=self.num_hiddens,
                             kernel_size=3,
                             strides=1,
                             padding='same',
                             name='residual_1x1')

        res3_out = res3(encoded)
        res1_out = res1(res3_out)

        res_encoding = layers.add([res_input, res1_out])

        if num_residual_layers == 2:
            res_encoding = layers.Activation('relu')(res_encoding)
            res3_out2 = res3(res_encoding)
            res1_out2 = res1(res3_out2)
            res_encoding = layers.add([res_encoding, res1_out2])
        else:
            raise Exception('Only support 1 or 2 residual layers')
        return tf.keras.Model(inputs=res_input, outputs=res_encoding, name=name)

    def build_model(self):
        input_img = layers.Input(shape=(self.image_size, self.image_size, self.num_channels))
        encode = layers.Conv2D(filters=self.num_hiddens // 2, kernel_size=4, strides=2, padding='same',
                               name='encoder_conv_1', activation='relu')(input_img)
        encode = layers.Conv2D(filters=self.num_hiddens, kernel_size=4, strides=2, padding='same', name='encoder_conv_2', activation='relu')(
            encode)
        bottom_encodeOut = layers.Conv2D(filters=self.num_hiddens,
                                         kernel_size=3,
                                         strides=1,
                                         padding='same',
                                         name='encoder_conv_3')(encode)

        bot_res_stack = self.residual_stack_builder(bottom_encodeOut[0].shape, self.num_residual_layers, self.num_residual_hiddens, name='bot_res_stack')

        bottom_encode_out = bot_res_stack(bottom_encodeOut)

        """
        ///////////////  TOP ENCODER ////////////////
        """

        top_encoding = layers.Conv2D(
            filters=self.num_hiddens // 2,
            kernel_size=4,
            strides=2,
            padding='same',
            activation='relu'
        )(bottom_encode_out)

        top_encoding = layers.Conv2D(
            filters=self.num_hiddens,
            kernel_size=3,
            strides=1,
            padding='same'
        )(top_encoding)

        top_res_stack = self.residual_stack_builder(top_encoding[0].shape, self.num_residual_layers, self.num_residual_hiddens, name='top_res_stack')

        top_encoding = top_res_stack(top_encoding)

        """
        /////////////// QUANTIZE 2 LAYERS  ////////////////
        """

        embed_top = layers.Conv2D(
            filters=self.embedding_dim,
            kernel_size=1,
            strides=1,
            padding='same',
            name='top_to_vq_conv'
        )(top_encoding)

        #quanter = snt.nets.VectorQuantizerEMA(self.embedding_dim, self.num_embeddings, self.commitment_cost, self.decay)
        #quant_layer = layers.Lambda(lambda x: quanter(x, is_training=True))
        self.vq_top = VQ(self.embedding_dim, self.num_embeddings, self.commitment_cost, name='vq_top')
        quant_top = self.vq_top(embed_top)
        decode_top = layers.Conv2D(filters=self.num_hiddens,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same',
                                   name='decoder_conv_1')(quant_top['quantize'])

        d_top_res = self.residual_stack_builder(decode_top[0].shape, self.num_residual_layers, self.num_residual_hiddens, name='decode_top_res_stack')

        decode_top = d_top_res(decode_top)

        decode_top = layers.Activation('relu')(decode_top)

        decode_top = layers.Conv2DTranspose(
            filters=self.embedding_dim,
            kernel_size=4,
            strides=2,
            padding='same')(decode_top)

        encoded_bottom = layers.concatenate([bottom_encode_out, decode_top])

        embed_bottom = layers.Conv2D(
            filters=self.embedding_dim,
            kernel_size=1,
            strides=1,
            padding='same',
            name='bottom_to_vq_conv'
        )(encoded_bottom)

        #bottom_quanter = snt.nets.VectorQuantizerEMA(self.embedding_dim, self.num_embeddings, self.commitment_cost, self.decay)
        #quant_bottom = layers.Lambda(lambda x: bottom_quanter(x,is_training=True))(embed_bottom)

        self.vq_bot = VQ(self.embedding_dim, self.num_embeddings, self.commitment_cost, name='vq_bottom')
        quant_bottom = self.vq_bot(embed_bottom)
        self.encoder = tf.keras.Model(inputs=input_img, outputs=[quant_top, quant_bottom], name='encoder')

        top_dec_input = layers.Input(shape=quant_top['quantize'][0].shape)
        bottom_dec_input = layers.Input(shape=quant_bottom['quantize'][0].shape)

        upsample_top = layers.Conv2DTranspose(filters=self.embedding_dim,
                                              kernel_size=4,
                                              strides=2,
                                              padding='same',
                                              name='upsample_top_conv')(top_dec_input)
        quant_combined = layers.concatenate([upsample_top, bottom_dec_input])
        decoding = layers.Conv2D(
            filters=self.num_hiddens,
            kernel_size=3,
            strides=1,
            padding='same',
            name='decoder_conv1'
        )(quant_combined)
        dec_res = self.residual_stack_builder(decoding[0].shape, self.num_residual_layers, self.num_residual_hiddens, name='decode_res_stack')
        decoding = dec_res(decoding)
        decoding = layers.Conv2DTranspose(
            filters=self.num_hiddens // 2,
            kernel_size=4,
            strides=2,
            padding='same',
            activation='relu'
        )(decoding)
        decode_out = layers.Conv2DTranspose(
            filters=self.num_channels,
            kernel_size=4,
            strides=2,
            padding='same'
        )(decoding)

        self.decoder=tf.keras.Model(inputs=[top_dec_input,bottom_dec_input], outputs=decode_out, name='decoder')

        self.model = tf.keras.Model(inputs=input_img, outputs=[self.decoder([quant_top['quantize'],quant_bottom['quantize']]), quant_top, quant_bottom],name='vqvae')


class VQ(layers.Layer):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost,
                 initializer='uniform', epsilon=1e-10, **kwargs):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.initializer = initializer
        self.epsilon = epsilon

        super(VQ, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add embedding weights.
        self.w = self.add_weight(name='embedding',
                                 shape=(self.embedding_dim, self.num_embeddings),
                                 initializer=self.initializer,
                                 trainable=True)

        # Finalize building.
        super(VQ, self).build(input_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embedding_dim': self.embedding_dim,
            'num_embeddings': self.num_embeddings,
            'commitment_cost': self.commitment_cost,
            'initializer': self.initializer,
            'epsilon': self.epsilon,
        })
        return config

    def call(self, x):
        # Flatten input except for last dimension.
        flat_inputs = tf.keras.backend.reshape(x, (-1, self.embedding_dim))

        # Calculate distances of input to embedding vectors.
        distances = (tf.keras.backend.sum(flat_inputs ** 2, axis=1, keepdims=True)
                     - 2 * tf.keras.backend.dot(flat_inputs, self.w)
                     + tf.keras.backend.sum(self.w ** 2, axis=0, keepdims=True))

        # Retrieve encoding indices.
        encoding_indices = tf.keras.backend.argmax(-distances, axis=1)
        encodings = tf.keras.backend.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = tf.keras.backend.reshape(encoding_indices, tf.keras.backend.shape(x)[:-1])
        quantized = self.quantize(encoding_indices)

        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + tf.stop_gradient(quantized - x)
        avg_probs = tf.reduce_mean(encodings, axis=0)
        perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.math.log(avg_probs + self.epsilon)))

        return {
            'quantize': quantized,
            'loss': loss,
            'perplexity': perplexity,
            'encodings': encodings,
            'encoding_indices': encoding_indices,
            'distances': distances
        }

    @property
    def embeddings(self):
        return self.w

    def quantize(self, encoding_indices):
        w = tf.keras.backend.transpose(self.embeddings.read_value())
        return tf.nn.embedding_lookup(w, encoding_indices)