from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.keras import layers
import sonnet as snt


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
        self.build_models()

    def residual_stack_builder(self, in_shape, num_residual_layers, num_residual_hiddens, name):
        res_input = layers.Input(shape=in_shape)
        encoded = layers.Activation('relu')(res_input)

        res3 = layers.Conv2D(filters=num_residual_hiddens,
                             kernel_size=3,
                             strides=1,
                             padding='same',
                             name="residual_3x3")

        res1 = layers.Conv2D(filters=self.num_hiddens,
                             kernel_size=3,
                             strides=1,
                             padding='same',
                             name='residual_1x1')

        res3_out = res3(encoded)
        res3_out = layers.Activation('relu')(res3_out)
        res1_out = res1(res3_out)

        res_encoding = layers.add([res_input, res1_out])

        if num_residual_layers == 2:
            res_encoding = layers.Activation('relu')(res_encoding)
            res3_out2 = res3(res_encoding)
            res3_out2 = layers.Activation('relu')(res3_out2)
            res1_out2 = res1(res3_out2)
            res_encoding = layers.add([res_encoding, res1_out2])
        else:
            raise Exception('Only support 1 or 2 residual layers')
        return tf.keras.Model(inputs=res_input, outputs=res_encoding, name=name)

    def build_models(self):
        input_img = layers.Input(shape=(self.image_size, self.image_size, self.num_channels))
        encode = layers.Conv2D(filters=self.num_hiddens // 2, kernel_size=4, strides=2, padding='same',
                               name='encoder_conv_1')(input_img)
        encode = layers.Activation('relu')(encode)
        encode = layers.Conv2D(filters=self.num_hiddens, kernel_size=4, strides=2, padding='same', name='encoder_conv_2')(
            encode)
        encode = layers.Activation('relu')(encode)
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
            padding='same'
        )(bottom_encode_out)

        top_encoding = layers.Activation('relu')(top_encoding)

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

        quanter = snt.nets.VectorQuantizerEMA(self.embedding_dim, self.num_embeddings, self.commitment_cost, self.decay)
        quant_layer = layers.Lambda(lambda x: quanter(x, is_training=True))

        quant_top = quant_layer(embed_top)
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

        bottom_quanter = snt.nets.VectorQuantizerEMA(self.embedding_dim, self.num_embeddings, self.commitment_cost, self.decay)
        quant_bottom = layers.Lambda(lambda x: bottom_quanter(x,is_training=True))(embed_bottom)

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
            padding='same'
        )(decoding)
        decoding = layers.Activation('relu')(decoding)
        decode_out = layers.Conv2DTranspose(
            filters=self.num_channels,
            kernel_size=4,
            strides=2,
            padding='same'
        )(decoding)

        self.decoder=tf.keras.Model(inputs=[top_dec_input,bottom_dec_input], outputs=decode_out, name='decoder')

        self.model = tf.keras.Model(inputs=input_img, outputs=self.decoder([quant_top['quantize'],quant_bottom['quantize']]),name='vqvae')




class Quantize(tf.keras.Layer):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        embed = tf.random.normal((dim,n_embed))
        #embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', tf.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def call(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = tf.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return layers.Embedding(embed_id, self.embed.transpose(0,1))(embed_id)


