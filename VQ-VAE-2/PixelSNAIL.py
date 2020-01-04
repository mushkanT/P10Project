import random
import tensorflow as tf
import numpy as np
from functools import lru_cache
import tensorflow_addons as tfa
import math
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

def wn_linear(in_dim, out_dim):
    return tfa.layers.wrappers.WeightNormalization(tf.keras.layers.Dense(out_dim))

class WNConv2d(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding='valid',
        bias=True,
        activation=None,
    ):
        super().__init__()

        if isinstance(padding,(int, float)):
            padding = 'same'

        self.conv = tfa.layers.wrappers.WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=out_channel,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                use_bias=bias,
            )
        )

        self.out_channel = out_channel

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        self.kernel_size = kernel_size

        self.activation = activation

    def __call__(self, input):
        out = self.conv(input)

        if self.activation is not None:
            out = self.activation(out)

        return out


def shift_down(input, size=1):
    return tf.pad(input, [[0,0], [size,0], [0, 0], [0,0]])[:, :input.shape[1], :, :]


def shift_right(input, size=1):
    return tf.pad(input, [[0,0], [0,0], [size, 0], [0, 0]])[:, :, :input.shape[2], :]


class CausalConv2d(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding='downright',
        activation=None,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 2

        self.kernel_size = kernel_size
        kernel_height = kernel_size[0]
        kernel_width = kernel_size[1]

        if padding == 'downright':
            pad =[[0,0], [kernel_height -1, 0],[kernel_width-1, 0], [0, 0]]

        elif padding == 'down' or padding == 'causal':
            kw_floor = kernel_width // 2

            pad =[[0,0], [kernel_height - 1, 0], [kw_floor, kw_floor], [0,0]]

        self.causal = 0
        if padding == 'causal':
            self.causal = kernel_width // 2
        self.pad = pad

        self.conv = WNConv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding='valid',
            activation=activation,
        )

    def __call__(self, input):
        out = tf.pad(input,self.pad)

        if self.causal > 0:
            if not self.conv.conv.built:
                self.conv.conv.build(out.shape)
            self.conv.conv.v[-1, self.causal:, :, :].assign(tf.zeros_like(self.conv.conv.v[-1, self.causal:, :, :]))


        out = self.conv(out)

        return out


class GatedResBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channel,
        channel,
        kernel_size,
        conv='wnconv2d',
        activation=tf.keras.activations.elu,
        dropout=0.1,
        auxiliary_channel=0,
        condition_dim=0,
    ):
        super().__init__()

        if conv == 'wnconv2d':
            self.conv1 = WNConv2d(in_channel,channel,kernel_size, padding=kernel_size//2)
            self.conv2 = WNConv2d(channel, in_channel * 2, kernel_size, padding=kernel_size//2)

        elif conv == 'causal_downright':
            self.conv1 = CausalConv2d(in_channel, channel, kernel_size, padding='downright')
            self.conv2 = CausalConv2d(channel, in_channel*2, kernel_size, padding='downright')

        elif conv == 'causal':
            self.conv1 = CausalConv2d(in_channel, channel, kernel_size, padding='causal')
            self.conv2 = CausalConv2d(channel, in_channel * 2, kernel_size, padding='causal')

        self.activation = activation

        if auxiliary_channel > 0:
            self.aux_conv = WNConv2d(auxiliary_channel, channel, 1)

        self.dropout = tf.keras.layers.Dropout(dropout)

        if condition_dim > 0:
            self.condition = WNConv2d(condition_dim, in_channel * 2, 1, bias=True)

    def __call__(self, input, aux_input=None, condition=None):
        out = self.conv1(self.activation(input))

        if aux_input is not None:
            out = out + self.aux_conv(self.activation(aux_input))

        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if condition is not None:
            condition = self.condition(condition)
            out += condition
        out = new_glu(out, dim=3)
        out += input

        return out
def new_glu(input, dim=3):
    a,b = tf.split(input, num_or_size_splits=2, axis=dim)
    sig_b = tf.sigmoid(b)
    return a * sig_b

@lru_cache(maxsize=64)
def causal_mask(size):
    shape = [size, size]
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8).T
    start_mask = np.ones(size).astype(np.float32)
    start_mask[0] = 0

    return (
        tf.reshape(tf.convert_to_tensor(mask), [1, mask.shape[0], mask.shape[1]]),
        tf.reshape(tf.convert_to_tensor(start_mask),[start_mask.shape[0], 1]),
    )


class CausalAttention(tf.keras.layers.Layer):
    def __init__(self, query_channel, key_channel, channel, n_head=8, dropout=0.1):
        super().__init__()

        self.query = wn_linear(query_channel, channel)
        self.key = wn_linear(key_channel, channel)
        self.value = wn_linear(key_channel, channel)

        self.dim_head = channel // n_head
        self.n_head = n_head

        self.dropout = tf.keras.layers.Dropout(dropout)

    def __call__(self, query, key):
        batch, height, width, _ = key.shape

        def reshape(input):
            return tf.transpose(tf.reshape(input,[batch, -1, self.n_head, self.dim_head]),perm=[0,2,1,3])

        query_flat = tf.reshape(query, [batch,-1,query.shape[3]])
        key_flat = tf.reshape(key, [batch, -1, key.shape[3]])
        query = reshape(self.query(query_flat))
        key = tf.transpose(reshape(self.key(key_flat)),perm=[0,1,3,2])
        value = reshape(self.value(key_flat))

        attn = tf.matmul(query, key) / math.sqrt(self.dim_head)
        mask, start_mask = causal_mask(height * width)
        mask = tf.cast(mask, query.dtype)
        start_mask = tf.cast(start_mask, query.dtype)
        attn = tf.where(mask == 0., -1e4, tf.cast(attn,tf.float32))
        attn = (tf.keras.layers.Softmax(3)(attn)) * start_mask
        attn = self.dropout(attn)

        out = attn @ value
        out = tf.transpose(out, [0,2,1,3])
        out = tf.reshape(out,[batch, height, width, self.dim_head * self.n_head])

        return out


class PixelBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channel,
        channel,
        kernel_size,
        n_res_block,
        attention=True,
        dropout=0.1,
        condition_dim=0,
    ):
        super().__init__()

        resblocks = []
        for i in range(n_res_block):
            resblocks.append(
                GatedResBlock(
                    in_channel,
                    channel,
                    kernel_size,
                    conv='causal',
                    dropout=dropout,
                    condition_dim=condition_dim,
                )
            )

        self.resblocks = resblocks

        self.attention = attention

        if attention:
            self.key_resblock = GatedResBlock(
                in_channel * 2 + 2, in_channel, 1, dropout=dropout
            )
            self.query_resblock = GatedResBlock(
                in_channel + 2, in_channel, 1, dropout=dropout
            )

            self.causal_attention = CausalAttention(
                in_channel + 2, in_channel * 2 + 2, in_channel // 2, dropout=dropout
            )

            self.out_resblock = GatedResBlock(
                in_channel,
                in_channel,
                1,
                auxiliary_channel=in_channel // 2,
                dropout=dropout,
            )

        else:
            self.out = WNConv2d(in_channel + 2, in_channel, 1)

    def __call__(self, input, background, condition=None):
        out = input

        for resblock in self.resblocks:
            out = resblock(out, condition=condition)

        if self.attention:
            key_cat = tf.concat([input, out, background], 3)
            key = self.key_resblock(key_cat)
            query_cat = tf.concat([out, background], 3)
            query = self.query_resblock(query_cat)
            attn_out = self.causal_attention(query, key)
            out = self.out_resblock(out, attn_out)

        else:
            bg_cat = tf.concat([out, background], 3)
            out = self.out(bg_cat)

        return out


class CondResNet(tf.keras.layers.Layer):
    def __init__(self, in_channel, channel, kernel_size, n_res_block):
        super().__init__()

        self.blocks = [WNConv2d(in_channel, channel, kernel_size, padding=kernel_size // 2)]

        for i in range(n_res_block):
            self.blocks.append(GatedResBlock(channel, channel, kernel_size))

        #self.blocks = tf.keras.Sequential(blocks)

    def __call__(self, input):
        for block in self.blocks:
            input = block(input)
        return input

class PixelSNAIL(tf.keras.Model):
    def __init__(
        self,
        shape,
        n_class,
        channel,
        kernel_size,
        n_block,
        n_res_block,
        res_channel,
        attention=True,
        dropout=0.1,
        n_cond_res_block=0,
        cond_res_channel=0,
        cond_res_kernel=3,
        n_out_res_block=0,
    ):
        super().__init__()

        height, width = shape

        self.n_class = n_class

        if kernel_size % 2 == 0:
            kernel = kernel_size + 1

        else:
            kernel = kernel_size

        self.horizontal = CausalConv2d(
            n_class, channel, [kernel // 2, kernel], padding='down'
        )
        self.vertical = CausalConv2d(
            n_class, channel, [(kernel + 1) // 2, kernel // 2], padding='downright'
        )

        coord_x = (tf.cast(tf.range(height), dtype=tf.float32) - height / 2) / height
        coord_x = tf.reshape(coord_x, [1,height,1,1])
        coord_x = tf.broadcast_to(coord_x, [1,height,width,1])
        coord_y = (tf.cast(tf.range(width), dtype=tf.float32) - width / 2) / width
        coord_y = tf.reshape(coord_y, [1,1,width,1])
        coord_y = tf.broadcast_to(coord_y, [1,height,width,1])
        self.register_bufffer = tf.concat([coord_x,coord_y],3)

        self.blocks = []

        for i in range(n_block):
            self.blocks.append(
                PixelBlock(
                    channel,
                    res_channel,
                    kernel_size,
                    n_res_block,
                    attention=attention,
                    dropout=dropout,
                    condition_dim=cond_res_channel,
                )
            )

        if n_cond_res_block > 0:
            self.cond_resnet = CondResNet(
                n_class, cond_res_channel, cond_res_kernel, n_cond_res_block
            )

        out = []

        for i in range(n_out_res_block):
            out.append(GatedResBlock(channel, res_channel, 1))

        out.extend([tf.keras.layers.ELU(), WNConv2d(channel, n_class, 1)])

        self.out = tf.keras.Sequential(out)


    def __call__(self, input, condition=None, cache=None):
        if cache is None:
            cache = {}
        batch, height, width = input.shape
        input = tf.one_hot(input, self.n_class)
        input = tf.cast(input, dtype=tf.float32)

        pre_horizontal = self.horizontal(input)
        horizontal = shift_down(pre_horizontal)
        pre_vertical = self.vertical(input)
        vertical = shift_right(pre_vertical)
        out = tf.keras.layers.add([horizontal, vertical])

        background = tf.broadcast_to(self.register_bufffer[:, :height, : , :],[batch, height, width, 2])

        if condition is not None:
            if 'condition' in cache:
                condition = cache['condition']
                condition = condition[:, :height, :, :]
            else:
                condition = tf.one_hot(condition, self.n_class)
                condition = tf.cast(condition, dtype=tf.float32)
                condition = self.cond_resnet(condition)
                condition = tf.keras.layers.UpSampling2D(size=2)(condition)
                cache['condition'] = tf.identity(tf.stop_gradient(condition))
                condition = condition[:, :height, :, :]

        for block in self.blocks:
            out = block(out, background, condition=condition)

        out = self.out(out)

        return out, cache