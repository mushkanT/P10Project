import tensorflow as tf
import numpy as np
from functools import partial, lru_cache
import tensorflow_addons as tfa

def wn_linear(in_dim, out_dim):
    return tfa.layers.wrappers.WeightNormalization(tf.keras.layers.Dense(out_dim, in_dim))

def glu(kernel_shape, layer_input, layer_name, residual=None):
    """ Gated Linear Unit """
    # Pad the left side to prevent kernels from viewing future context
    kernel_width = kernel_shape[1]
    left_pad = kernel_width - 1
    paddings = [[0,0],[0,0],[left_pad,0],[0,0]]
    padded_input = tf.pad(layer_input, paddings, "CONSTANT")

    # Kaiming intialization
    stddev = np.sqrt(2.0 / (kernel_shape[1] * kernel_shape[2]))

    # First conv layer
    W_g = tf.Variable(stddev, dtype=tf.float32)
    W_v = tf.Variable(tf.random_normal(kernel_shape, stddev=stddev), name="W%s" % layer_name)
    W =  (W_g / tf.nn.l2_normalize(W_v, 0)) * W_v
    b = tf.Variable(tf.zeros([kernel_shape[2] * kernel_shape[3]]), name="b%s" % layer_name)
    conv1 = tf.nn.depthwise_conv2d(
        padded_input,
        W,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv1")
    conv1 = tf.nn.bias_add(conv1, b)

    # Second gating sigmoid layer
    V_g = tf.Variable(stddev, dtype=tf.float32)
    V_v = tf.Variable(tf.random_normal(kernel_shape, stddev=stddev), name="V%s" % layer_name)
    V = (V_g / tf.nn.l2_normalize(V_v, 0)) * V_v
    c = tf.Variable(tf.zeros([kernel_shape[2] * kernel_shape[3]]), name="c%s" % layer_name)
    conv2 = tf.nn.depthwise_conv2d(
        padded_input,
        V,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv2")
    conv2 = tf.nn.bias_add(conv2, c)

    # Preactivation residual
    if residual is not None:
        conv1 = tf.add(conv1, residual)
        conv2 = tf.add(conv2, residual)

    h = tf.multiply(conv1, tf.sigmoid(conv2, name="sig"))
    return h


class WNConv2d(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        activation=None,
    ):
        super().__init__()

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
    return tf.pad(input, [0, 0, size, 0])[:, :, : input.shape[2], :]


def shift_right(input, size=1):
    return tf.pad(input, [size, 0, 0, 0])[:, :, :, : input.shape[3]]


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
            pad =[[0,0],[kernel_width-1,0],[kernel_height -1 , 0], [0,0]]

        elif padding == 'down' or padding == 'causal':
            kw_floor = kernel_width // 2

            pad =[[0,0], [kw_floor, kw_floor], [kernel_height - 1, 0], [0,0]]

        self.causal = 0
        if padding == 'causal':
            self.causal = kernel_size[1] // 2
        self.pad = pad

        self.conv = WNConv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=0,
            activation=activation,
        )

    def __call__(self, input):
        out = tf.pad(input,self.pad)

        if self.causal > 0:
            self.conv.conv.weight_v.data[:, :, -1, self.causal :].zero_()

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
            conv_module = partial(WNConv2d, padding=kernel_size // 2)

        elif conv == 'causal_downright':
            conv_module = partial(CausalConv2d, padding='downright')

        elif conv == 'causal':
            conv_module = partial(CausalConv2d, padding='causal')

        self.activation = activation
        self.conv1 = conv_module(in_channel, channel, kernel_size)

        if auxiliary_channel > 0:
            self.aux_conv = WNConv2d(auxiliary_channel, channel, 1)

        self.dropout = tf.keras.layers.Dropout(dropout)

        self.conv2 = conv_module(channel, in_channel * 2, kernel_size)

        if condition_dim > 0:
            # self.condition = nn.Linear(condition_dim, in_channel * 2, bias=False)
            self.condition = WNConv2d(condition_dim, in_channel * 2, 1, bias=False)


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
            # out = out + condition.view(condition.shape[0], 1, 1, condition.shape[1])

        out = glu([1,1,1,1],out)
        out += input

        return out


@lru_cache(maxsize=64)
def causal_mask(size):
    shape = [size, size]
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8).T
    start_mask = np.ones(size).astype(np.float32)
    start_mask[0] = 0

    return (
        tf.convert_to_tensor(mask).unsqueeze(0),
        tf.convert_to_tensor(start_mask).unsqueeze(1),
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
        batch, _, height, width = key.shape

        def reshape(input):
            return input.view(batch, -1, self.n_head, self.dim_head).transpose(1, 2)

        query_flat = query.view(batch, query.shape[1], -1).transpose(1, 2)
        key_flat = key.view(batch, key.shape[1], -1).transpose(1, 2)
        query = reshape(self.query(query_flat))
        key = reshape(self.key(key_flat)).transpose(2, 3)
        value = reshape(self.value(key_flat))

        attn = tf.matmul(query, key) / tf.sqrt(self.dim_head)
        mask, start_mask = causal_mask(height * width)
        mask = mask.type_as(query)
        start_mask = start_mask.type_as(query)
        attn = attn.masked_fill(mask == 0, -1e4)
        attn = tf.keras.layers.Softmax(attn, 3) * start_mask
        attn = self.dropout(attn)

        out = attn @ value
        out = out.transpose(1, 2).reshape(
            batch, height, width, self.dim_head * self.n_head
        )
        out = out.permute(0, 3, 1, 2)

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

        blocks = [WNConv2d(in_channel, channel, kernel_size, padding=kernel_size // 2)]

        for i in range(n_res_block):
            blocks.append(GatedResBlock(channel, channel, kernel_size))

        self.blocks = tf.keras.Sequential(*blocks)

    def __call__(self, input):
        return self.blocks(input)


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

        coord_x = (tf.range(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (tf.range(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer('background', tf.concat([coord_x, coord_y], 3))

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

        out.extend([tf.keras.layers.ELU, WNConv2d(channel, n_class, 1)])

        self.out = tf.keras.Sequential(*out)

    def __call__(self, input, condition=None, cache=None):
        if cache is None:
            cache = {}
        batch, height, width = input.shape
        input = (
            tf.one_hot(input, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
        )
        horizontal = shift_down(self.horizontal(input))
        vertical = shift_right(self.vertical(input))
        out = horizontal + vertical

        background = self.background[:, :, :height, :].expand(batch, 2, height, width)

        if condition is not None:
            if 'condition' in cache:
                condition = cache['condition']
                condition = condition[:, :, :height, :]

            else:
                condition = (
                    tf.one_hot(condition, self.n_class)
                    .permute(0, 3, 1, 2)
                    .type_as(self.background)
                )
                condition = self.cond_resnet(condition)
                condition = tf.keras.layers.UpSampling2D.interpolate(condition, size=2)
                cache['condition'] = tf.identity(tf.stop_gradient(condition))
                condition = condition[:, :, :height, :]

        for block in self.blocks:
            out = block(out, background, condition=condition)

        out = self.out(out)

        return out, cache