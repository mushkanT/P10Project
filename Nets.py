import tensorflow as tf
from tensorflow import keras
import numpy as np
import Utils as u
layers = tf.keras.layers


# Clip model weights to a given hypercube
class ClipConstraint(tf.keras.constraints.Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
       return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


prelu_init = tf.keras.initializers.Constant(0.25)


def encoder(args):
    input_dim = args.dataset_dim[1]
    channels = args.dataset_dim[3]
    model = keras.Sequential()

    # normal
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=[input_dim, input_dim, channels]))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # classifier
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(args.noise_dim))
    # compile model
    return model



# 32x32
def cifargan_gen(args):
    g_dim = args.g_dim
    z_dim = args.noise_dim
    img_dim = args.dataset_dim[1]
    channels = args.dataset_dim[3]
    img_resize = img_dim//(2*2*2)

    model = keras.Sequential()
    # foundation for 4x4 image
    model.add(layers.Dense(g_dim * img_resize * img_resize, input_dim=z_dim, kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(layers.Reshape((img_resize, img_resize, g_dim)))
    # upsample to 8x8
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    # output layer
    model.add(layers.Conv2D(channels, (6, 6), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    return model


def cifargan_disc(args):
    d_dim = args.d_dim
    input_dim = args.dataset_dim[1]
    channels = args.dataset_dim[3]
    model = keras.Sequential()

    # normal
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=[input_dim, input_dim, channels], kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(layers.LeakyReLU(alpha=0.2))
    # classifier
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    # compile model
    return model


# 64x64
def gan64_gen(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*16*16)(noise)
    model = tf.keras.layers.Reshape((16, 16, 1024))(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (3,3), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.LeakyReLU(alpha=0.2))(model)

    model = (tf.keras.layers.Conv2DTranspose(256, (3,3), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.LeakyReLU(alpha=0.2))(model)

    model = (tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.LeakyReLU(alpha=0.2))(model)

    model = (tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.LeakyReLU(alpha=0.2))(model)

    # Generator 1
    img1 = tf.keras.layers.Conv2DTranspose(channels, (6,6), strides=(1, 1), activation='tanh', padding='same')(model)

    return keras.Model(noise, img1)


def gan64_disc(args):
    d_dim = args.d_dim
    input_dim = args.dataset_dim[1]
    channels = args.dataset_dim[3]
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    model = keras.Sequential()
    # normal
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # classifier
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1))
    # compile model
    return model


def gan128_gen(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*4*4, kernel_initializer=args.w_init, kernel_regularizer=args.wd)(noise)
    model = tf.keras.layers.Reshape((4, 4, 1024))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(64, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    # Generator 1
    img1 = (tf.keras.layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    img1 = (tf.keras.layers.BatchNormalization())(img1)
    img1 = (tf.keras.layers.PReLU(prelu_init))(img1)

    img1 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)

    return keras.Model(noise, img1)


def gan128_disc(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    img1 = tf.keras.layers.Input(shape=img_shape)

    x1 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU(prelu_init)(x1)

    x1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU(prelu_init)(x1)

    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2048, kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(1, kernel_initializer=args.w_init, kernel_regularizer=args.wd))

    output1 = model(x1)

    return keras.Model(img1, output1)


def gan256_gen(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(2048*4*4, kernel_regularizer=args.wd)(noise)
    model = tf.keras.layers.Reshape((4, 4, 2048))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(1024, (4,4), strides=(2, 2), padding='same', kernel_regularizer=args.wd))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (4,4), strides=(2, 2), padding='same', kernel_regularizer=args.wd))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2, 2), padding='same', kernel_regularizer=args.wd))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2, 2), padding='same', kernel_regularizer=args.wd))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(64, (4,4), strides=(2, 2), padding='same', kernel_regularizer=args.wd))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    img1 = (tf.keras.layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same', kernel_regularizer=args.wd))(model)
    img1 = (tf.keras.layers.BatchNormalization(momentum=0.8))(img1)
    img1 = (tf.keras.layers.PReLU(prelu_init))(img1)
    img1 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same', kernel_regularizer=args.wd)(img1)

    return keras.Model(noise, img1)


def gan256_disc(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    img1 = tf.keras.layers.Input(shape=img_shape)
    x1 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=args.wd)(img1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU(prelu_init)(x1)

    x1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=args.wd)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU(prelu_init)(x1)

    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv2D(2048, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(1, kernel_regularizer=args.wd))

    output1 = model(x1)

    return keras.Model(img1, output1)


# Toy
def toy_gen(n_dim):
    inputs = keras.Input(shape=(n_dim,), name='digits')
    x = layers.Dense(128, activation='tanh', name='dense1')(inputs)
    x = layers.Dense(128, activation='tanh', name='dense2')(x)
    x = layers.Dense(128, activation='tanh', name='dense3')(x)
    outputs = layers.Dense(2, activation='linear', name='preds')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def toy_disc(args):
    inputs = keras.Input(shape=(args.batch_size, 2), name='digits')
    x = layers.Dense(128, activation='tanh', name='dense1')(inputs)
    x = layers.Dense(128, activation='tanh', name='dense2')(x)
    x = layers.Dense(128, activation='tanh', name='dense3')(x)
    outputs = layers.Dense(1, name='preds')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# Mnist negative + edge
def cogan_generators_digit(args):
    channels = args.dataset_dim[3]

    output1 = []
    output2 = []

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*4*4, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(noise)
    model = tf.keras.layers.Reshape((4, 4, 1024))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (3,3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    features_4x4 = (tf.keras.layers.PReLU())(model)
    output1.append(features_4x4)
    output2.append(features_4x4)

    model = (tf.keras.layers.Conv2DTranspose(256, (3,3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    features_8x8 = (tf.keras.layers.PReLU())(model)
    output1.append(features_8x8)
    output2.append(features_8x8)

    model = (tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)
    output1.append(model)
    output2.append(model)

    # Generator 1
    img1 = tf.keras.layers.Conv2DTranspose(channels, (6,6), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model)

    # Generator 2
    img2 = tf.keras.layers.Conv2DTranspose(channels, (6,6), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model)

    output1.append(img1)
    output2.append(img2)
    return keras.Model(noise, output1), keras.Model(noise, output2)


def cogan_discriminators_digit(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    x1 = tf.keras.layers.Conv2D(20, (5, 5), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)
    x1 = tf.keras.layers.MaxPool2D()(x1)

    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    x2 = tf.keras.layers.Conv2D(20, (5, 5), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img2)
    x2 = tf.keras.layers.MaxPool2D()(x2)

    # Shared discriminator layers
    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(50, (5, 5), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(500, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))

    output1 = model(x1, training=True)
    output2 = model(x2, training=True)

    return keras.Model(img1, output1), keras.Model(img2, output2)


def cogan_generators_digit_noshare(args):
    channels = args.dataset_dim[3]


    output1 = []
    output2 = []

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*4*4, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(noise)
    model = tf.keras.layers.Reshape((4, 4, 1024))(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.PReLU(prelu_init)(model)

    # Generator 1
    if args.use_firstlayer:
        model1 = (tf.keras.layers.Conv2DTranspose(1024, (4,4), strides=(1, 1), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
        model1 = (tf.keras.layers.BatchNormalization())(model1)
        features1_4x4 = (tf.keras.layers.PReLU(prelu_init))(model1)
        output1.append(features1_4x4)

        model1 = (tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(features1_4x4)
    else:
        model1 = (tf.keras.layers.Conv2DTranspose(512, (3,3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)

    model1 = (tf.keras.layers.BatchNormalization())(model1)
    features1_8x8 = (tf.keras.layers.PReLU(prelu_init))(model1)
    output1.append(features1_8x8)

    model1 = (tf.keras.layers.Conv2DTranspose(256, (3,3), strides=(2, 2), padding='same',kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(features1_8x8)
    model1 = (tf.keras.layers.BatchNormalization())(model1)
    features1_16x16 = (tf.keras.layers.PReLU(prelu_init))(model1)
    output1.append(features1_16x16)

    model1 = (tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(features1_16x16)
    model1 = (tf.keras.layers.BatchNormalization())(model1)
    model1 = (tf.keras.layers.PReLU(prelu_init))(model1)
    img1 = tf.keras.layers.Conv2DTranspose(channels, (6,6), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model1)

    output1.append(img1)

    # Generator 2
    if args.use_firstlayer:
        model2 = (tf.keras.layers.Conv2DTranspose(1024, (4,4), strides=(1, 1), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
        model2 = (tf.keras.layers.BatchNormalization())(model2)
        features2_4x4 = (tf.keras.layers.PReLU(prelu_init))(model2)
        output2.append(features2_4x4)
        model2 = (tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init,kernel_regularizer=args.wd, bias_initializer=args.bi))(features2_4x4)
    else:
        model2 = (tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init,kernel_regularizer=args.wd, bias_initializer=args.bi))(model)

    model2 = (tf.keras.layers.BatchNormalization())(model2)
    features2_8x8 = (tf.keras.layers.PReLU(prelu_init))(model2)
    output2.append(features2_8x8)

    model2 = (tf.keras.layers.Conv2DTranspose(256, (3,3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(features2_8x8)
    model2 = (tf.keras.layers.BatchNormalization())(model2)
    features2_16x16 = (tf.keras.layers.PReLU(prelu_init))(model2)
    output2.append(features2_16x16)

    model2 = (tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same',kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(features2_16x16)
    model2 = (tf.keras.layers.BatchNormalization())(model2)
    model2 = (tf.keras.layers.PReLU(prelu_init))(model2)
    img2 = tf.keras.layers.Conv2DTranspose(channels, (6,6), strides=(1, 1), activation='tanh', padding='same',kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model2)
    output2.append(img2)

    return keras.Model(noise, output1), keras.Model(noise, output2)


def cogan_discriminators_digit_noshare(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    x1 = tf.keras.layers.Conv2D(20, (5, 5), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)
    x1 = tf.keras.layers.MaxPool2D()(x1)

    model1 = tf.keras.layers.Conv2D(50, (5, 5), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(x1)
    model1 = tf.keras.layers.MaxPool2D()(model1)
    model1 = tf.keras.layers.Flatten()(model1)
    model1 = tf.keras.layers.Dense(500, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model1)
    model1 = tf.keras.layers.PReLU(prelu_init)(model1)
    model1 = tf.keras.layers.Dropout(0.5)(model1)
    model1 = tf.keras.layers.Dense(1, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model1)

    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    x2 = tf.keras.layers.Conv2D(20, (5, 5), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img2)
    x2 = tf.keras.layers.MaxPool2D()(x2)

    model2 = tf.keras.layers.Conv2D(50, (5, 5), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(x2)
    model2 = tf.keras.layers.MaxPool2D()(model2)
    model2 = tf.keras.layers.Flatten()(model2)
    model2 = tf.keras.layers.Dense(500, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model2)
    model2 = tf.keras.layers.PReLU(prelu_init)(model2)
    model2 = tf.keras.layers.Dropout(0.5)(model2)
    model2 = tf.keras.layers.Dense(1, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model2)

    return keras.Model(img1, model1), keras.Model(img2, model2)


# Mnist rotate
def cogan_generators_rotate(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    # Shared weights between generators
    model = keras.Sequential()
    model.add(tf.keras.layers.Dense(1024, input_dim=args.noise_dim, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1024, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1024, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1024, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.BatchNormalization())

    feature_repr = model(noise)

    # Generator 1
    g1 = tf.keras.layers.Dense(np.prod(img_shape), activation='sigmoid', kernel_regularizer=args.wd, bias_initializer=args.bi)(feature_repr)
    img1 = tf.keras.layers.Reshape(img_shape)(g1)

    # Generator 2
    g2 = tf.keras.layers.Dense(np.prod(img_shape), activation='sigmoid', kernel_regularizer=args.wd, bias_initializer=args.bi)(feature_repr)
    img2 = tf.keras.layers.Reshape(img_shape)(g2)

    return keras.Model(noise, img1), keras.Model(noise, img2)


def cogan_discriminators_rotate(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    model1 = tf.keras.layers.Conv2D(20, (5,5), padding='same', kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)
    model1 = tf.keras.layers.MaxPool2D()(model1)
    model1 = tf.keras.layers.Conv2D(50, (5,5), padding='same', kernel_regularizer=args.wd, bias_initializer=args.bi)(model1)
    model1 = tf.keras.layers.MaxPool2D()(model1)
    model1 = tf.keras.layers.Dense(500, kernel_regularizer=args.wd, bias_initializer=args.bi)(model1)
    model1 = tf.keras.layers.LeakyReLU()(model1)

    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    model2 = tf.keras.layers.Conv2D(20, (5,5), padding='same', kernel_regularizer=args.wd, bias_initializer=args.bi)(img2)
    model2 = tf.keras.layers.MaxPool2D()(model2)
    model2 = tf.keras.layers.Conv2D(50, (5,5), padding='same', kernel_regularizer=args.wd, bias_initializer=args.bi)(model2)
    model2 = tf.keras.layers.MaxPool2D()(model2)
    model2 = tf.keras.layers.Dense(500, kernel_regularizer=args.wd, bias_initializer=args.bi)(model2)
    model2 = tf.keras.layers.LeakyReLU()(model2)

    # Shared discriminator layers
    model = keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(8,8,500)))
    model.add(tf.keras.layers.Dense(1, kernel_regularizer=args.wd, bias_initializer=args.bi))

    validity1 = model(model1)
    validity2 = model(model2)

    return keras.Model(img1, validity1), keras.Model(img2, validity2)

# Faces No share
def cogan_generators_faces_noshare(args):
    channels = args.dataset_dim[3]

    output1 = []
    output2 = []

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024 * 4 * 4, kernel_initializer=args.w_init, kernel_regularizer=args.wd,
                                  bias_initializer=args.bi)(noise)
    model = tf.keras.layers.Reshape((4, 4, 1024))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model1 = (tf.keras.layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model1 = (tf.keras.layers.BatchNormalization())(model1)
    feature1_1 = (tf.keras.layers.PReLU(prelu_init))(model1)
    output1.append(feature1_1)

    model1 = (tf.keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(feature1_1)
    model1 = (tf.keras.layers.BatchNormalization())(model1)
    feature2_1 = (tf.keras.layers.PReLU(prelu_init))(model1)
    output1.append(feature2_1)

    model1 = (tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init,kernel_regularizer=args.wd, bias_initializer=args.bi))(feature2_1)
    model1 = (tf.keras.layers.BatchNormalization())(model1)
    feature3_1 = (tf.keras.layers.PReLU(prelu_init))(model1)
    output1.append(feature3_1)

    model1 = (tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(feature3_1)
    model1 = (tf.keras.layers.BatchNormalization())(model1)
    feature4_1 = (tf.keras.layers.PReLU(prelu_init))(model1)
    output1.append(feature4_1)

    # Generator 1
    img1 = (tf.keras.layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(feature4_1)
    img1 = (tf.keras.layers.BatchNormalization())(img1)
    img1 = (tf.keras.layers.PReLU(prelu_init))(img1)
    img1 = tf.keras.layers.Conv2DTranspose(channels, (3, 3), strides=(1, 1), activation='tanh', padding='same',
                                           kernel_initializer=args.w_init, kernel_regularizer=args.wd,
                                           bias_initializer=args.bi)(img1)
    output1.append(img1)


    model2 = (tf.keras.layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model2 = (tf.keras.layers.BatchNormalization())(model2)
    feature1_2 = (tf.keras.layers.PReLU(prelu_init))(model2)
    output2.append(feature1_2)

    model2 = (tf.keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(feature1_2)
    model2 = (tf.keras.layers.BatchNormalization())(model2)
    feature2_2 = (tf.keras.layers.PReLU(prelu_init))(model2)
    output2.append(feature2_2)

    model2 = (tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init,
                                        kernel_regularizer=args.wd, bias_initializer=args.bi))(feature2_2)
    model2 = (tf.keras.layers.BatchNormalization())(model2)
    feature3_2 = (tf.keras.layers.PReLU(prelu_init))(model2)
    output2.append(feature3_2)

    model2 = (
        tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init,
                                        kernel_regularizer=args.wd, bias_initializer=args.bi))(feature3_2)
    model2 = (tf.keras.layers.BatchNormalization())(model2)
    feature4_2 = (tf.keras.layers.PReLU(prelu_init))(model2)
    output2.append(feature4_2)


    # Generator 2
    img2 = (tf.keras.layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init,
                                            kernel_regularizer=args.wd, bias_initializer=args.bi))(feature4_2)
    img2 = (tf.keras.layers.BatchNormalization())(img2)
    img2 = (tf.keras.layers.PReLU(prelu_init))(img2)
    img2 = tf.keras.layers.Conv2DTranspose(channels, (3, 3), strides=(1, 1), activation='tanh', padding='same',
                                           kernel_initializer=args.w_init, kernel_regularizer=args.wd,
                                           bias_initializer=args.bi)(img2)
    output2.append(img2)

    return keras.Model(noise, output1), keras.Model(noise, output2)

def cogan_discriminators_faces_noshare(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    x1 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init,
                                kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)
    x1 = tf.keras.layers.LayerNormalization()(x1)
    x1 = tf.keras.layers.PReLU(prelu_init)(x1)
    x1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init,
                                kernel_regularizer=args.wd, bias_initializer=args.bi)(x1)
    x1 = tf.keras.layers.LayerNormalization()(x1)
    x1 = tf.keras.layers.PReLU(prelu_init)(x1)

    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    x2 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init,
                                kernel_regularizer=args.wd, bias_initializer=args.bi)(img2)
    x2 = tf.keras.layers.LayerNormalization()(x2)
    x2 = tf.keras.layers.PReLU(prelu_init)(x2)
    x2 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init,
                                kernel_regularizer=args.wd, bias_initializer=args.bi)(x2)
    x2 = tf.keras.layers.LayerNormalization()(x2)
    x2 = tf.keras.layers.PReLU(prelu_init)(x2)

    # Shared discriminator layers
    model1 = keras.Sequential()
    model1.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init,
                                     kernel_regularizer=args.wd, bias_initializer=args.bi))
    model1.add(tf.keras.layers.LayerNormalization())
    model1.add(tf.keras.layers.PReLU(prelu_init))
    model1.add(tf.keras.layers.Dropout(0.1))

    model1.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init,
                                     kernel_regularizer=args.wd, bias_initializer=args.bi))
    model1.add(tf.keras.layers.LayerNormalization())
    model1.add(tf.keras.layers.PReLU(prelu_init))
    model1.add(tf.keras.layers.Dropout(0.3))

    model1.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init,
                                     kernel_regularizer=args.wd, bias_initializer=args.bi))
    model1.add(tf.keras.layers.LayerNormalization())
    model1.add(tf.keras.layers.PReLU(prelu_init))
    model1.add(tf.keras.layers.Dropout(0.3))

    model1.add(tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init,
                                     kernel_regularizer=args.wd, bias_initializer=args.bi))
    model1.add(tf.keras.layers.LayerNormalization())
    model1.add(tf.keras.layers.PReLU(prelu_init))
    model1.add(tf.keras.layers.Dropout(0.5))

    model1.add(tf.keras.layers.Flatten())
    model1.add(tf.keras.layers.Dense(2048, kernel_initializer=args.w_init, kernel_regularizer=args.wd,
                                    bias_initializer=args.bi))
    model1.add(tf.keras.layers.LayerNormalization())
    model1.add(tf.keras.layers.PReLU(prelu_init))
    model1.add(tf.keras.layers.Dropout(0.5))

    model1.add(
        tf.keras.layers.Dense(1, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))

    model2 = keras.Sequential()
    model2.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init,
                                     kernel_regularizer=args.wd, bias_initializer=args.bi))
    model2.add(tf.keras.layers.LayerNormalization())
    model2.add(tf.keras.layers.PReLU(prelu_init))
    model2.add(tf.keras.layers.Dropout(0.1))

    model2.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init,
                                     kernel_regularizer=args.wd, bias_initializer=args.bi))
    model2.add(tf.keras.layers.LayerNormalization())
    model2.add(tf.keras.layers.PReLU(prelu_init))
    model2.add(tf.keras.layers.Dropout(0.3))

    model2.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init,
                                     kernel_regularizer=args.wd, bias_initializer=args.bi))
    model2.add(tf.keras.layers.LayerNormalization())
    model2.add(tf.keras.layers.PReLU(prelu_init))
    model2.add(tf.keras.layers.Dropout(0.3))

    model2.add(tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init,
                                     kernel_regularizer=args.wd, bias_initializer=args.bi))
    model2.add(tf.keras.layers.LayerNormalization())
    model2.add(tf.keras.layers.PReLU(prelu_init))
    model2.add(tf.keras.layers.Dropout(0.5))

    model2.add(tf.keras.layers.Flatten())
    model2.add(tf.keras.layers.Dense(2048, kernel_initializer=args.w_init, kernel_regularizer=args.wd,
                                    bias_initializer=args.bi))
    model2.add(tf.keras.layers.LayerNormalization())
    model2.add(tf.keras.layers.PReLU(prelu_init))
    model2.add(tf.keras.layers.Dropout(0.5))

    model2.add(
        tf.keras.layers.Dense(1, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))

    output1 = model1(x1)
    output2 = model2(x2)

    return keras.Model(img1, output1), keras.Model(img2, output2)


# Faces
def cogan_generators_faces(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*4*4, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(noise)
    model = tf.keras.layers.Reshape((4, 4, 1024))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(64, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    # Generator 1
    img1 = (tf.keras.layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    img1 = (tf.keras.layers.BatchNormalization())(img1)
    img1 = (tf.keras.layers.PReLU(prelu_init))(img1)
    img1 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)

    # Generator 2
    img2 = (tf.keras.layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    img2 = (tf.keras.layers.BatchNormalization())(img2)
    img2 = (tf.keras.layers.PReLU(prelu_init))(img2)
    img2 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img2)

    return keras.Model(noise, img1), keras.Model(noise, img2)


def cogan_discriminators_faces(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    x1 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU(prelu_init)(x1)
    x1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU(prelu_init)(x1)

    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    x2 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.PReLU(prelu_init)(x2)
    x2 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.PReLU(prelu_init)(x2)

    # Shared discriminator layers
    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2048, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(1, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))

    output1 = model(x1, training=True)
    output2 = model(x2, training=True)

    return keras.Model(img1, output1), keras.Model(img2, output2)

def patch_cogan_disc(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    x1 = tf.keras.layers.Conv2D(32, 4, (2, 2), padding='same', kernel_initializer=args.w_init)(img1)
    x1 = tf.keras.layers.LeakyReLU(0.2)(x1)

    model1 = tf.keras.layers.Conv2D(64, 4, (2, 2), padding='same', kernel_initializer=args.w_init)(x1)
    model1 = tf.keras.layers.LayerNormalization()(model1)
    model1 = tf.keras.layers.LeakyReLU(0.2)(model1)
    model1 = tf.keras.layers.Conv2D(128, 4, (2, 2), padding='same', kernel_initializer=args.w_init)(model1)
    model1 = tf.keras.layers.LayerNormalization()(model1)
    model1 = tf.keras.layers.LeakyReLU(0.2)(model1)
    model1 = tf.keras.layers.Conv2D(256, 4, (2, 2), padding='same', kernel_initializer=args.w_init)(model1)
    model1 = tf.keras.layers.LayerNormalization()(model1)
    model1 = tf.keras.layers.LeakyReLU(0.2)(model1)

    out1 = tf.keras.layers.Conv2D(1,(4,4),padding='same', kernel_initializer=args.w_init)(model1)


    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    x2 = tf.keras.layers.Conv2D(32, 4, (2, 2), padding='same', kernel_initializer=args.w_init)(img2)
    x2 = tf.keras.layers.LeakyReLU(0.2)(x2)

    model2 = tf.keras.layers.Conv2D(64, 4, (2, 2), padding='same', kernel_initializer=args.w_init)(x2)
    model2 = tf.keras.layers.LayerNormalization()(model2)
    model2 = tf.keras.layers.LeakyReLU(0.2)(model2)
    model2 = tf.keras.layers.Conv2D(128, 4, (2, 2), padding='same', kernel_initializer=args.w_init)(model2)
    model2 = tf.keras.layers.LayerNormalization()(model2)
    model2 = tf.keras.layers.LeakyReLU(0.2)(model2)
    model2 = tf.keras.layers.Conv2D(256, 4, (2, 2), padding='same', kernel_initializer=args.w_init)(model2)
    model2 = tf.keras.layers.LayerNormalization()(model2)
    model2 = tf.keras.layers.LeakyReLU(0.2)(model2)

    out2 = tf.keras.layers.Conv2D(1,(4,4),padding='same', kernel_initializer=args.w_init)(model2)

    return keras.Model(img1, out1), keras.Model(img2, out2)


# 256x256 CoGANs
def cogan_generators_256(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(2048*4*4, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(noise)
    model = tf.keras.layers.Reshape((4, 4, 2048))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(1024, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.LeakyReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.LeakyReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(64, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(prelu_init))(model)

    # Generator 1
    img1 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model)

    # Generator 2
    img2 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model)

    return keras.Model(noise, img1), keras.Model(noise, img2)


def cogan_discriminators_256(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    x1 = tf.keras.layers.Conv2D(32, (5, 5), padding='same', strides=(2, 2), kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU(prelu_init)(x1)
    x1 = tf.keras.layers.Conv2D(64, (5, 5), padding='same', strides=(2, 2), kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU(prelu_init)(x1)

    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    x2 = tf.keras.layers.Conv2D(32, (5, 5), padding='same', strides=(2, 2), kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.PReLU(prelu_init)(x2)
    x2 = tf.keras.layers.Conv2D(64, (5, 5), padding='same', strides=(2, 2), kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.PReLU(prelu_init)(x2)

    # Shared discriminator layers
    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same', strides=(2, 2), kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Conv2D(256, (5, 5), padding='same', strides=(2, 2), kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(512, (5, 5), padding='same', strides=(2, 2), kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(1024, (5, 5), padding='same', strides=(2, 2), kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2048, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(1, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))

    output1 = model(x1)
    output2 = model(x2)

    return keras.Model(img1, output1), keras.Model(img2, output2)


def mnist_classifier(args, num_classes):
    img_shape = (32, 32, 3)
    input = tf.keras.layers.Input(shape=img_shape)
    model = tf.keras.layers.Conv2D(32, (3,3))(input)
    model = tf.keras.layers.LeakyReLU()(model)
    model = tf.keras.layers.Conv2D(64, (3,3))(model)
    model = tf.keras.layers.LeakyReLU()(model)
    model = tf.keras.layers.MaxPooling2D((2,2))(model)
    model = tf.keras.layers.Dropout(0.25)(model)
    model = tf.keras.layers.Flatten()(model)
    model = tf.keras.layers.Dense(128)(model)
    model = tf.keras.layers.LeakyReLU()(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(model)

    return tf.keras.Model(input, output)


def celeba_classifier(args, num_classes):
    img_shape = (128,128,3)
    input = tf.keras.layers.Input(shape=img_shape)

    model = tf.keras.layers.Conv2D(32, (3,3))(input)
    model = tf.keras.layers.LeakyReLU()(model)
    model = tf.keras.layers.Conv2D(64, (3,3))(model)
    model = tf.keras.layers.LeakyReLU()(model)
    model = tf.keras.layers.MaxPooling2D((2,2))(model)
    model = tf.keras.layers.Dropout(0.25)(model)
    model = tf.keras.layers.Flatten()(model)
    model = tf.keras.layers.Dense(128)(model)
    model = tf.keras.layers.LeakyReLU()(model)
    model = tf.keras.layers.Dropout(0.5)(model)

    output = tf.keras.layers.Dense(num_classes, activation='sigmoid')(model)
    return tf.keras.Model(input, output)





