import tensorflow as tf
from tensorflow import keras
import numpy as np
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


init = tf.keras.initializers.RandomNormal(stddev=0.02)
init = 'glorot_uniform'
#dcgan without batchnorm in disc (wgan, wgan-gp)
bn_mom = 0.99


# 32x32
def cifargan_gen(args):
    g_dim = args.g_dim
    z_dim = args.noise_dim
    img_dim = args.dataset_dim[1]
    channels = args.dataset_dim[3]
    img_resize = img_dim//(2*2*2)

    model = keras.Sequential()
    # foundation for 4x4 image
    model.add(layers.Dense(g_dim * img_resize * img_resize, input_dim=z_dim, kernel_initializer=init))
    model.add(layers.Reshape((img_resize, img_resize, g_dim)))
    # upsample to 8x8
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same', kernel_initializer=init))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    # output layer
    if args.input_scale:
        print('tanh')
        model.add(layers.Conv2D(channels, (6, 6), activation='tanh', padding='same', kernel_initializer=init))
    else:
        print('sigmoid')
        model.add(layers.Conv2D(channels, (3, 3), activation='sigmoid', padding='same', kernel_initializer=init))
    return model


def cifargan_disc(args):
    d_dim = args.d_dim
    input_dim = args.dataset_dim[1]
    channels = args.dataset_dim[3]
    model = keras.Sequential()

    # normal
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=[input_dim, input_dim, channels], kernel_initializer=init))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(layers.LeakyReLU(alpha=0.2))
    # classifier
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, kernel_initializer=init))
    # compile model
    return model


# 64x64
def gan64_gen(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*16*16)(noise)
    model = tf.keras.layers.Reshape((16, 16, 1024))(model)

    model = (tf.keras.layers.Conv2DTranspose(1024, (4,4), strides=(1, 1), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.LeakyReLU(alpha=0.2))(model)

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

    model = tf.keras.layers.Dense(1024*4*4)(noise)
    model = tf.keras.layers.Reshape((4, 4, 1024))(model)

    model = (tf.keras.layers.Conv2DTranspose(1024, (4,4), strides=(1, 1), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (4,4), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(64, (4,4), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    # Generator 1
    img1 = (tf.keras.layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same'))(model)
    img1 = (tf.keras.layers.BatchNormalization())(img1)
    img1 = (tf.keras.layers.PReLU())(img1)

    img1 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same')(img1)

    return keras.Model(noise, img1)


def gan128_disc(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    img1 = tf.keras.layers.Input(shape=img_shape)

    x1 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')(img1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU()(x1)

    x1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU()(x1)

    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())

    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())

    model.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())

    model.add(tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2048))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dense(1))

    output1 = model(x1)

    return keras.Model(img1, output1)


def gan256_gen(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*8*8)(noise)
    model = tf.keras.layers.Reshape((8, 8, 1024))(model)

    model = (tf.keras.layers.Conv2DTranspose(1024, (4,4), strides=(1, 1), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (4,4), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(64, (4,4), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.PReLU())(model)

    img1 = (tf.keras.layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same'))(model)
    img1 = (tf.keras.layers.BatchNormalization(momentum=0.8))(img1)
    img1 = (tf.keras.layers.PReLU())(img1)
    img1 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same')(img1)

    return keras.Model(noise, img1)


def gan256_disc(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    x1 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')(img1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU()(x1)
    x1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU()(x1)

    # Shared discriminator layers
    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())

    model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())

    model.add(tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())

    model.add(tf.keras.layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2048))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dense(1))

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


# CoGAN
# Mnist negative + edge
def cogan_generators_digit(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*4*4)(noise)
    model = tf.keras.layers.Reshape((4, 4, 1024))(model)

    model = (tf.keras.layers.Conv2DTranspose(1024, (4,4), strides=(1, 1), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (3,3), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(256, (3,3), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    # Generator 1
    img1 = tf.keras.layers.Conv2DTranspose(channels, (6,6), strides=(1, 1), activation='tanh', padding='same')(model)

    # Generator 2
    img2 = tf.keras.layers.Conv2DTranspose(channels, (6,6), strides=(1, 1), activation='tanh', padding='same')(model)

    return keras.Model(noise, img1), keras.Model(noise, img2)


def cogan_discriminators_digit(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    x1 = tf.keras.layers.Conv2D(20, (5, 5), padding='same')(img1)
    x1 = tf.keras.layers.MaxPool2D()(x1)

    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    x2 = tf.keras.layers.Conv2D(20, (5, 5), padding='same')(img2)
    x2 = tf.keras.layers.MaxPool2D()(x2)

    # Shared discriminator layers
    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(50, (5, 5), padding='same'))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(500))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dense(1))

    output1 = model(x1)
    output2 = model(x2)

    return keras.Model(img1, output1), keras.Model(img2, output2)


def cogan_generators_digit_noshare(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*4*4)(noise)
    model = tf.keras.layers.Reshape((4, 4, 1024))(model)

    # Generator 1
    model1 = (tf.keras.layers.Conv2DTranspose(1024, (4,4), strides=(1, 1), padding='same'))(model)
    model1 = (tf.keras.layers.BatchNormalization())(model1)
    model1 = (tf.keras.layers.PReLU())(model1)

    model1 = (tf.keras.layers.Conv2DTranspose(512, (3,3), strides=(2, 2), padding='same'))(model1)
    model1 = (tf.keras.layers.BatchNormalization())(model1)
    model1 = (tf.keras.layers.PReLU())(model1)

    model1 = (tf.keras.layers.Conv2DTranspose(256, (3,3), strides=(2, 2), padding='same'))(model1)
    model1 = (tf.keras.layers.BatchNormalization())(model1)
    model1 = (tf.keras.layers.PReLU())(model1)

    model1 = (tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same'))(model1)
    model1 = (tf.keras.layers.BatchNormalization())(model1)
    model1 = (tf.keras.layers.PReLU())(model1)
    img1 = tf.keras.layers.Conv2DTranspose(channels, (6,6), strides=(1, 1), activation='tanh', padding='same')(model1)

    # Generator 2
    model2 = (tf.keras.layers.Conv2DTranspose(1024, (4,4), strides=(1, 1), padding='same'))(model)
    model2 = (tf.keras.layers.BatchNormalization())(model2)
    model2 = (tf.keras.layers.PReLU())(model2)

    model2 = (tf.keras.layers.Conv2DTranspose(512, (3,3), strides=(2, 2), padding='same'))(model2)
    model2 = (tf.keras.layers.BatchNormalization())(model2)
    model2 = (tf.keras.layers.PReLU())(model2)

    model2 = (tf.keras.layers.Conv2DTranspose(256, (3,3), strides=(2, 2), padding='same'))(model2)
    model2 = (tf.keras.layers.BatchNormalization())(model2)
    model2 = (tf.keras.layers.PReLU())(model2)

    model2 = (tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same'))(model2)
    model2 = (tf.keras.layers.BatchNormalization())(model2)
    model2 = (tf.keras.layers.PReLU())(model2)
    img2 = tf.keras.layers.Conv2DTranspose(channels, (6,6), strides=(1, 1), activation='tanh', padding='same')(model2)

    return keras.Model(noise, img1), keras.Model(noise, img2)


def cogan_discriminators_digit_noshare(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    x1 = tf.keras.layers.Conv2D(20, (5, 5), padding='same')(img1)
    x1 = tf.keras.layers.MaxPool2D()(x1)

    model1 = tf.keras.layers.Conv2D(50, (5, 5), padding='same')(x1)
    model1 = tf.keras.layers.MaxPool2D()(model1)
    model1 = tf.keras.layers.Flatten()(model1)
    model1 = tf.keras.layers.Dense(500)(model1)
    model1 = tf.keras.layers.PReLU()(model1)
    model1 = tf.keras.layers.Dense(1)(model1)

    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    x2 = tf.keras.layers.Conv2D(20, (5, 5), padding='same')(img2)
    x2 = tf.keras.layers.MaxPool2D()(x2)

    model2 = tf.keras.layers.Conv2D(50, (5, 5), padding='same')(x2)
    model2 = tf.keras.layers.MaxPool2D()(model2)
    model2 = tf.keras.layers.Flatten()(model2)
    model2 = tf.keras.layers.Dense(500)(model2)
    model2 = tf.keras.layers.PReLU()(model2)
    model2 = tf.keras.layers.Dense(1)(model2)

    return keras.Model(img1, model1), keras.Model(img2, model2)


# Mnist rotate
def cogan_generators_rotate(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    # Shared weights between generators
    model = keras.Sequential()
    model.add(tf.keras.layers.Dense(1024, input_dim=args.noise_dim))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.BatchNormalization())

    feature_repr = model(noise)

    # Generator 1
    g1 = tf.keras.layers.Dense(np.prod(img_shape), activation='sigmoid')(feature_repr)
    img1 = tf.keras.layers.Reshape(img_shape)(g1)

    # Generator 2
    g2 = tf.keras.layers.Dense(np.prod(img_shape), activation='sigmoid')(feature_repr)
    img2 = tf.keras.layers.Reshape(img_shape)(g2)

    return keras.Model(noise, img1), keras.Model(noise, img2)


def cogan_discriminators_rotate(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    model1 = tf.keras.layers.Conv2D(20, (5,5), padding='same')(img1)
    model1 = tf.keras.layers.MaxPool2D()(model1)
    model1 = tf.keras.layers.Conv2D(50, (5,5), padding='same')(model1)
    model1 = tf.keras.layers.MaxPool2D()(model1)
    model1 = tf.keras.layers.Dense(500)(model1)
    model1 = tf.keras.layers.LeakyReLU()(model1)

    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    model2 = tf.keras.layers.Conv2D(20, (5,5), padding='same')(img2)
    model2 = tf.keras.layers.MaxPool2D()(model2)
    model2 = tf.keras.layers.Conv2D(50, (5,5), padding='same')(model2)
    model2 = tf.keras.layers.MaxPool2D()(model2)
    model2 = tf.keras.layers.Dense(500)(model2)
    model2 = tf.keras.layers.LeakyReLU()(model2)

    # Shared discriminator layers
    model = keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(8,8,500)))
    model.add(tf.keras.layers.Dense(1))

    validity1 = model(model1)
    validity2 = model(model2)

    return keras.Model(img1, validity1), keras.Model(img2, validity2)


# Faces
def cogan_generators_faces(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*4*4)(noise)
    model = tf.keras.layers.Reshape((4, 4, 1024))(model)

    model = (tf.keras.layers.Conv2DTranspose(1024, (4,4), strides=(1, 1), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (4,4), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(64, (4,4), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    # Generator 1
    img1 = (tf.keras.layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same'))(model)
    img1 = (tf.keras.layers.BatchNormalization())(img1)
    img1 = (tf.keras.layers.PReLU())(img1)
    img1 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same')(img1)

    # Generator 2
    img2 = (tf.keras.layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same'))(model)
    img2 = (tf.keras.layers.BatchNormalization())(img2)
    img2 = (tf.keras.layers.PReLU())(img2)
    img2 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same')(img2)

    return keras.Model(noise, img1), keras.Model(noise, img2)


def cogan_discriminators_faces(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    x1 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')(img1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU()(x1)
    x1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU()(x1)

    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    x2 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')(img2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.PReLU()(x2)
    x2 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.PReLU()(x2)

    # Shared discriminator layers
    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())

    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())

    model.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())

    model.add(tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2048))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dense(1))

    output1 = model(x1)
    output2 = model(x2)

    return keras.Model(img1, output1), keras.Model(img2, output2)


# 256x256 CoGANs
def cogan_generators_256(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*16*16)(noise)
    model = tf.keras.layers.Reshape((16, 16, 1024))(model)

    model = (tf.keras.layers.Conv2DTranspose(1024, (4,4), strides=(1, 1), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (4,4), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.LeakyReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(64, (4,4), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)

    # Generator 1
    img1 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same')(model)

    # Generator 2
    img2 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same')(model)

    return keras.Model(noise, img1), keras.Model(noise, img2)


def cogan_discriminators_256(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    x1 = tf.keras.layers.Conv2D(32, (5, 5), padding='same', strides=(2, 2))(img1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU()(x1)
    x1 = tf.keras.layers.Conv2D(64, (5, 5), padding='same', strides=(2, 2))(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU()(x1)

    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    x2 = tf.keras.layers.Conv2D(32, (5, 5), padding='same', strides=(2, 2))(img2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.PReLU()(x2)
    x2 = tf.keras.layers.Conv2D(64, (5, 5), padding='same', strides=(2, 2))(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.PReLU()(x2)

    # Shared discriminator layers
    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same', strides=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())

    model.add(tf.keras.layers.Conv2D(256, (5, 5), padding='same', strides=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())

    model.add(tf.keras.layers.Conv2D(512, (5, 5), padding='same', strides=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())

    model.add(tf.keras.layers.Conv2D(1024, (5, 5), padding='same', strides=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2048))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dense(1))

    output1 = model(x1)
    output2 = model(x2)

    return keras.Model(img1, output1), keras.Model(img2, output2)
