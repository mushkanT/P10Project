import tensorflow as tf
from tensorflow import keras
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


#32x32
def cifargan_bn_gen(args):
    g_dim = args.g_dim
    z_dim = args.noise_dim
    img_dim = args.dataset_dim[1]
    channels = args.dataset_dim[3]
    img_resize = img_dim//(2*2*2)

    model = keras.Sequential()
    # foundation for 4x4 image
    model.add(layers.Dense(g_dim * img_resize * img_resize, input_dim=z_dim, kernel_initializer=init))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((img_resize, img_resize, g_dim)))
    # upsample to 8x8
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    #model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization(momentum=bn_mom))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    #model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization(momentum=bn_mom))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    #model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization(momentum=bn_mom))
    model.add(layers.LeakyReLU(alpha=0.2))
    # output layer
    model.add(layers.Conv2D(channels, (3, 3), activation='tanh', padding='same', kernel_initializer=init))
    return model


def cifargan_bn_disc(args):
    d_dim = args.d_dim
    input_dim = args.dataset_dim[1]
    channels = args.dataset_dim[3]

    model = keras.Sequential()
    # normal
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=[input_dim, input_dim, channels], kernel_initializer=init))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(layers.BatchNormalization(momentum=bn_mom))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(layers.BatchNormalization(momentum=bn_mom))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(layers.BatchNormalization(momentum=bn_mom))
    model.add(layers.LeakyReLU(alpha=0.2))
    # classifier
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1), kernel_initializer=init)
    # compile model
    return model


def cifargan_ups_gen(args):
    g_dim = args.g_dim
    z_dim = args.noise_dim
    img_dim = args.dataset_dim[1]
    channels = args.dataset_dim[3]
    img_resize = img_dim//(2*2*2)

    model = keras.Sequential()
    # foundation for 4x4 image
    model.add(layers.Dense(g_dim * img_resize * img_resize, input_dim=z_dim, kernel_initializer=init))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((img_resize, img_resize, g_dim)))
    # upsample to 8x8
    model.add(layers.UpSampling2D(size=(2,2), interpolation='nearest'))
    model.add(layers.Conv2D(128, (4, 4), strides=(1, 1), padding='same', kernel_initializer=init))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(layers.UpSampling2D(size=(2,2), interpolation='nearest'))
    model.add(layers.Conv2D(128, (4, 4), strides=(1, 1), padding='same', kernel_initializer=init))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(layers.UpSampling2D(size=(2,2), interpolation='nearest'))
    model.add(layers.Conv2D(128, (4, 4), strides=(1, 1), padding='same', kernel_initializer=init))
    model.add(layers.LeakyReLU(alpha=0.2))
    # output layer
    model.add(layers.Conv2D(channels, (3, 3), activation='tanh', padding='same', kernel_initializer=init))
    return model


def cifargan_gen(args):
    g_dim = args.g_dim
    z_dim = args.noise_dim
    img_dim = args.dataset_dim[1]
    channels = args.dataset_dim[3]
    img_resize = img_dim//(2*2*2)

    model = keras.Sequential()
    # foundation for 4x4 image
    model.add(layers.Dense(g_dim * img_resize * img_resize, input_dim=z_dim, kernel_initializer=init))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((img_resize, img_resize, g_dim)))
    # upsample to 8x8
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(layers.LeakyReLU(alpha=0.2))
    # output layer
    if args.input_scale:
        print('tanh')
        model.add(layers.Conv2D(channels, (3, 3), activation='tanh', padding='same', kernel_initializer=init))
    else:
        print('sigmoid')
        model.add(layers.Conv2D(channels, (3, 3), activation='sigmoid', padding='same', kernel_initializer=init))
    return model


def cifargan_disc(args):
    d_dim = args.d_dim
    input_dim = args.dataset_dim[1]
    channels = args.dataset_dim[3]
    #const = ClipConstraint(0.01)
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


#64x64
def dcgan_gen(args):
    g_dim = args.g_dim
    z_dim = args.noise_dim
    img_dim = args.dataset_dim[1]
    channels = args.dataset_dim[3]
    img_resize = img_dim//(2*2*2*2)

    model = keras.Sequential()
    # foundation for 4x4 image
    model.add(layers.Dense(g_dim * img_resize * img_resize, input_dim=z_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((img_resize, img_resize, g_dim)))
    # upsample to 8x8
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    # upsample to 16x16
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # output layer
    model.add(layers.Conv2DTranspose(channels, (3, 3), activation='tanh', padding='same'))
    return model


def dcgan_disc(args):
    d_dim = args.d_dim
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

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # classifier
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1))
    # compile model
    return model


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