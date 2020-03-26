import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
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
    model.add(layers.Conv2DTranspose(1024, (4, 4), strides=(1, 1), padding='same', kernel_initializer=init))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
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
def cogan_generators_conv(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*4*4)(noise)
    model = tf.keras.layers.Reshape((4, 4, 1024))(model)

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

    # Generator 1
    img1 = tf.keras.layers.Conv2DTranspose(channels, (6,6), strides=(1, 1), activation='tanh', padding='same')(model)

    # Generator 2
    img2 = tf.keras.layers.Conv2DTranspose(channels, (6,6), strides=(1, 1), activation='tanh', padding='same')(model)

    return keras.Model(noise, img1), keras.Model(noise, img2)


def cogan_generators_fc(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Shared weights between generators
    model = keras.Sequential()
    model.add(tf.keras.layers.Dense(256, input_dim=args.noise_dim))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    noise = tf.keras.layers.Input(shape=(args.noise_dim,))
    feature_repr = model(noise)

    # Generator 1
    g1 = tf.keras.layers.Dense(1024)(feature_repr)
    g1 = tf.keras.layers.LeakyReLU(alpha=0.2)(g1)
    g1 = tf.keras.layers.BatchNormalization(momentum=0.8)(g1)
    g1 = tf.keras.layers.Dense(np.prod(img_shape), activation='tanh')(g1)
    img1 = tf.keras.layers.Reshape(img_shape)(g1)

    # Generator 2
    g2 = tf.keras.layers.Dense(1024)(feature_repr)
    g2 = tf.keras.layers.LeakyReLU(alpha=0.2)(g2)
    g2 = tf.keras.layers.BatchNormalization(momentum=0.8)(g2)
    g2 = tf.keras.layers.Dense(np.prod(img_shape), activation='tanh')(g2)
    img2 = tf.keras.layers.Reshape(img_shape)(g2)

    return keras.Model(noise, img1), keras.Model(noise, img2)


def cogan_discriminators_conv(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    #x1 = tf.keras.layers.Conv2D(20, (5, 5), padding='same')(img1)
    #x1 = tf.keras.layers.MaxPool2D()(x1)

    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    #x2 = tf.keras.layers.Conv2D(20, (5, 5), padding='same')(img2)
    #x2 = tf.keras.layers.MaxPool2D()(x2)

    # Shared discriminator layers
    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(20, (5, 5), padding='same'))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(50, (5, 5), padding='same'))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(500))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    #model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    #output1 = model(x1)
    #output2 = model(x2)
    img1_embedding = model(img1)
    img2_embedding = model(img2)

    # Discriminator 1
    output1 = tf.keras.layers.Dense(1, activation='sigmoid')(img1_embedding)
    # Discriminator 2
    output2 = tf.keras.layers.Dense(1, activation='sigmoid')(img2_embedding)

    return keras.Model(img1, output1), keras.Model(img2, output2)


def cogan_discriminators_fc(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    img1 = tf.keras.layers.Input(shape=img_shape)
    img2 = tf.keras.layers.Input(shape=img_shape)

    # Shared discriminator layers
    model = keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=img_shape))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    img1_embedding = model(img1)
    img2_embedding = model(img2)

    # Discriminator 1
    validity1 = tf.keras.layers.Dense(1, activation='sigmoid')(img1_embedding)
    # Discriminator 2
    validity2 = tf.keras.layers.Dense(1, activation='sigmoid')(img2_embedding)

    return keras.Model(img1, validity1), keras.Model(img2, validity2)


'''
******************************************
--------------- CROSS GAN ----------------
******************************************
'''
def generator_init_block(input, out_channels):
    input = tf.keras.layers.Dense(out_channels * 4 * 4)(input)
    input = tf.keras.layers.Reshape((4, 4, out_channels))(input)
    input = (tf.keras.layers.Conv2DTranspose(out_channels, (4, 4), strides=(1, 1), padding='same'))(input)
    input = (tf.keras.layers.BatchNormalization(momentum=0.8))(input)
    input = (tf.keras.layers.LeakyReLU(alpha=0.2))(input)
    return input

def generator_block(input, out_channels):
    input = (tf.keras.layers.Conv2DTranspose(out_channels, (3, 3), strides=(2, 2), padding='same'))(input)
    input = (tf.keras.layers.BatchNormalization(momentum=0.8))(input)
    input = (tf.keras.layers.LeakyReLU(alpha=0.2))(input)
    return input

def rgb_converter_conv(input, out_channels):
    return tf.keras.layers.Conv2D(out_channels, (1,1))(input)


def cross_cogan_generators(args):
    img_channels = args.dataset_dim[3]

    # output containers for images of all scales
    outputs_model1 = []
    outputs_model2 = []

    # input noise vector shared by both generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    # Initial conv block to produce many filters for convolutions and 4x4 image
    model1 = generator_init_block(noise, args.g_filters[0])
    model2 = generator_init_block(noise, args.g_filters[0])

    # Use to_rgb_conv to convert 1024 filter outputs to 3 filter outputs as rgb 4x4 images
    outputs_model1.append(rgb_converter_conv(model1,img_channels)) # 4x4 image for model1
    outputs_model2.append(rgb_converter_conv(model2,img_channels)) # 4x4 image for model2

    # Create dynamic model according to calculated depth
    for i in range(1,args.depth):                                 # 1 to args.depth since we have done the first layer above
        model1 = generator_block(model1, args.g_filters[i])
        model2 = generator_block(model2, args.g_filters[i])
        if args.cross_depth > i:
            outputs_model1.append(rgb_converter_conv(model1,img_channels))
            outputs_model2.append(rgb_converter_conv(model2,img_channels))

    # TODO: KIG LIGE PÅ OM DETTE SKAL VÆRE SIDSTE LAG I MODELLEN
    # Generator 1
    outputs_model1.append(tf.keras.layers.Conv2DTranspose(img_channels, (6, 6), strides=(1, 1), activation='tanh', padding='same')(model1))

    # Generator 2
    outputs_model2.append(tf.keras.layers.Conv2DTranspose(img_channels, (6, 6), strides=(1, 1), activation='tanh', padding='same')(model2))

    return tf.keras.Model(inputs=noise, outputs=outputs_model1), keras.Model(inputs=noise, outputs=outputs_model2)

def discriminator_block(main_input, scale_input, out_channels):
    main_input = tf.keras.layers.Conv2D(out_channels, (5, 5), padding='same')(main_input)
    main_input = tf.keras.layers.MaxPool2D()(main_input)
    if scale_input is not None: # Only apply concatination layer if scale input is submitted
        main_input = tf.keras.layers.Concatenate()([main_input, rgb_converter_conv(scale_input, out_channels=out_channels)])
    return main_input

def cross_cogan_discriminators(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # First input layers corresponding to full image shape
    model1_inputlayers = [tf.keras.layers.Input(shape=img_shape)]
    model2_inputlayers = [tf.keras.layers.Input(shape=img_shape)]

    nr_skip_cross = args.depth - args.cross_depth
    new_input_size = img_shape[0] / (math.pow(2,nr_skip_cross))     # calculate shape of first skipped inputlayer
    #  all inputs layers for cross resolutions
    for i in range(0,args.cross_depth):
        input_shape = (new_input_size, new_input_size, img_shape[2])
        model1_inputlayers.append(tf.keras.layers.Input(shape=input_shape))
        model2_inputlayers.append(tf.keras.layers.Input(shape=input_shape))
        new_input_size = new_input_size / 2     # Assume that all subsequent layers are inputlayers

    #create model 1
    model1 = model1_inputlayers[0]
    for i in range(1,args.depth):
        if nr_skip_cross <= i:  #If we have skipped all the depth connection uptil our wanted cross connection
            model1 = discriminator_block(model1, model2_inputlayers[i-nr_skip_cross+1], args.d_filters[i-1])
        else:   # Otherwise we do not add the cross connection
            model1 = discriminator_block(model1, None, args.d_filters[i-1])

    model1 = tf.keras.layers.Flatten()(model1)
    model1 = tf.keras.layers.Dense(args.d_filters[-1])(model1)
    model1 = tf.keras.layers.LeakyReLU(alpha=0.2)(model1)

    #create model 2
    model2 = model2_inputlayers[0]
    for i in range(1, args.depth):
        if nr_skip_cross <= i:
            model2 = discriminator_block(model2, model1_inputlayers[i-nr_skip_cross+1], args.d_filters[i-1])
        else:
            model2 = discriminator_block(model2, None, args.d_filters[i-1])
    model2 = tf.keras.layers.Flatten()(model2)
    model2 = tf.keras.layers.Dense(args.d_filters[-1])(model2)
    model2 = tf.keras.layers.LeakyReLU(alpha=0.2)(model2)


    output1 = tf.keras.layers.Dense(1, activation='sigmoid')(model1)
    output2 = tf.keras.layers.Dense(1, activation='sigmoid')(model2)

    #prepare cross-input to discriminator
    input1 = model1_inputlayers[:1]
    input2 = model2_inputlayers[:1]
    input1.extend(model2_inputlayers[1:])
    input2.extend(model1_inputlayers[1:])

    return keras.Model(inputs=input1, outputs=output1), keras.Model(inputs=input2, outputs=output2)


'''
******************************************
------------- CROSS MSG GAN --------------
******************************************
'''
def cross_msg_cogan_generators(args):
    img_channels = args.dataset_dim[3]

    # output containers for images of all scales
    outputs_model1 = []
    outputs_model2 = []

    # input noise vector shared by both generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    # Initial conv block to produce many filters for convolutions and 4x4 image
    model1 = generator_init_block(noise, args.g_filters[0])
    model2 = generator_init_block(noise, args.g_filters[0])

    # Use to_rgb_conv to convert 1024 filter outputs to 3 filter outputs as rgb 4x4 images
    outputs_model1.append(rgb_converter_conv(model1,img_channels)) # 4x4 image for model1
    outputs_model2.append(rgb_converter_conv(model2,img_channels)) # 4x4 image for model2

    # Create dynamic model according to calculated depth
    for i in range(1,args.depth):                                 # 1 to args.depth since we have done the first layer above
        model1 = generator_block(model1, args.g_filters[i])
        model2 = generator_block(model2, args.g_filters[i])
        if args.cross_depth > i:
            outputs_model1.append(rgb_converter_conv(model1,img_channels))
            outputs_model2.append(rgb_converter_conv(model2,img_channels))

    # TODO: KIG LIGE PÅ OM DETTE SKAL VÆRE SIDSTE LAG I MODELLEN
    # Generator 1
    outputs_model1.append(tf.keras.layers.Conv2DTranspose(img_channels, (6, 6), strides=(1, 1), activation='tanh', padding='same')(model1))

    # Generator 2
    outputs_model2.append(tf.keras.layers.Conv2DTranspose(img_channels, (6, 6), strides=(1, 1), activation='tanh', padding='same')(model2))

    return tf.keras.Model(inputs=noise, outputs=outputs_model1), keras.Model(inputs=noise, outputs=outputs_model2)

def cross_msg_cogan_discriminators(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # First input layers corresponding to full image shape
    model1_inputlayers = [tf.keras.layers.Input(shape=img_shape)]
    model2_inputlayers = [tf.keras.layers.Input(shape=img_shape)]

    nr_skip_cross = args.depth - args.cross_depth
    new_input_size = img_shape[0] / (math.pow(2,nr_skip_cross))     # calculate shape of first skipped inputlayer
    #  all inputs layers for cross resolutions
    for i in range(0,args.cross_depth):
        input_shape = (new_input_size, new_input_size, img_shape[2])
        model1_inputlayers.append(tf.keras.layers.Input(shape=input_shape))
        model2_inputlayers.append(tf.keras.layers.Input(shape=input_shape))
        new_input_size = new_input_size / 2     # Assume that all subsequent layers are inputlayers

    #create model 1
    model1 = model1_inputlayers[0]
    for i in range(1,args.depth):
        if nr_skip_cross <= i:  #If we have skipped all the depth connection uptil our wanted cross connection
            model1 = discriminator_block(model1, model2_inputlayers[i-nr_skip_cross+1], args.d_filters[i-1])
        else:   # Otherwise we do not add the cross connection
            model1 = discriminator_block(model1, None, args.d_filters[i-1])

    model1 = tf.keras.layers.Flatten()(model1)
    model1 = tf.keras.layers.Dense(args.d_filters[-1])(model1)
    model1 = tf.keras.layers.LeakyReLU(alpha=0.2)(model1)

    #create model 2
    model2 = model2_inputlayers[0]
    for i in range(1, args.depth):
        if nr_skip_cross <= i:
            model2 = discriminator_block(model2, model1_inputlayers[i-nr_skip_cross+1], args.d_filters[i-1])
        else:
            model2 = discriminator_block(model2, None, args.d_filters[i-1])
    model2 = tf.keras.layers.Flatten()(model2)
    model2 = tf.keras.layers.Dense(args.d_filters[-1])(model2)
    model2 = tf.keras.layers.LeakyReLU(alpha=0.2)(model2)


    output1 = tf.keras.layers.Dense(1, activation='sigmoid')(model1)
    output2 = tf.keras.layers.Dense(1, activation='sigmoid')(model2)

    #prepare cross-input to discriminator
    input1 = model1_inputlayers[:1]
    input2 = model2_inputlayers[:1]
    input1.extend(model2_inputlayers[1:])
    input2.extend(model1_inputlayers[1:])

    return keras.Model(inputs=input1, outputs=output1), keras.Model(inputs=input2, outputs=output2)