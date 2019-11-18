import os

from VAE_Model import VAE as VAE_model
from keras.datasets import mnist, cifar10
import numpy as np


def VAE_MNIST(RUN_ID, RUN_FOLDER):
    # run params
    SECTION = 'vae'
    DATA_NAME = 'MNIST'
    RUN_FOLDER += SECTION + '/'
    RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
        os.mkdir(os.path.join(RUN_FOLDER, 'images'))
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

    # 'load' or 'build'
    mode = 'build'

    VAE = VAE_model(
        input_dim=(28, 28, 1),
        encoder_conv_filters=[32, 64, 64, 64],
        encoder_conv_kernel_size=[3, 3, 3, 3],
        encoder_conv_strides=[1, 2, 2, 1],
        decoder_conv_t_filters=[64, 64, 32, 1],
        decoder_conv_t_kernel_size=[3, 3, 3, 3],
        decoder_conv_t_strides=[1, 2, 2, 1],
        z_dim=50
    )

    if mode == 'build':
        VAE.save(RUN_FOLDER)
    else:
        VAE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

    VAE.encoder.summary()
    VAE.decoder.summary()

    L_RATE = 0.0005
    R_LOSS_FACTOR = 1000

    VAE.compile(learning_rate=L_RATE, r_loss_factor=R_LOSS_FACTOR)

    BATCH_SIZE = 100
    EPOCHS = 50
    PRINT_N_BATCHES = 100
    INIT_EPOCH = 0

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))

    VAE.train(
        x_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        run_folder=RUN_FOLDER,
        print_n_batches=PRINT_N_BATCHES,
        init_epoch=INIT_EPOCH
    )

def VAE_CIFAR(RUN_ID, RUN_FOLDER):
    # run params
    SECTION = 'vae'
    DATA_NAME = 'CIFAR10'
    RUN_FOLDER += SECTION + '/'
    RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
        os.mkdir(os.path.join(RUN_FOLDER, 'images'))
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

    # 'load' or 'build'
    mode = 'build'

    VAE = VAE_model(
        input_dim=(32, 32, 3),
        encoder_conv_filters=[32, 64, 64, 64],
        encoder_conv_kernel_size=[3, 3, 3, 3],
        encoder_conv_strides=[2, 2, 2, 2],
        decoder_conv_t_filters=[64, 64, 32, 3],
        decoder_conv_t_kernel_size=[3, 3, 3, 3],
        decoder_conv_t_strides=[2, 2, 2, 2],
        z_dim=50
    )

    if mode == 'build':
        VAE.save(RUN_FOLDER)
    else:
        VAE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

    VAE.encoder.summary()
    VAE.decoder.summary()

    L_RATE = 0.0005
    R_LOSS_FACTOR = 10000

    VAE.compile(learning_rate=L_RATE, r_loss_factor=R_LOSS_FACTOR)

    BATCH_SIZE = 100
    EPOCHS = 50
    PRINT_N_BATCHES = 100
    INIT_EPOCH = 0
    label = 1

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # train_mask = [y[0]==label for y in y_train]
    # test_mask = [y[0]==label for y in y_test]

    # x_train = np.concatenate([x_train[train_mask], x_test[test_mask]])
    # y_train = np.concatenate([y_train[train_mask], y_test[test_mask]])

    x_train = (x_train.astype('float32') - 127.5) / 127.5

    VAE.train(
        x_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        run_folder=RUN_FOLDER,
        print_n_batches=PRINT_N_BATCHES,
        init_epoch=INIT_EPOCH
    )


