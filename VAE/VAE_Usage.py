import os
from VAE_Model import VAE as VAE_model
import argparse
import DataHandler as datahandler

def VAE_MNIST(RUN_ID, RUN_FOLDER, lr, r_loss, batch_size, epochs, print_n_batches, init_epoch, z_dim):
    # run params
    SECTION = 'vae'
    DATA_NAME = 'MNIST'
    RUN_FOLDER += SECTION + '/'
    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
    RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
        os.mkdir(os.path.join(RUN_FOLDER, 'images'))
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
    else:
        raise Exception('Run folder with id:' + RUN_ID + ' already found. Please choose new ID')

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
        z_dim=z_dim
    )

    if mode == 'build':
        VAE.save(RUN_FOLDER)
    else:
        VAE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

    VAE.encoder.summary()
    VAE.decoder.summary()

    VAE.compile(learning_rate=lr, r_loss_factor=r_loss)

    (x_train, x_test) = datahandler.mnist(norm_setting=1)

    VAE.train(
        x_train,
        batch_size=batch_size,
        epochs=epochs,
        run_folder=RUN_FOLDER,
        print_n_batches=print_n_batches,
        init_epoch=init_epoch
    )

def VAE_CIFAR(RUN_ID, RUN_FOLDER, lr, r_loss, batch_size, epochs, print_n_batches, init_epoch, z_dim):
    # run params
    SECTION = 'vae'
    DATA_NAME = 'CIFAR10'
    RUN_FOLDER += SECTION + '/'
    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
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
        z_dim=z_dim
    )

    if mode == 'build':
        VAE.save(RUN_FOLDER)
    else:
        VAE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

    VAE.encoder.summary()
    VAE.decoder.summary()

    VAE.compile(learning_rate=lr, r_loss_factor=r_loss)

    (x_train, x_test) = datahandler.cifar10(norm_setting=1)

    VAE.train(
        x_train,
        batch_size=batch_size,
        epochs=epochs,
        run_folder=RUN_FOLDER,
        print_n_batches=print_n_batches,
        init_epoch=init_epoch
    )




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='Can be mnist|cifar10')
    parser.add_argument('--lr', type=float, default='1e-4', help='Learning rate')
    parser.add_argument('--r_loss_factor', type=float, default=10000, help='dunno')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--print_n_batches', type=int, default=None, help='Prints status every n\'th batch. Default is None which is once per epoch')
    parser.add_argument('--init_epoch', type=int, default=0, help='Determine start epoch')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--run_id', type=str, help='ID of current run')
    parser.add_argument('--run_folder', type=str, help='folder that contains run generated items (images, weights etc.)')
    parser.add_argument('--z_dim', type=int, default=50, help='Size of latent dimensions for encoding')

    args = parser.parse_args()

    print(args)

    if args.dataset == 'mnist':
        if args.print_n_batches is None:
            args.print_n_batches = 60000 // args.batch_size
        VAE_MNIST(args.run_id, args.run_folder, args.lr, args.r_loss_factor, args.batch_size, args.epochs, args.print_n_batches, args.init_epoch, args.z_dim)
    elif args.dataset == 'cifar10':
        if args.print_n_batches is None:
            args.print_n_batches = 50000 // args.batch_size
        VAE_CIFAR(args.run_id, args.run_folder, args.lr, args.r_loss_factor, args.batch_size, args.epochs, args.print_n_batches, args.init_epoch, args.z_dim)
    else:
        raise('Dataset not supported')


