import tensorflow as tf
import numpy as np
import Nets as nets
import Data as dt
import Train as t
import argparse
import os.path
#import o2img as o2i

parser = argparse.ArgumentParser()

# Settings
parser.add_argument('--dataset', type=str,            default = 'toy'       , help=' toy | mnist | cifar10 ')
parser.add_argument('--img_dim', type=int,            default = 32          , help='Dataset image dimension')
parser.add_argument('--n_train', type=int,            default = 60000       , help='training set size, default to mnist')
parser.add_argument('--n_test', type=int,             default = 10000       , help='test set size, default to mnist')
parser.add_argument('--noise_dim', type=int,          default = 10         , help='size of the latent vector')
parser.add_argument('--loss', type=str,               default = 'wgan-gp'   , help='wgan-gp | wgan | ce')
parser.add_argument('--batch_size', type=int,         default = 100)
parser.add_argument('--epochs', type=int,             default = 500)
parser.add_argument('--n_critic', type=int,           default = 5)
parser.add_argument('--clip', type=float,             default = 0.01        , help='upper bound for clipping')
parser.add_argument('--gp_lambda', type=int,          default = 10)
parser.add_argument('--lr_d', type=float,             default = 1e-3)
parser.add_argument('--lr_g', type=float,             default = 1e-4)
parser.add_argument('--optim_d', type=str,            default = 'adam'      , help='adam')
parser.add_argument('--optim_g', type=str,            default = 'adam'      , help='adam')
parser.add_argument('--num_samples_to_gen', type=int, default = 16)
parser.add_argument('--images_while_training', type=int, default=50          , help='Every x epoch to print images while training')
parser.add_argument('--dir', type=str,                default='/user/student.aau.dk/mjuuln15/output_data'            , help='Directory to save images, models, weights etc')
parser.add_argument('--g_dim', type=int,              default=256           , help='generator layer dimensions')
parser.add_argument('--d_dim', type=int,              default=64           , help='discriminator layer dimensions')


args = parser.parse_args()

# local debugging
#args.dataset = 'cifar10'
#args.noise_dim = 100
#args.dir = 'C:/Users/marku/Desktop'
#args.epochs = 1
#o2i.load_images('C:/Users/marku/Desktop/GAN_training_output', 'mnist')


# Write config
file = open(os.path.join(args.dir, 'config.txt'), 'w')
file.write(str(args))
file.close()


# Choose correct model architecture
if args.dataset == "toy":
    dat = dt.createToyDataRing()
    train_dataset = tf.data.Dataset.from_tensor_slices(dat).shuffle(args.n_train).batch(args.batch_size)
    #o2i.plot_toy_distribution(dat, "2 dimensional toy data distribution")
    generator = nets.generator_toy(args.noise_dim)
    discriminator = nets.discriminator_toy()
elif args.dataset == "mnist":
    dat = dt.mnist()
    train_dataset = tf.data.Dataset.from_tensor_slices(dat).shuffle(args.n_train).batch(args.batch_size)
    generator = nets.generator_dcgan(args.img_dim, 1, args.g_dim, args.noise_dim)
    discriminator = nets.discriminator_dcgan(args.img_dim, 1, args.d_dim)
elif args.dataset == 'cifar10':
    dat = dt.cifar10()
    train_dataset = tf.data.Dataset.from_tensor_slices(dat).shuffle(args.n_train).batch(args.batch_size)
    generator = nets.generator_dcgan(args.img_dim, 3, args.g_dim, args.noise_dim)
    discriminator = nets.discriminator_dcgan(args.img_dim, 3, args.d_dim)
else:
    raise Exception("Dataset not available")

# Choose correct optimizers
if args.optim_d == "adam":
    args.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_d, beta_1=0.5)
if args.optim_g == "adam":
    args.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_g, beta_1=0.5)
else:
    raise Exception("Optimizer not available")

args.dataset_dim = dat.shape[0]

# We will reuse this seed overtime for visualization
args.seed = tf.random.normal([args.num_samples_to_gen, args.noise_dim])

# Start training
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    with tf.device('/GPU:0'):
        g_loss, d_loss, images_while_training, full_training_time = t.train(train_dataset, discriminator, generator, args)
else:
    g_loss, d_loss, images_while_training, full_training_time = t.train(train_dataset, discriminator, generator, args)

# Write losses, image values, full training time and save models

np.save(os.path.join(args.dir, 'g_loss'), g_loss)
np.save(os.path.join(args.dir, 'd_loss'), d_loss)
np.save(os.path.join(args.dir, 'itw'), images_while_training)

file = open(os.path.join(args.dir, 'config.txt'), 'a')
file.write(str(full_training_time))
file.close()

generator.save(args.dir+'/generator')
discriminator.save(args.dir+'/discriminator')





