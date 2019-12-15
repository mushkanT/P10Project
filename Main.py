import tensorflow as tf
import numpy as np
import Nets as nets
import Data as dt
import Train as t
import Utils as u
import argparse
import os.path
#import o2img as o2i
#import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

# Settings
parser.add_argument('--dataset',        type=str,         default='toy',    help=' toy | mnist | cifar10 | lsun')
parser.add_argument('--loss',           type=str,         default='ce',     help='wgan-gp | wgan | ce')
parser.add_argument('--batch_size',     type=int,         default=100)
parser.add_argument('--epochs',         type=int,         default=500)
parser.add_argument('--disc_iters',     type=int,         default=1)
parser.add_argument('--clip',           type=float,       default=0.01,     help='upper bound for clipping')
parser.add_argument('--gp_lambda',      type=int,         default=10)
parser.add_argument('--lr_d',           type=float,       default=1e-4)
parser.add_argument('--lr_g',           type=float,       default=1e-4)
parser.add_argument('--optim_d',        type=str,         default='adam',   help='adam')
parser.add_argument('--optim_g',        type=str,         default='adam',   help='adam')
parser.add_argument('--num_samples_to_gen', type=int,     default=16)
parser.add_argument('--images_while_training', type=int,  default=50,       help='Every x epoch to print images while training')
parser.add_argument('--dir',            type=str,         default='/user/student.aau.dk/mjuuln15/output_data'            , help='Directory to save images, models, weights etc')
parser.add_argument('--g_dim',          type=int,         default=64,       help='generator layer dimensions')
parser.add_argument('--d_dim',          type=int,         default=64,       help='discriminator layer dimensions')
parser.add_argument('--gan_type',       type=str,         default='dcgan',  help='dcgan | infogan | tfgan | presgan')
parser.add_argument('--c_cat',          type=int,         default=10,       help='Amount of control variables for infogan')
parser.add_argument('--c_cont',         type=int,         default=2,        help = 'Amount of continuous control variables for infogan')
parser.add_argument('--img_dim',        type=int,         default=32,       help='Dataset image dimension')
parser.add_argument('--noise_dim',      type=int,         default=10,       help='size of the latent vector')
parser.add_argument('--limit_dataset',  type=bool,        default=False,    help='True to limit mnist/cifar dataset to one class')
parser.add_argument('--scale_data',     type=int,         default=0,        help='Scale images in dataset to MxM')

args = parser.parse_args()

# Debugging
#args.dataset = 'cifar10'
#args.scale_data = 64
#args.noise_dim = 100
#args.epochs = 250
#args.disc_iters = 5
#args.gan_type='dcgan'
#args.loss='wgan-gp'
#args.images_while_training = 5
#args.limit_dataset = True
#args.dir = 'C:/Users/marku/Desktop'
#o2i.load_images('C:/Users/marku/Desktop/GAN_training_output')
#o2i.test_trunc_trick(args)

# We will reuse this seed overtime for visualization
args.seed = tf.random.normal([args.num_samples_to_gen, args.noise_dim])
# Set random seed for reproducability
tf.random.set_seed(2019)

# Choose data
if args.dataset == "toy":
    dat = dt.createToyDataRing()
    #o2i.plot_toy_distribution(dat)
    train_dat = tf.data.Dataset.from_tensor_slices(dat).shuffle(dat.shape[0]).batch(args.batch_size).repeat()
elif args.dataset == "mnist":
    dat = dt.mnist(args.limit_dataset)
    if args.scale_data != 0:
        dat = tf.image.resize(dat, [args.scale_data, args.scale_data])
    train_dat = tf.data.Dataset.from_tensor_slices(dat).shuffle(dat.shape[0]).batch(args.batch_size).repeat()
elif args.dataset == 'cifar10':
    dat = dt.cifar10(args.limit_dataset)
    if args.scale_data != 0:
        dat = tf.image.resize(dat, [args.scale_data, args.scale_data])
    train_dat = tf.data.Dataset.from_tensor_slices(dat).shuffle(dat.shape[0]).batch(args.batch_size).repeat()
elif args.dataset == 'lsun':
    dat = dt.lsun(args.limit_dataset)
    dat = dat.shuffle(100000).batch(args.batch_size)
    if args.scale_data != 0:
        dat = dat.map(lambda x: tf.image.resize(x['image'], [args.scale_data, args.scale_data]))
    train_dat = dat.repeat()
else:
    raise NotImplementedError()

# Poor temporary fix
args.dataset_dim = dat.shape if not args.dataset == 'lsun' else [3000000, 64, 64, 3]

# Choose optimizers
if args.optim_d == "adam":
    args.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_d, beta_1=0.5)
if args.optim_g == "adam":
    args.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_g, beta_1=0.5)
else:
    raise NotImplementedError()

# Choose model
if args.dataset == 'toy':
    generator = nets.toy_gen(args.noise_dim)
    discriminator = nets.toy_disc(args)
    auxiliary = None
elif args.gan_type == 'infogan':
    generator = nets.infogan_gen(args)
    discriminator, auxiliary = nets.infogan_disc(args)
elif args.gan_type == 'tfgan':
    generator = nets.tfgan_gen(args)
    discriminator = nets.tfgan_disc(args)
    auxiliary = None
elif args.gan_type == 'dcgan':
    generator = nets.dcgan_gen(args)
    discriminator = nets.dcgan_disc(args)
    auxiliary = None
else:
    raise NotImplementedError()

# Write config
u.write_config(args)

# Start training
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    with tf.device('/GPU:0'):
        print('Using GPU')
        ganTrainer = t.GANTrainer(generator, discriminator, auxiliary, train_dat)
        g_loss, d_loss, images_while_training, full_training_time = ganTrainer.train(args)
else:
    print('Using CPU')
    ganTrainer = t.GANTrainer(generator, discriminator, auxiliary, train_dat)
    g_loss, d_loss, images_while_training, full_training_time = ganTrainer.train(args)

# Write losses, image values, full training time and save models
np.save(os.path.join(args.dir, 'g_loss'), g_loss)
np.save(os.path.join(args.dir, 'd_loss'), d_loss)
np.save(os.path.join(args.dir, 'itw'), images_while_training)

file = open(os.path.join(args.dir, 'config.txt'), 'a')
file.write(str(full_training_time))
file.close()

generator.save(args.dir+'/generator')
discriminator.save(args.dir+'/discriminator')





