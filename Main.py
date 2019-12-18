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
parser.add_argument('--dataset',        type=str,         default='toy',    help=' toy | mnist | cifar10 | lsun | frey')
parser.add_argument('--loss',           type=str,         default='ce',     help='wgan-gp | wgan | ce')
parser.add_argument('--batch_size',     type=int,         default=100)
parser.add_argument('--epochs',         type=int,         default=500)
parser.add_argument('--disc_iters',     type=int,         default=1)
parser.add_argument('--clip',           type=float,       default=0.01,     help='upper bound for clipping')
parser.add_argument('--gp_lambda',      type=int,         default=10)
parser.add_argument('--lr_d',           type=float,       default=1e-4)
parser.add_argument('--lr_g',           type=float,       default=1e-4)
parser.add_argument('--optim_d',        type=str,         default='adam',   help='adam | sgd')
parser.add_argument('--optim_g',        type=str,         default='adam',   help='adam')
parser.add_argument('--num_samples_to_gen', type=int,     default=16)
parser.add_argument('--images_while_training', type=int,  default=50,       help='Every x epoch to print images while training')
parser.add_argument('--dir',            type=str,         default='/user/student.aau.dk/mjuuln15/output_data'            , help='Directory to save images, models, weights etc')
parser.add_argument('--g_dim',          type=int,         default=256,       help='generator layer dimensions')
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
#args.dataset = 'lsun'
#args.scale_data = 32
#args.noise_dim = 100
#args.epochs = 10
#args.disc_iters = 5
#args.gan_type='cifargan'
#args.loss='wgan-gp'
#args.images_while_training = 1
#args.limit_dataset = True
#args.dir = 'C:/Users/marku/Desktop'
#o2i.load_images('C:/Users/marku/Desktop/GAN_training_output')
#o2i.test_trunc_trick(args)

# We will reuse this seed overtime for visualization
args.seed = tf.random.normal([args.num_samples_to_gen, args.noise_dim])

# Set random seed for reproducability
tf.random.set_seed(2019)

# Choose data
train_dat, shape = dt.select_dataset(args)
args.dataset_dim = shape

# Choose optimizers
if args.optim_d == "adam":
    args.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_d, beta_1=0.5)
elif args.optim_d == "sgd":
    args.gen_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
if args.optim_g == "adam":
    args.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_g, beta_1=0.5)
else:
    raise NotImplementedError()

# Choose model
generator, discriminator, auxiliary = u.select_models(args)

# Write config
u.write_config(args)

# Start training
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    with tf.device('/GPU:0'):
        print('Using GPU')
        ganTrainer = t.GANTrainer(generator, discriminator, auxiliary, train_dat)
        g_loss, d_loss, images_while_training, full_training_time, acc_fakes, acc_reals = ganTrainer.train(args)
else:
    print('Using CPU')
    ganTrainer = t.GANTrainer(generator, discriminator, auxiliary, train_dat)
    g_loss, d_loss, images_while_training, full_training_time, acc_fakes, acc_reals = ganTrainer.train(args)

# Write losses, image values, full training time and save models
np.save(os.path.join(args.dir, 'g_loss'), g_loss)
np.save(os.path.join(args.dir, 'd_loss'), d_loss)
np.save(os.path.join(args.dir, 'itw'), images_while_training)
np.save(os.path.join(args.dir, 'acc_fakes'), acc_fakes)
np.save(os.path.join(args.dir, 'acc_reals'), acc_reals)

generator._name='gen'
discriminator._name='disc'

with open(os.path.join(args.dir, 'config.txt'), 'a') as file:
    file.write('\nFull training time: '+str(full_training_time)+'\n')
    generator.summary(print_fn=lambda x: file.write(x + '\n'))
    discriminator.summary(print_fn=lambda x: file.write(x + '\n'))

generator.save(args.dir+'/generator')
discriminator.save(args.dir+'/discriminator')





