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
parser.add_argument('--dataset',        type=str,           default='toy',      help=' toy | mnist | cifar10 | lsun | frey')
parser.add_argument('--loss',           type=str,           default='ce',       help='wgan-gp | wgan | ce')
parser.add_argument('--batch_size',     type=int,           default=100)
parser.add_argument('--epochs',         type=int,           default=500)
parser.add_argument('--disc_iters',     type=int,           default=1)
parser.add_argument('--clip',           type=float,         default=0.01,       help='upper bound for clipping')
parser.add_argument('--gp_lambda',      type=int,           default=10)
parser.add_argument('--lr_d',           type=float,         default=1e-4)
parser.add_argument('--lr_g',           type=float,         default=1e-4)
parser.add_argument('--b1',             type=float,         default=0.9)
parser.add_argument('--b2',             type=float,         default=0.999)
parser.add_argument('--optim_d',        type=str,           default='adam',     help='adam | sgd | rms')
parser.add_argument('--optim_g',        type=str,           default='adam',     help='adam | rms')
parser.add_argument('--num_samples_to_gen', type=int,       default=64)
parser.add_argument('--images_while_training', type=int,    default=50,         help='Every x epoch to print images while training')
parser.add_argument('--dir',            type=str,           default='/user/student.aau.dk/mjuuln15/output_data'            , help='Directory to save images, models, weights etc')
parser.add_argument('--g_dim',          type=int,           default=256,        help='generator layer dimensions')
parser.add_argument('--d_dim',          type=int,           default=64,         help='discriminator layer dimensions')
parser.add_argument('--gan_type',       type=str,           default='dcgan',    help='dcgan | infogan | tfgan | cifargan_u')
parser.add_argument('--c_cat',          type=int,           default=10,         help='Amount of control variables for infogan')
parser.add_argument('--c_cont',         type=int,           default=2,          help = 'Amount of continuous control variables for infogan')
parser.add_argument('--img_dim',        type=int,           default=32,         help='Dataset image dimension')
parser.add_argument('--noise_dim',      type=int,           default=10,         help='size of the latent vector')
parser.add_argument('--limit_dataset',  type=bool,          default=False,      help='True to limit mnist/cifar dataset to one class')
parser.add_argument('--scale_data',     type=int,           default=0,          help='Scale images in dataset to MxM')
parser.add_argument('--label_flipping', type=bool,          default=True,       help='Flip 5% of labels during training of disc')
parser.add_argument('--label_smooth',   type=bool,          default=True,       help='Smooth the labels of the disc from 1 to 0 occasionally')
parser.add_argument('--input_noise',    type=bool,          default=True,       help='Add gaussian noise to the discriminator inputs')

args = parser.parse_args()

# Debugging
#args.dataset = 'cifar10'
#args.scale_data = 64
#args.batch_size = 2
#args.noise_dim = 100
#args.epochs = 10
#args.disc_iters = 5
#args.gan_type='dcgan'
#args.loss='ce'
#args.images_while_training = 1
#args.limit_dataset = True
#args.dir = 'C:/Users/marku/Desktop'
#o2i.load_images('C:/Users/marku/Desktop/GAN_training_output')
#o2i.test_trunc_trick(args)

# We will reuse this seed overtime for visualization
args.seed = tf.random.normal([args.num_samples_to_gen, args.noise_dim])

# Set random seeds for reproducability
tf.random.set_seed(2019)
np.random.seed(2019)

# Choose data
train_dat, shape = dt.select_dataset(args)
args.dataset_dim = shape

# GEN optimiser
if args.optim_g == "adam":
    args.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_d, beta_1=args.b1, beta_2=args.b2)
elif args.optim_g == "rms":
    args.gen_optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.lr_d)
else:
    raise NotImplementedError()

# DISC optimiser
if args.optim_d == "adam":
    args.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_g, beta_1=args.b1, beta_2=args.b2)
elif args.optim_d == "rms":
    args.disc_optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.lr_g)
elif args.optim_d == "sgd":
    args.disc_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
else:
    raise NotImplementedError()

# Choose model
generator, discriminator, auxiliary = u.select_models(args)

discriminator.summary()
generator.summary()



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





