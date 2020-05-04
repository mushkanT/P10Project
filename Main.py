import tensorflow as tf
import numpy as np
import Nets as nets
import Data as dt
import GAN_trainer as gan_t
import CoGAN_trainer as cogan_t
import time
import Utils as u
import argparse
import os.path
#import o2img as o2i
#import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

# Settings
parser.add_argument('--dataset',        type=str,           default='toy',      help=' toy | mnist | cifar10 | lsun | frey | svhn')
parser.add_argument('--loss',           type=str,           default='ce',       help=' wgan | ce')
parser.add_argument('--disc_penalty',   type=str,           default='none',       help='none | wgan-gp')
parser.add_argument('--gen_penalty',    type=str,           default='none',       help='weight | feature')
parser.add_argument('--batch_size',     type=int,           default=128)
parser.add_argument('--epochs',         type=int,           default=5000)
parser.add_argument('--disc_iters',     type=int,           default=1)
parser.add_argument('--clip',           type=float,         default=0.01,       help='upper bound for clipping')
parser.add_argument('--penalty_weight_d',      type=int,           default=10)
parser.add_argument('--penalty_weight_g',      type=int,           default=10)
parser.add_argument('--lr_d',           type=float,         default=0.0002)
parser.add_argument('--lr_g',           type=float,         default=0.0002)
parser.add_argument('--b1',             type=float,         default=0.5)
parser.add_argument('--b2',             type=float,         default=0.999)
parser.add_argument('--optim_d',        type=str,           default='adam',     help='adam | sgd | rms')
parser.add_argument('--optim_g',        type=str,           default='adam',     help='adam | rms')
parser.add_argument('--num_samples_to_gen', type=int,       default=8)
parser.add_argument('--images_while_training', type=int,    default=1,         help='Every x epoch to print images while training')
parser.add_argument('--dir',            type=str,           default='/user/student.aau.dk/mjuuln15/output_data',     help='Directory to save images, models, weights etc')
parser.add_argument('--g_dim',          type=int,           default=256,        help='generator layer dimensions')
parser.add_argument('--d_dim',          type=int,           default=64,         help='discriminator layer dimensions')
parser.add_argument('--gan_type',       type=str,           default='cogan',    help='64 | 128 | cifargan | cogan')
parser.add_argument('--noise_dim',      type=int,           default=100,        help='size of the latent vector')
parser.add_argument('--limit_dataset',  type=bool,          default=False,      help='limit dataset to one class')
parser.add_argument('--scale_data',     type=int,           default=0,          help='Scale images in dataset to MxM')
parser.add_argument('--label_smooth',   type=bool,          default=False,      help='Smooth the labels of the disc from 1 to 0 occasionally')
parser.add_argument('--input_noise',    type=bool,          default=False,      help='Add gaussian noise to the discriminator inputs')
parser.add_argument('--input_scale',    type=bool,          default=True,       help='True=-1,1 False=0,1')
parser.add_argument('--purpose',        type=str,		    default='',		    help='purpose of this experiment')
parser.add_argument('--grayscale',      type=bool,		    default=False)

# CoGAN
parser.add_argument('--g_arch',         type=str,           default='digit',       help='digit | rotate | 256 | face | digit_noshare')
parser.add_argument('--d_arch',         type=str,           default='digit',       help='digit | rotate | 256 | face | digit_noshare')
parser.add_argument('--cogan_data',     type=str,           default='mnist2edge',  help='mnist2edge | mnist2rotate | mnist2svhn | mnist2negative | celeb_a | apple2orange | horse2zebra | vangogh2photo')

args = parser.parse_args()

# Debugging
#args.dataset = 'mnist'
#args.gan_type = "cogan"
#args.loss = 'ce'
#args.disc_penalty = 'wgan-gp'
#args.gen_penalty = 'weight'
#args.dir = 'C:/Users/marku/Desktop/gan_training_output/testing'
#args.g_arch = 'face'
#args.d_arch = 'face'
#args.cogan_data = 'Eyeglasses'
# args.batch_size = 64
#args.epochs = 2
# args.disc_iters = 5
# args.images_while_training = 10
# args.limit_dataset = True
# #args.scale_data = 64
#args.label_smooth=True


#o2i.load_images('C:/Users/marku/Desktop/GAN_training_output')
#o2i.test_trunc_trick(args)

# We will reuse this seed overtime for visualization
args.seed = tf.random.normal([args.num_samples_to_gen, args.noise_dim])
#args.seed = np.random.normal(0, 1, args.num_samples_to_gen, 100)

# Set random seeds for reproducability
tf.random.set_seed(2020)
np.random.seed(2020)

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

# Choose gan type
if args.gan_type == 'cogan':
    # Choose data
    start = time.time()
    domain1, domain2, shape = dt.select_dataset_cogan(args)
    args.dataset_dim = shape
    data_load_time = time.time() - start

    # Select architectures
    generator1, generator2, discriminator1, discriminator2 = u.select_cogan_architecture(args)

    # Write config
    u.write_config(args)

    # Start training
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        with tf.device('/GPU:0'):
            print('Using GPU')
            ganTrainer = cogan_t.GANTrainer(generator1, generator2, discriminator1, discriminator2, domain1, domain2)
            full_training_time = ganTrainer.train(args)
    else:
        print('Using CPU')
        ganTrainer = cogan_t.GANTrainer(generator1, generator2, discriminator1, discriminator2, domain1, domain2)
        full_training_time = ganTrainer.train(args)

    generator1._name = 'gen1'
    discriminator1._name = 'disc1'
    generator2._name = 'gen2'
    discriminator2._name = 'disc2'

    with open(os.path.join(args.dir, 'config.txt'), 'a') as file:
        file.write('\nFull training time: ' + str(full_training_time) + '\nData load time: ' + str(data_load_time) + '\n')
        generator1.summary(print_fn=lambda x: file.write(x + '\n'))
        discriminator1.summary(print_fn=lambda x: file.write(x + '\n'))
        generator2.summary(print_fn=lambda x: file.write(x + '\n'))
        discriminator2.summary(print_fn=lambda x: file.write(x + '\n'))

    generator1.save(args.dir + '/generator1')
    discriminator1.save(args.dir + '/discriminator1')
    generator2.save(args.dir + '/generator2')
    discriminator2.save(args.dir + '/discriminator2')

else:
    # Choose data

    start = time.time()
    train_dat, shape = dt.select_dataset_gan(args)
    if shape is None:
        args.dataset_dim = train_dat.element_spec[0].shape
    else:
        args.dataset_dim = shape
    data_load_time = time.time() - start
    if args.input_noise:
        args.variance = 0.1

    # Select architectures
    generator, discriminator = u.select_gan_architecture(args)

    # Write config
    u.write_config(args)

    # Start training
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        with tf.device('/GPU:0'):
            print('Using GPU')
            ganTrainer = gan_t.GANTrainer(generator, discriminator, train_dat)
            full_training_time = ganTrainer.train(args)
    else:
        print('Using CPU')
        ganTrainer = gan_t.GANTrainer(generator, discriminator, train_dat)
        full_training_time = ganTrainer.train(args)

    generator._name='gen'
    discriminator._name='disc'

    with open(os.path.join(args.dir, 'config.txt'), 'a') as file:
        file.write('\nFull training time: '+str(full_training_time)+'\nData load time: '+str(data_load_time))
        generator.summary(print_fn=lambda x: file.write(x + '\n'))
        discriminator.summary(print_fn=lambda x: file.write(x + '\n'))

    generator.save(args.dir+'/generator')
    discriminator.save(args.dir+'/discriminator')





