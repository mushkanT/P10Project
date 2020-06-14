import tensorflow as tf
import numpy as np
import Data as dt
import GAN_trainer as gan_t
import CoGAN_trainer as cogan_t
import time
import Utils as u
import argparse
import os.path
#import seaborn as sns
#import scipy
#import matplotlib.pyplot as plt
#import PIL


parser = argparse.ArgumentParser()

# Settings
parser.add_argument('--dataset',               type=str,    default='toy',      help=' toy | mnist | cifar10 | lsun | frey | svhn')
parser.add_argument('--loss',                  type=str,    default='ce',       help=' wgan | ce')
parser.add_argument('--disc_penalty',          type=str,    default='none',     help='none | wgan-gp')
parser.add_argument('--gen_penalty',           type=str,    default='none',     help='weight | feature')
parser.add_argument('--batch_size',            type=int,    default=128)
parser.add_argument('--epochs',                type=int,    default=5000)
parser.add_argument('--disc_iters',            type=int,    default=1)
parser.add_argument('--clip',                  type=float,  default=0.01,       help='upper bound for clipping')
parser.add_argument('--penalty_weight_d',      type=int,    default=10)
parser.add_argument('--penalty_weight_g',      type=int,    default=10)
parser.add_argument('--lr_d',                  type=float,  default=0.0002)
parser.add_argument('--lr_g',                  type=float,  default=0.0002)
parser.add_argument('--b1',                    type=float,  default=0.5)
parser.add_argument('--b2',                    type=float,  default=0.999)
parser.add_argument('--optim_d',               type=str,    default='adam',     help='adam | sgd | rms')
parser.add_argument('--optim_g',               type=str,    default='adam',     help='adam | rms')
parser.add_argument('--num_samples_to_gen',    type=int,    default=8)
parser.add_argument('--images_while_training', type=int,    default=1,          help='Every x epoch to print images while training')
parser.add_argument('--dir',                   type=str,    default='/user/student.aau.dk/mjuuln15/output_data',     help='Directory to save images, models, weights etc')
parser.add_argument('--g_dim',                 type=int,    default=256,        help='generator layer dimensions')
parser.add_argument('--d_dim',                 type=int,    default=64,         help='discriminator layer dimensions')
parser.add_argument('--gan_type',              type=str,    default='cogan',    help='64 | 128 | 32 | cogan')
parser.add_argument('--noise_dim',             type=int,    default=100,        help='size of the latent vector')
parser.add_argument('--limit_dataset',         type=bool,   default=False,      help='limit dataset to one class')
parser.add_argument('--scale_data',            type=int,    default=0,          help='Scale images in dataset to MxM')
parser.add_argument('--label_smooth',          type=bool,   default=False,      help='Smooth the labels of the disc from 1 to 0 occasionally')
parser.add_argument('--input_noise',           type=bool,   default=False,      help='Add gaussian noise to the discriminator inputs')
parser.add_argument('--purpose',               type=str,    default='',		    help='purpose of this experiment')
parser.add_argument('--grayscale',             type=bool,   default=False)
parser.add_argument('--weight_decay',          type=float,  default=0.0001)
parser.add_argument('--bias_init',             type=float,  default=0)
parser.add_argument('--prelu_init',            type=float,  default=0.25)
parser.add_argument('--noise_type',            type=str,    default='uniform',     help='normal | uniform')
parser.add_argument('--weight_init',           type=str,    default='normal',      help='normal (0.02 mean)| xavier | he')
parser.add_argument('--norm',                  type=str,    default='batch',       help='batch | instance | layer')
# CoGAN
parser.add_argument('--g_arch',                type=str,    default='digit',       help='digit | rotate | 256 | face | digit_noshare')
parser.add_argument('--d_arch',                type=str,    default='digit',       help='digit | rotate | 256 | face | digit_noshare')
parser.add_argument('--cogan_data',            type=str,    default='mnist2edge',  help='mnist2edge | mnist2rotate | mnist2svhn | mnist2negative | Smiling | '
                                                                                        'Blond_Hair | Male | apple2orange | horse2zebra | vangogh2photo')

args = parser.parse_args()

# Debugging

#args.gan_type = "res128"
#args.norm='layer'
#args.loss = 'wgan'
#args.dir = 'C:/Users/marku/Desktop/gan_training_output/testing'
#args.g_arch = 'digit'
#args.d_arch = 'digit'
#args.batch_size = 16
#args.cogan_data = 'mnist2svhn'
#args.dataset = 'apple2orange'
#args.noise_type='normal'
#args.epochs = 6001
#args.disc_iters = 5
#args.images_while_training = 200
#args.noise_dim=10
#
#args.disc_penalty = 'wgan-gp'
#args.lr_d=0.0001
#args.lr_g=0.0001
#args.b1=0
#args.b2=0.9


args.wd = tf.keras.regularizers.l2(args.weight_decay)
args.bi = tf.keras.initializers.Constant(args.bias_init)
args.w_init = u.select_weight_init(args.weight_init)

# We will reuse this seed overtime for visualization
args.seed = u.gen_noise(args, gen_noise_seed=True)

# Set random seeds for reproducability
tf.random.set_seed(2020)
np.random.seed(2020)


#u.latent_walk('C:/users/marku/Desktop/gan_training_output/relax_weight_sharing/26508/generator1','C:/Users/marku/Desktop/gan_training_output/relax_weight_sharing/26508/generator2',100,3)


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

    # Write config
    u.write_config(args)

    # Select architectures
    generator1, generator2, discriminator1, discriminator2 = u.select_cogan_architecture(args)

    # Start training
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        print('Amount of GPUs: ' + str(len(tf.config.experimental.list_physical_devices('GPU'))) + ' --Using GPU')
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

    # Write config
    u.write_config(args)

    # Select architectures
    generator, discriminator = u.select_gan_architecture(args)

    # Start training
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        print('Amount of GPUs: ' + str(len(tf.config.experimental.list_physical_devices('GPU'))) + ' --Using GPU')
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





