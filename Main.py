import tensorflow as tf
import numpy as np
import Nets as nets
import Data as dt
import Train as t
import utils as u
import argparse

parser = argparse.ArgumentParser()

# Settings
parser.add_argument('--dataset', type=str,            default = 'mnist'       , help=' toy | mnist | cifar10 ')
parser.add_argument('--n_train', type=int,            default = 60000       , help='training set size, default to mnist')
parser.add_argument('--n_test', type=int,             default = 10000       , help='test set size, default to mnist')

parser.add_argument('--noise_dim', type=int,          default = 100         , help='size of the latent vector')

parser.add_argument('--loss', type=str,               default = 'wgan-gp'   , help='wgan-gp | wgan | ce')
parser.add_argument('--batch_size', type=int,         default = 100)
parser.add_argument('--epochs', type=int,             default = 1000)
parser.add_argument('--n_critic', type=int,           default = 1)
parser.add_argument('--clip', type=float,             default = 0.01        , help='upper bound for clipping')
parser.add_argument('--gp_lambda', type=int,          default = 10)
parser.add_argument('--lr_d', type=float,             default = 1e-4)
parser.add_argument('--lr_g', type=float,             default = 1e-4)
parser.add_argument('--optim_d', type=str,            default = 'adam'      , help='adam')
parser.add_argument('--optim_g', type=str,            default = 'adam'      , help='adam')
parser.add_argument('--num_samples_to_gen', type=int, default = 16)

args = parser.parse_args()

# Choose correct model architecture
if args.dataset == "toy":
    dat = dt.createToyDataRing()
    train_dataset = tf.data.Dataset.from_tensor_slices(dat).shuffle(args.n_train).batch(args.batch_size)
    #u.plot_toy_distribution(dat, "2 dimensional toy data distribution")
    generator = nets.generator_toy(args.noise_dim)
    discriminator = nets.discriminator_toy()
elif args.dataset == "mnist":
    dat = dt.mnist()
    train_dataset = tf.data.Dataset.from_tensor_slices(dat).shuffle(args.n_train).batch(args.batch_size)
    generator = nets.generator_dcgan()
    discriminator = nets.discriminator_dcgan()
else:
    raise Exception("Dataset not available")

# Choose correct optimizers
if args.optim_d == "adam":
    args.gen_optimizer = tf.keras.optimizers.Adam(args.lr_d)

if args.optim_g == "adam":
    args.disc_optimizer = tf.keras.optimizers.Adam(args.lr_g)

args.dataset_dim = dat.shape[0]

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
args.seed = tf.random.normal([args.num_samples_to_gen, args.noise_dim])

if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    with tf.device('/GPU:0'):
        g_loss, d_loss = t.train(train_dataset, discriminator, generator, args)
else:
    g_loss, d_loss = t.train(train_dataset, discriminator, generator, args)

u.plot_loss(g_loss, d_loss)




'''
#Test generator and discriminator - before training.
generator = nets.generator_dcgan()

noise = tf.random.normal([1, 10])
generated_image = generator(noise, training=False)

plt.imshow(generated_image, cmap='gray')
plt.show()

discriminator = nets.discriminator_dcgan()
decision = discriminator(generated_image)
print (decision)
'''