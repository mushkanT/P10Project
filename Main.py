import tensorflow as tf
import numpy as np
import Nets as nets
import Data as dt
import Train as t
import Utils as u
import argparse
import os.path

parser = argparse.ArgumentParser()

# Settings
parser.add_argument('--dataset', type=str,            default = 'toy'       , help=' toy | mnist | cifar10 ')
parser.add_argument('--n_train', type=int,            default = 60000       , help='training set size, default to mnist')
parser.add_argument('--n_test', type=int,             default = 10000       , help='test set size, default to mnist')

parser.add_argument('--noise_dim', type=int,          default = 10         , help='size of the latent vector')

parser.add_argument('--loss', type=str,               default = 'wgan-gp'   , help='wgan-gp | wgan | ce')
parser.add_argument('--batch_size', type=int,         default = 100)
parser.add_argument('--epochs', type=int,             default = 1)
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

args = parser.parse_args()

args.dir = 'C:/Users/marku/Desktop'

# Write config
file = open(os.path.join(args.dir, 'config.txt'), 'w')
file.write(str(args))
file.close()

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


# Write losses, image values, and full training time + save models
for i in range(len(g_loss)):
    tf.print(g_loss[i])
tf.print('|')
for i in range(len(d_loss)):
    tf.print(d_loss[i])
for i in range(len(images_while_training)):
    tf.print(images_while_training[i])
tf.print('|'+str(full_training_time))

'''
file = open(os.path.join(args.dir, 'losses.txt'), 'w')
file.write(str(g_loss))
file.write('|'+str(d_loss))
file.close()

file = open(os.path.join(args.dir, 'itw.txt'), 'w')
file.write(str(images_while_training))
file.close()
'''

file = open(os.path.join(args.dir, 'config.txt'), 'a')
file.write(str(full_training_time))
file.close()

generator.    save(args.dir+'/generator')
discriminator.save(args.dir+'/discriminator')

#u.plot_loss(g_loss, d_loss)




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