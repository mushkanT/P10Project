import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from Code import Utils as u, Data as d, Losses as l

layers = tf.keras.layers

tf.random.set_seed(2020)
np.random.seed(2020)

parser = argparse.ArgumentParser()
parser.add_argument('--dir',            type=str,           default='/user/student.aau.dk/mjuuln15/output_data',     help='Directory to save images, models, weights etc')
parser.add_argument('--sample_itr',            type=int,           default=250)
parser.add_argument('--purpose', type=str, default='')
args = parser.parse_args()

args.dir = 'C:/Users/marku/Desktop/gan_training_output/testing'
args.sample_itr = 10
args.cogan_data='mnist2edge'
args.g_arch = 'digit_noshare'
args.d_arch = 'digit_noshare'
args.batch_size = 64
args.noise_dim = 100

u.write_config(args)

X1, X2, shape = d.select_dataset_cogan(args)
args.dataset_dim = shape
gen_a, gen_b, disc_a, disc_b = u.select_cogan_architecture(args)


class CCEncoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CCEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=[32, 32, 1]),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim),
            ]
        )

    def encode(self, x):
        reprod_latent = self.encoder(x)
        return reprod_latent


class CCGenerator(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CCGenerator, self).__init__()
        self.latent_dim = latent_dim

        self.generator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=4 * 4 * 1024, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(4, 4, 1024)),
                tf.keras.layers.Conv2DTranspose(
                    filters=512,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.PReLU(),
                tf.keras.layers.Conv2DTranspose(
                    filters=256,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.PReLU(),
                tf.keras.layers.Conv2DTranspose(
                    filters=128,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.PReLU(),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=6, strides=(1, 1), activation='tanh', padding="SAME"),
            ]
        )

    def generate(self, z):
        image = self.generator(z)
        return image


class CCDiscriminator(tf.keras.Model):
    def __init__(self):
        super(CCDiscriminator, self).__init__()

        self.discriminator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=[32, 32, 1]),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=5, strides=(2, 2)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.PReLU(),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.PReLU(),

                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(1),
            ]
        )

    def discriminate(self, x):
        response = self.discriminator(x)
        return response


optimizer_d = tf.keras.optimizers.Adam(1e-4)
optimizer_g = tf.keras.optimizers.Adam(1e-4)


def recon_criterion(input, target):
    return tf.math.reduce_mean(tf.math.abs(input - target))


def compute_generator_loss(generator_a, generator_b, discriminator_a, discriminator_b):
    noise = tf.random.normal([64, 100])
    # generate
    x_a = generator_a.generate(noise)
    x_b = generator_b.generate(noise)

    # GAN loss
    d_a = discriminator_a.discriminate(x_a)
    loss_gen_adv_a = l.cross_entropy_gen(d_a)
    d_b = discriminator_b.discriminate(x_b)
    loss_gen_adv_b = l.cross_entropy_gen(d_b)

    total_loss = loss_gen_adv_a + loss_gen_adv_b
    return total_loss


def compute_generator_loss_single(generator, discriminator, noise):
    # generate
    x = generator(noise, training=True)
    d = discriminator(x, training=True)
    # GAN loss
    loss_gen_adv = l.cross_entropy_gen(d)
    return loss_gen_adv


def compute_encoder_loss(generator_a, generator_b, encoder_a, encoder_b, noise):
    # generate
    x_a = generator_a(noise)
    x_b = generator_b(noise)
    # encode (within domain)
    latent_recon_x_a = encoder_a.encode(x_a)
    latent_recon_x_b = encoder_b.encode(x_b)
    # encode (cross domain)
    latent_recon_x_ba = encoder_a.encode(x_b)
    latent_recon_x_ab = encoder_b.encode(x_a)
    # generate again
    x_ba = generator_a(latent_recon_x_a)
    x_ab = generator_b(latent_recon_x_b)
    # encode again
    latent_recon_x_aba = encoder_a.encode(x_ba)
    latent_recon_x_bab = encoder_b.encode(x_ab)

    # reconstruction loss
    img_recon_a = recon_criterion(x_ba, x_a)
    img_recon_b = recon_criterion(x_ab, x_b)
    latent_recon_a = recon_criterion(latent_recon_x_a, noise)
    latent_recon_b = recon_criterion(latent_recon_x_b, noise)

    # questionable
    latent_recon_a_cross = recon_criterion(latent_recon_x_ba, noise)
    latent_recon_b_cross = recon_criterion(latent_recon_x_ab, noise)
    latent_cycrecon_aba = recon_criterion(latent_recon_x_aba, noise)
    latent_cycrecon_bab = recon_criterion(latent_recon_x_bab, noise)

    total_loss = img_recon_a + \
                 img_recon_b + \
                 latent_recon_a + \
                 latent_recon_b + \
                 latent_recon_a_cross + \
                 latent_recon_b_cross + \
                 latent_cycrecon_aba + \
                 latent_cycrecon_bab

    return total_loss


def compute_discriminator_loss(generator_a, generator_b, discriminator_a, discriminator_b, x_a, x_b):
    noise = tf.random.normal([64, 100])
    x_a_fake = generator_a.generate(noise)
    x_b_fake = generator_b.generate(noise)

    x_a_fake = discriminator_a.discriminate(x_a_fake)
    x_b_fake = discriminator_b.discriminate(x_b_fake)
    x_a = discriminator_a.discriminate(x_a)
    x_b = discriminator_b.discriminate(x_b)

    # GAN loss
    loss_disc_adv_a = l.cross_entropy_disc(x_a_fake, x_a)
    loss_disc_adv_b = l.cross_entropy_disc(x_b_fake, x_b)

    total_loss = loss_disc_adv_a + loss_disc_adv_b

    return total_loss


def compute_discriminator_loss_single(generator, discriminator, x, noise):
    x_fake = generator(noise, training=True)
    x_fake = discriminator(x_fake, training=True)
    x_real = discriminator(x, training=True)
    # GAN loss
    loss_disc_adv = l.cross_entropy_disc(x_fake, x_real)
    return loss_disc_adv


def compute_apply_gradients(generator_a, generator_b, discriminator_a, discriminator_b, encoder_a, encoder_b, x_a, x_b):

    # Train D1 and D2
    noise = tf.random.normal([64, 100])

    with tf.GradientTape() as tape:
        loss1 = compute_discriminator_loss_single(generator_a, discriminator_a, x_a, noise)
    gradients = tape.gradient(loss1, discriminator_a.trainable_variables)
    optimizer_d.apply_gradients(zip(gradients, discriminator_a.trainable_variables))

    with tf.GradientTape() as tape:
        loss2 = compute_discriminator_loss_single(generator_b, discriminator_b, x_b, noise)
    gradients = tape.gradient(loss2, discriminator_b.trainable_variables)
    optimizer_d.apply_gradients(zip(gradients, discriminator_b.trainable_variables))

    # Train G1 and G2 + E1 and E2
    noise = tf.random.normal([64, 100])

    with tf.GradientTape() as tape:
        loss3 = compute_generator_loss_single(generator_a, discriminator_a, noise)
    gradients = tape.gradient(loss3, generator_a.trainable_variables)
    optimizer_g.apply_gradients(zip(gradients, generator_a.trainable_variables))

    with tf.GradientTape() as tape:
        loss4 = compute_encoder_loss(generator_a, generator_b, encoder_a, encoder_b, noise)
    gradients = tape.gradient(loss4, encoder_a.trainable_variables)
    optimizer_g.apply_gradients(zip(gradients, encoder_a.trainable_variables))

    with tf.GradientTape() as tape:
        loss5 = compute_generator_loss_single(generator_b, discriminator_b, noise)
    gradients = tape.gradient(loss5, generator_b.trainable_variables)
    optimizer_g.apply_gradients(zip(gradients, generator_b.trainable_variables))

    with tf.GradientTape() as tape:
        loss6 = compute_encoder_loss(generator_a, generator_b, encoder_a, encoder_b, noise)
    gradients = tape.gradient(loss6, encoder_b.trainable_variables)
    optimizer_g.apply_gradients(zip(gradients, encoder_b.trainable_variables))

    return loss1, loss2, loss3, loss4, loss5, loss6


def sample_images(g1, g2, epoch, seed, dir):
    r, c = 4, 4
    gen_batch1 = g1(seed)
    gen_batch2 = g2(seed)

    gen_imgs = np.concatenate([gen_batch1, gen_batch2])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0

    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(os.path.join(dir, "images/%d.png" % epoch))
    plt.close()


it = iter(X1)
it1 = iter(X2)
encoder_a = CCEncoder(100)
encoder_b = CCEncoder(100)

z = tf.random.normal([8, 100])

for i in range(10000):
    images1 = next(it)
    images2 = next(it1)
    l1, l2, l3, l4, l5, l6 = compute_apply_gradients(gen_a, gen_b, disc_a, disc_b, encoder_a, encoder_b, images1, images2)
    print("iteration: " + str(i) + " \t D1: " + str(l1.numpy()) + " \t D2: " + str (l2.numpy()) + " G1: " + str(l3.numpy())+ " G2: " + str(l5.numpy()) + " E1: " + str(l5.numpy())+ " E2: " + str(l6.numpy()))
    if i % args.sample_itr == 0:
        sample_images(gen_a, gen_b, i, z, args.dir)


