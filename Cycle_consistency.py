import tensorflow as tf
import numpy as np
import Losses as l
import os
import matplotlib.pyplot as plt
import cv2
import argparse
import Utils as u
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

u.write_config(args)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# 28x28 -> 32x32
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
padding = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
train_images = tf.pad(train_images, padding, "CONSTANT")

# Split dataset
X1 = train_images[:int(train_images.shape[0] / 2)]
X2 = train_images[int(train_images.shape[0] / 2):]

edges = np.zeros((X2.shape[0], 32, 32, 1))
for idx, i in enumerate(X2):
    i = np.squeeze(i)
    dilation = cv2.dilate(i, np.ones((3, 3), np.uint8), iterations=1)
    edge = dilation - i
    edges[idx - X2.shape[0], :, :, 0] = edge
X2 = tf.convert_to_tensor(edges)

X1 = (X1 - 127.5) / 127.5  # Normalize the images to [-1, 1]
X2 = (X2 - 127.5) / 127.5  # Normalize the images to [-1, 1]

X1 = tf.data.Dataset.from_tensor_slices(X1).shuffle(X1.shape[0]).repeat().batch(
    64)
X2 = tf.data.Dataset.from_tensor_slices(X2).shuffle(X2.shape[0]).repeat().batch(
    64)


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


#@tf.function
def compute_generator_loss(generator_a, generator_b, discriminator_a, discriminator_b, noise):
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


def compute_encoder_loss(generator_a, generator_b, encoder_a, encoder_b, noise):
    # generate
    x_a = generator_a.generate(noise)
    x_b = generator_b.generate(noise)
    # encode (within domain)
    latent_recon_x_a = encoder_a.encode(x_a)
    latent_recon_x_b = encoder_b.encode(x_b)
    # encode (cross domain)
    latent_recon_x_ba = encoder_a.encode(x_b)
    latent_recon_x_ab = encoder_b.encode(x_a)
    # generate again
    x_ba = generator_a.generate(latent_recon_x_ba)
    x_ab = generator_b.generate(latent_recon_x_ab)
    # encode again
    latent_recon_x_aba = encoder_a.encode(x_ba)
    latent_recon_x_bab = encoder_b.encode(x_ab)

    # reconstruction loss
    loss_gen_recon_x_a = recon_criterion(x_ba, x_a)
    loss_gen_recon_x_b = recon_criterion(x_ab, x_b)
    loss_gen_recon_latent_a = recon_criterion(latent_recon_x_a, noise)
    loss_gen_recon_latent_b = recon_criterion(latent_recon_x_b, noise)
    loss_gen_recon_latent_a_cross = recon_criterion(latent_recon_x_ba, noise)
    loss_gen_recon_latent_b_cross = recon_criterion(latent_recon_x_ab, noise)
    loss_gen_cycrecon_latent_aba = recon_criterion(latent_recon_x_aba, noise)
    loss_gen_cycrecon_latent_bab = recon_criterion(latent_recon_x_bab, noise)

    total_loss = loss_gen_recon_x_a + \
                 loss_gen_recon_x_b + \
                 loss_gen_recon_latent_a + \
                 loss_gen_recon_latent_b + \
                 loss_gen_recon_latent_a_cross + \
                 loss_gen_recon_latent_b_cross + \
                 loss_gen_cycrecon_latent_aba + \
                 loss_gen_cycrecon_latent_bab

    return total_loss


#@tf.function
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


#@tf.function
def compute_discriminator_loss_single(generator, discriminator, x):
    noise = tf.random.normal([64, 100])
    x_a_fake = generator.generate(noise)

    x_a_fake = discriminator.discriminate(x_a_fake)
    x_a = discriminator.discriminate(x)

    # GAN loss
    loss_disc_adv_a = l.cross_entropy_disc(x_a_fake, x_a)

    return loss_disc_adv_a


#@tf.function
def compute_apply_gradients(generator_a, generator_b, discriminator_a, discriminator_b, encoder_a, encoder_b, noise, x_a, x_b):

    with tf.GradientTape() as tape:
        loss = compute_generator_loss(generator_a, generator_b, discriminator_a, discriminator_b, noise)
    gradients = tape.gradient(loss, generator_a.trainable_variables)
    optimizer_g.apply_gradients(zip(gradients, generator_a.trainable_variables))

    with tf.GradientTape() as tape:
        loss = compute_encoder_loss(generator_a, generator_b, encoder_a, encoder_b, noise)
    gradients = tape.gradient(loss, encoder_a.trainable_variables)
    optimizer_g.apply_gradients(zip(gradients, encoder_a.trainable_variables))

    with tf.GradientTape() as tape:
        loss = compute_generator_loss(generator_a, generator_b, discriminator_a, discriminator_b, noise)
    gradients = tape.gradient(loss, generator_b.trainable_variables)
    optimizer_g.apply_gradients(zip(gradients, generator_b.trainable_variables))

    with tf.GradientTape() as tape:
        loss = compute_encoder_loss(generator_a, generator_b, encoder_a, encoder_b, noise)
    gradients = tape.gradient(loss, encoder_b.trainable_variables)
    optimizer_g.apply_gradients(zip(gradients, encoder_b.trainable_variables))

    with tf.GradientTape() as tape:
        loss2 = compute_discriminator_loss_single(generator_a, discriminator_a, x_a)
    gradients = tape.gradient(loss2, discriminator_a.trainable_variables)
    optimizer_d.apply_gradients(zip(gradients, discriminator_a.trainable_variables))

    with tf.GradientTape() as tape:
        loss1 = compute_discriminator_loss_single(generator_b, discriminator_b, x_b)
    gradients = tape.gradient(loss1, discriminator_b.trainable_variables)
    optimizer_d.apply_gradients(zip(gradients, discriminator_b.trainable_variables))

    return loss, loss2, loss1


def sample_images(g1, g2, epoch, seed, dir):
    r, c = 4, 4
    gen_batch1 = g1.generate(seed)
    gen_batch2 = g2.generate(seed)

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

gen_a = CCGenerator(100)
gen_b = CCGenerator(100)
encoder_a = CCEncoder(100)
encoder_b = CCEncoder(100)
disc_a = CCDiscriminator()
disc_b = CCDiscriminator()

z = tf.random.normal([8, 100])

for i in range(10000):
    images = next(it)
    images1 = next(it1)
    images1 = tf.cast(images1, dtype=tf.float32)
    noise = tf.random.normal([64, 100])
    l_ge, l_d, l_d1 = compute_apply_gradients(gen_a, gen_b, disc_a, disc_b, encoder_a, encoder_b, noise, images, images1)
    print("iteration: " + str(i) + " \t ga, gb, ea, eb loss: " + str(l_ge.numpy()) + " \t da: " + str (l_d.numpy()) + " db: " + str(l_d1.numpy()))
    if i % args.sample_itr == 0:
        sample_images(gen_a, gen_b, i, z, args.dir)


