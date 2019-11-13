
import tensorflow as tf
import glob
import imageio
import numpy as np
import os
import time
import Losses as losses
import Nets as nets
import Data as dt

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 500
noise_dim = 10
num_examples_to_generate = 16

# Toy data set
dat = dt.createToyDataRing()
toy_train_dataset = tf.data.Dataset.from_tensor_slices(dat).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dt.plot_distribution(dat, "Toy data distribution")

# Mnist
'''
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
'''

# Settings
generator_optimizer = tf.keras.optimizers.Adam(1e-3)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-3)

generator = nets.generator_toy()
discriminator = nets.discriminator_toy()

loss_func = "wgan-gp"

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
#@tf.function
def train_step(images, loss):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        if loss == "wgan-gp":
            disc_loss, gen_loss = losses.wasserstein_gp(images, generated_images, real_output, fake_output, BATCH_SIZE, 10, discriminator)
        elif loss == "ce":
            disc_loss, gen_loss = losses.crossEntropy(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs, loss):
    for epoch in range(epochs):
        start = time.time()

        if epoch%100 == 0:
            dt.draw_samples_and_plot_2d(generator, epoch)

        for image_batch in dataset:
            train_step(image_batch, loss)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


train(toy_train_dataset, EPOCHS, loss_func)

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