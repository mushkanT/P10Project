
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import Losses as losses
import Nets as nets
import Data as dt
import pandas as pd

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Create and plot toy data set
dat = dt.createToyDataRing()
#dt.plot_distribution(dat, "Toy data distribution")

# Batch and shuffle the data
toy_train_dataset = tf.data.Dataset.from_tensor_slices(dat).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

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

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 500
noise_dim = 10
num_examples_to_generate = 16
generator = nets.generator_toy()
discriminator = nets.discriminator_toy()
# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = losses.generator_loss_CE(fake_output)
        disc_loss = losses.discriminator_loss_CE(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

#        generate_and_save_images(generator, epoch + 1, seed)


        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        if epoch%100 == 0:
            dt.draw_samples_and_plot_2d(generator)


  # Generate after the final epoch
  # generate_and_save_images(generator, epochs, seed)




def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions, cmap='gray')
        plt.axis('off')

    plt.show()


train(toy_train_dataset,EPOCHS)

#Draw 10000 samples from trained generator and plot them (only works for 2d)
a = []
for c in range(10000):
    noise = tf.random.normal([1, 10])
    generated_image = generator(noise, training=False)
    a.append(generated_image)

#dt.plot_distribution(dat)
a = tf.convert_to_tensor(tf.reshape(a,[10000,2]))
dt.plot_distribution(a)