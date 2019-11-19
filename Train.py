import tensorflow as tf
import numpy as np
from keras import backend as K
import time
import Data as dt
import utils as u
import Losses as losses


def train_step(images, discriminator, generator, args):
    noise = tf.random.normal([args["batch_size"], args["noise_dim"]])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        if args["loss"] == "wgan-gp":
            disc_loss, gen_loss = losses.wasserstein_gp(images, generated_images, real_output, fake_output, args["batch_size"], 10, discriminator)
        elif args["loss"] == "ce":
            disc_loss, gen_loss = losses.crossEntropy(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    args["gen_optimizer"].apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    args["disc_optimizer"].apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train_discriminator(dataset, discriminator, generator, args):
    for images in dataset:
        noise = tf.random.normal([args["batch_size"], args["noise_dim"]])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            if args["loss"] == "wgan-gp":
                disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                alpha = tf.random.uniform(shape=[args["batch_size"], 1], minval=0., maxval=1.)

                with tf.GradientTape() as gTape:
                    # interpolated_images = alpha * images + ((1 - alpha) * generated_images)

                    # WGAN-GP implementation does not correspond with what they wrote in their paper
                    differences = generated_images - images
                    interpolated_images = images + (alpha * differences)

                    gTape.watch(interpolated_images)
                    disc_interpolates = discriminator(interpolated_images, training=False)

                gradients = gTape.gradient(disc_interpolates, interpolated_images)
                l2_norm = tf.sqrt(tf.math.reduce_sum(tf.math.square(gradients), 1))
                gradient_penalty = tf.reduce_mean((l2_norm - 1.) ** 2)

                disc_loss += gradient_penalty * args["lambda"]

            elif args["loss"] == "wgan":
                disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

            elif args["loss"] == "ce":
                cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                real_loss = cross_entropy(tf.ones_like(real_output), real_output)
                fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
                disc_loss = real_loss + fake_loss

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        args["disc_optimizer"].apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train_generator(dataset, discriminator, generator, args):
    for _ in dataset:
        noise = tf.random.normal([args["batch_size"], args["noise_dim"]])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)
            fake_output = discriminator(generated_images, training=True)

            if args["loss"] == "wgan-gp" or args["wgan"]:
                gen_loss = -tf.reduce_mean(fake_output)
            elif args["loss"] == "ce":
                cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        args["gen_optimizer"].apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


def train(dataset, discriminator, generator, args):
    for epoch in range(args["epochs"]):
        if epoch % 10 == 0:
            u.draw_samples_and_plot_2d(generator, epoch)

        start = time.time()

        '''
        # take n steps with critic before training generator
        for i in range(args["n_critic"]):
            train_discriminator(dataset, discriminator, generator, args)

        train_generator(dataset, discriminator, generator, args)
        '''

        for images in dataset:
            train_step(images, discriminator, generator, args)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
