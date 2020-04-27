import tensorflow as tf
import keras.backend as K
import numpy as np
import time
import Utils as u
import random
import matplotlib.pyplot as plt
import random
import os

TINY = 1e-8


class GANTrainer(object):

    def __init__(self,
                 generator,
                 discriminator,
                 dataset
                 ):
        self.generator = generator
        self.discriminator = discriminator
        self.dataset = dataset

    def train_discriminator(self, real_data, args):
               
        #real_data=real_data[:int(real_data.shape[0]/2)]

        if args.input_noise:
            disc_input_noise = tf.random.normal(shape=[args.batch_size, args.dataset_dim[1], args.dataset_dim[1], args.dataset_dim[3]], mean=0, stddev=args.variance)
            args.variance = args.variance * 0.95
        else:
            disc_input_noise = 0

        noise = tf.random.normal(shape=[args.batch_size, args.noise_dim])
        generated_images = self.generator(noise, training=True)
        # comb = tf.concat([generated_images, real_data], axis=0)

        real_labels = tf.ones((args.batch_size, 1))
        fake_labels = tf.zeros((args.batch_size, 1))
        # labels = tf.concat([real_labels, fake_labels], axis=0)

        with tf.GradientTape() as disc_tape:

            #disc_resp = self.discriminator(comb)
            fake_output = self.discriminator(generated_images+disc_input_noise, training=True)
            real_output = self.discriminator(real_data+disc_input_noise, training=True)

            # Losses
            if args.loss == "wgan-gp":
                disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                if args.dataset == 'toy':
                    alpha = tf.random.uniform(shape=[args.batch_size, 1], minval=0., maxval=1.)
                else:
                    alpha = tf.random.uniform(shape=[args.batch_size, 1, 1, 1], minval=0., maxval=1.)
                    #alpha = tf.random.uniform(shape=[int(args.batch_size/2), 1, 1, 1], minval=0., maxval=1.)
                   
                #interpolated_images = alpha * real_data + (1 - alpha) * generated_images

                # Used in their impl -> Flips signs (large alpha = towards fake)
                differences = generated_images - real_data
                interpolated_images = real_data + (alpha * differences)
 
                with tf.GradientTape() as gTape:
                    gTape.watch(interpolated_images)
                    disc_interpolates = self.discriminator(interpolated_images, training=True)

                gradients = gTape.gradient(disc_interpolates, interpolated_images)
                gradients += 1e-8
                gradient_l2_norm = tf.sqrt(tf.math.reduce_sum(tf.math.square(gradients), 1))
                gradient_penalty = tf.reduce_mean((gradient_l2_norm - 1.) ** 2)
                disc_loss += gradient_penalty * args.gp_lambda

            elif args.loss == "wgan":
                disc_loss = tf.reduce_mean(fake_output)-tf.reduce_mean(real_output)

            elif args.loss == "ce":
                cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

                if args.label_flipping:
                    real_indices = np.random.randint(size=[args.batch_size, 1], low=1, high=100)
                    fake_indices = np.random.randint(size=[args.batch_size, 1], low=1, high=100)

                    flip_mask_real = np.array([y <= 5 for y in real_indices])
                    flip_mask_fake = np.array([y <= 5 for y in fake_indices])

                    real_labels = real_labels - flip_mask_real
                    fake_labels = fake_labels + flip_mask_fake

                if args.label_smooth:
                    real_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))
                    fake_labels += 0.05 * tf.random.uniform(tf.shape(fake_labels))
                    #labels = 0.05 * tf.random.uniform(tf.shape(labels))

                #disc_loss = cross_entropy(labels, disc_resp)
                real_loss = cross_entropy(tf.convert_to_tensor(real_labels, dtype=tf.float32), real_output)
                fake_loss = cross_entropy(tf.convert_to_tensor(fake_labels, dtype=tf.float32), fake_output)
                disc_loss = real_loss + fake_loss
            else:
                raise Exception('Cost function does not exists')

        # Apply gradients
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        args.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Clip weights if wgan loss function
        if args.loss == "wgan":
            for i, var in enumerate(self.discriminator.trainable_variables):
                self.discriminator.trainable_variables[i].assign(tf.clip_by_value(var, -args.clip, args.clip))

        return disc_loss

    def train_generator(self, args):
        noise = tf.random.normal([args.batch_size, args.noise_dim])

        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Losses
            if args.loss == "wgan-gp" or args.loss == "wgan":
                gen_loss = -tf.reduce_mean(fake_output)
            elif args.loss == "ce":
                cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
            else:
                raise Exception('Cost function does not exists')

        # Apply gradients
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        args.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return gen_loss

    def train(self, args):
        gen_loss = []
        disc_loss = []
        images_while_training = []
        full_training_time = 0
        if args.dataset != 'lsun':
            it = iter(self.dataset)
        else:
            it = self.dataset

        for epoch in range(args.epochs):
            start = time.time()

            disc_iters_loss = []
            # take x steps with critic before training generator
            for i in range(args.disc_iters):
                if args.dataset in ['celeba', 'lsun']:
                    batch = next(it)
                else:
                    batch = next(it)[0]

                if isinstance(batch, np.ndarray):
                    batch = tf.convert_to_tensor(batch)
                if batch[0].dtype == tf.float64:
                    batch = tf.dtypes.cast(batch, dtype=tf.float32)
                d_loss = self.train_discriminator(batch, args)
                disc_iters_loss.append(d_loss)

            gen_loss.append(tf.reduce_mean(self.train_generator(args)).numpy())
            full_training_time += time.time() - start
            disc_loss.append(tf.reduce_mean(disc_iters_loss).numpy())

            # Generate samples and save
            if args.images_while_training != 0:
                if epoch % args.images_while_training == 0:
                    if args.dataset == "toy":
                        images_while_training.append(u.draw_2d_samples(self.generator, args.noise_dim))
                    else:
                        self.sample_images(epoch, args.seed, args.dir, args.dataset_dim[3])

        self.plot_losses(args.dir, disc_loss, gen_loss)
        return full_training_time

    def sample_images(self, epoch, seed, dir, channels):
        r, c = 2, 4
        gen_batch1 = self.generator.predict(seed)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_batch1 + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        # black/white images
        if channels == 1:
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig(os.path.join(dir, "images/%d.png" % epoch))
            plt.close()
        # color images
        else:
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig(os.path.join(dir, "images/%d.png" % epoch))
            plt.close()

    def plot_losses(self, dir, d_loss, gen_loss):
        plt.plot(gen_loss, label='Generator loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(dir, 'losses/gen_loss.png'))
        plt.close()

        plt.plot(d_loss, label='Discriminator loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(dir, 'losses/disc_loss.png'))
        plt.close()
