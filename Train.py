import tensorflow as tf
import numpy as np
import time
import Utils as u


class GANTrainer(object):

    def __init__(self,
                 generator,
                 discriminator,
                 auxiliary,
                 dataset
                 ):
        self.generator = generator
        self.discriminator = discriminator
        self.auxiliary = auxiliary
        self.dataset = dataset

    def train_discriminator(self, batch, args):
        noise = tf.random.normal([args.batch_size, args.noise_dim])

        with tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            real_output = self.discriminator(batch, training=True)

            # Losses
            if args.loss == "wgan-gp":
                disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                if args.dataset == 'toy':
                    alpha = tf.random.uniform(shape=[args.batch_size, 1], minval=0., maxval=1.)
                else:
                    alpha = tf.random.uniform(shape=[args.batch_size, 1, 1, 1], minval=0., maxval=1.)
                differences = generated_images - batch
                interpolated_images = batch + (alpha * differences)

                with tf.GradientTape() as gTape:
                    gTape.watch(interpolated_images)
                    disc_interpolates = self.discriminator(interpolated_images)

                gradients = gTape.gradient(disc_interpolates, interpolated_images)
                gradient_l2_norm = tf.sqrt(tf.math.reduce_sum(tf.math.square(gradients), 1))
                gradient_penalty = tf.reduce_mean((gradient_l2_norm - 1.) ** 2)
                disc_loss += gradient_penalty * args.gp_lambda

            elif args.loss == "wgan":
                disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

            elif args.loss == "ce":
                cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                real_loss = cross_entropy(tf.ones_like(real_output), real_output)
                fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
                disc_loss = real_loss + fake_loss
            else:
                raise Exception('Cost function does not exists')

        # Apply gradients
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        args.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        # Clip weights if wgan loss function
        if args.loss == "wgan":
            for var in self.discriminator.trainable_variables:
                tf.clip_by_value(var, -args.clip, args.clip)
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

        # Apply graidents
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        args.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return gen_loss

    def train_discriminator_infogan(self):
        return 1

    def train_generator_infogan(self):
        return 1

    def train(self, args):
        gen_loss = []
        disc_loss = []
        images_while_training = []
        full_training_time = 0

        batches_pr_epoch = args.dataset_dim[0] // args.batch_size
        n_steps = batches_pr_epoch // args.n_critic

        if self.auxiliary is None:
            train_d = self.train_discriminator
            train_g = self.train_generator
        else:
            train_d = self.train_discriminator_infogan
            train_g = self.train_generator_infogan

        for epoch in range(args.epochs):

            # TODO: Should find a better way to do this
            self.dataset.shuffle(args.dataset_dim[0])
            it = iter(self.dataset)
            start = time.time()

            for _ in range(n_steps):
                disc_loss_n_critic = []
                # take n steps with critic before training generator
                for i in range(args.n_critic):
                    batch = next(it)
                    disc_loss_n_critic.append(train_d(batch, args))

                gen_loss.append(tf.reduce_mean(train_g(args)).numpy())
                disc_loss.append(tf.reduce_mean(disc_loss_n_critic).numpy())

            full_training_time += time.time()-start

            if args.images_while_training != 0:
                if epoch % args.images_while_training == 0:
                    if args.dataset == "toy":
                        images_while_training.append(u.draw_2d_samples(self.generator, args.noise_dim))
                    else:
                        images_while_training.append(u.draw_samples(self.generator, args.seed))

        return gen_loss, disc_loss, images_while_training, full_training_time