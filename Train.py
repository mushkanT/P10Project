import tensorflow as tf
import keras.backend as K
import numpy as np
import time
import Utils as u
#import matplotlib.pyplot as plt


TINY = 1e-8


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

    def train_discriminator(self, real_data, args):
        noise = tf.random.normal(shape=[args.batch_size, args.noise_dim])

        if args.input_noise:
            disc_input_noise = tf.random.normal(shape=[args.batch_size, args.dataset_dim[1], args.dataset_dim[1], args.dataset_dim[3]], mean=0, stddev=0.1)
        else:
            disc_input_noise = 0

        with tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=False)

            fake_output = self.discriminator(generated_images+disc_input_noise, training=True)
            real_output = self.discriminator(real_data+disc_input_noise, training=True)

            # Losses
            if args.loss == "wgan-gp":
                disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                if args.dataset == 'toy':
                    alpha = tf.random.uniform(shape=[args.batch_size, 1], minval=0., maxval=1.)
                else:
                    alpha = tf.random.uniform(shape=[args.batch_size, 1, 1, 1], minval=0., maxval=1.)

                with tf.GradientTape() as gTape:

                    interpolated_images = alpha * real_data + (1 - alpha) * generated_images

                    # Used in their impl -> Flips signs (large alpha = towards fake)
                    # differences = generated_images - real_data
                    # interpolated_images = real_data + (alpha * differences)

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
                real_labels = np.ones_like(real_output)
                fake_labels = np.zeros_like(fake_output)

                if args.label_flipping:
                    real_indices = np.random.randint(size=[args.batch_size, 1], low=1, high=100)
                    fake_indices = np.random.randint(size=[args.batch_size, 1], low=1, high=100)

                    flip_mask_real = np.array([y <= 5 for y in real_indices])
                    flip_mask_fake = np.array([y <= 5 for y in fake_indices])

                    real_labels = real_labels - flip_mask_real
                    fake_labels = fake_labels + flip_mask_fake

                if args.label_smooth:
                    label_smoothing = np.random.uniform(low=0.9, high=1., size=[1])
                    real_labels = real_labels * label_smoothing

                real_loss = cross_entropy(real_labels, real_output)
                fake_loss = cross_entropy(fake_labels, fake_output)
                disc_loss = real_loss + fake_loss
            else:
                raise Exception('Cost function does not exists')

        # Apply gradients
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        a = zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        args.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        # Calc disc accuracy
        acc_real = K.mean(K.equal(tf.ones_like(real_output), K.round(tf.keras.activations.sigmoid(real_output))))
        acc_fake = K.mean(K.equal(tf.zeros_like(real_output), K.round(tf.keras.activations.sigmoid(fake_output))))
        # Clip weights if wgan loss function
        if args.loss == "wgan":
            for var in self.discriminator.trainable_variables:
                tf.clip_by_value(var, -args.clip, args.clip)
        return disc_loss, acc_fake, acc_real

    def train_generator(self, args):
        noise = tf.random.normal([args.batch_size, args.noise_dim])

        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=False)

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

    def train_discriminator_infogan(self, real_data, args):
        noise = tf.random.normal([args.batch_size, args.noise_dim])

        # Select control variable distributions
        # Categorical variables:
        C_cat = np.random.multinomial()

        # Continuous variables:
        c_cont_list = []
        for i in range(args.c_cont):
            c_cont_list.append(tf.random.uniform(shape=[args.batch_size,args.c_cat]))


        with tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=False)
            fake_output = self.discriminator(generated_images, training=True)
            real_output = self.discriminator(real_data, training=True)

            # Original GAN loss
            cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss

        return disc_loss

    def train_generator_infogan(self):
        return 1

    def train(self, args):
        gen_loss = []
        disc_loss = []
        images_while_training = []
        acc_fakes, acc_reals = [],[]
        full_training_time = 0
        if args.dataset != 'lsun':
            it = iter(self.dataset)
        else:
            it = self.dataset
        batches_pr_epoch = args.dataset_dim[0] // args.batch_size
        n_steps = batches_pr_epoch // args.disc_iters  # Steps per epoch (Generator iterations)

        if self.auxiliary is None:
            train_d = self.train_discriminator
            train_g = self.train_generator
        else:
            train_d = self.train_discriminator_infogan
            train_g = self.train_generator_infogan

        # Image before training
        if args.images_while_training != 0:
            if args.dataset == "toy":
                images_while_training.append(u.draw_2d_samples(self.generator, args.noise_dim))
            else:
                images_while_training.append(u.draw_samples(self.generator, args.seed))

        for epoch in range(args.epochs):
            start = time.time()

            for _ in range(n_steps):
                disc_iters_loss = []
                # take x steps with critic before training generator
                for i in range(args.disc_iters):
                    batch = next(it)
                    if isinstance(batch, np.ndarray):
                        batch = tf.convert_to_tensor(batch)
                    if batch.dtype == tf.float64:
                        batch = tf.dtypes.cast(batch, dtype=tf.float32)
                    if batch.shape[0] != args.batch_size:
                        continue
                    d_loss, acc_fake, acc_real = train_d(batch, args)
                    disc_iters_loss.append(d_loss)
                    acc_fakes.append(acc_fake)
                    acc_reals.append(acc_real)
                    #disc_iters_loss.append(train_d(batch, args))

                gen_loss.append(tf.reduce_mean(train_g(args)).numpy())
                disc_loss.append(tf.reduce_mean(disc_iters_loss).numpy())

            full_training_time += time.time()-start

            # Generate samples and save
            if args.images_while_training != 0:
                if epoch % args.images_while_training == 0:
                    if args.dataset == "toy":
                        images_while_training.append(u.draw_2d_samples(self.generator, args.noise_dim))
                    else:
                        images_while_training.append(u.draw_samples(self.generator, args.seed))

        return gen_loss, disc_loss, images_while_training, full_training_time, acc_fakes, acc_reals
