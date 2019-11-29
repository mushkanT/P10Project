import tensorflow as tf
import numpy as np
import time
import Utils as u


def train_discriminator(batch, discriminator, generator, args):
    noise = tf.random.normal([args.batch_size, args.noise_dim])

    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        real_output = discriminator(batch, training=True)

        if args.loss == "wgan-gp":
            disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            alpha = tf.random.uniform(shape=[args.batch_size, 1, 1, 1], minval=0., maxval=1.)
            differences = generated_images - batch
            interpolated_images = batch + (alpha * differences)

            with tf.GradientTape() as gTape:
                gTape.watch(interpolated_images)
                disc_interpolates = discriminator(interpolated_images)

            gradients = gTape.gradient(disc_interpolates, interpolated_images)
            gradient_l2_norm = tf.sqrt(tf.math.reduce_sum(tf.math.square(gradients), 1))
            gradient_penalty = tf.reduce_mean((gradient_l2_norm - 1.) ** 2)
            disc_loss += gradient_penalty * args.gp_lambda

        elif args.loss == "wgan":
            disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

        elif args.loss == "ce":
            cross_entropy = tf.keras.l.BinaryCrossentropy(from_logits=True)
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss
        else:
            raise Exception('Cost function does not exists')

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    args.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    # Clip weights if wgan loss function
    if args.loss == "wgan":
        for var in discriminator.trainable_variables:
            tf.clip_by_value(var, -args.clip, args.clip)
    return disc_loss


def train_generator(discriminator, generator, args):
    noise = tf.random.normal([args.batch_size, args.noise_dim])

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)

        if args.loss == "wgan-gp" or args.loss == "wgan":
            gen_loss = -tf.reduce_mean(fake_output)
        elif args.loss == "ce":
            cross_entropy = tf.keras.l.BinaryCrossentropy(from_logits=True)
            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        else:
            raise Exception('Cost function does not exists')

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    args.gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return gen_loss


def train(dataset, discriminator, generator, args):
    gen_loss = []
    disc_loss = []
    images_while_training = []
    full_training_time = 0

    batches_pr_epoch = args.dataset_dim // args.batch_size
    n_steps = batches_pr_epoch // args.n_critic

    for epoch in range(args.epochs):
        if args.images_while_training != 0:
            if epoch % args.images_while_training == 0:
                if args.dataset == "toy":
                    images_while_training.append(u.draw_2d_samples(generator, args.noise_dim))
                else:
                    images_while_training.append(u.draw_samples(generator, args.seed, args.dataset))

        # TODO: Should find a better way to do this
        dataset.shuffle(args.n_train)
        it = iter(dataset)

        start = time.time()

        for _ in range(n_steps):
            disc_loss_n_critic = []
            # take n steps with critic before training generator
            for i in range(args.n_critic):
                batch = next(it)
                disc_loss_n_critic.append(train_discriminator(batch, discriminator, generator, args))

            gen_loss.append(tf.reduce_mean(train_generator(discriminator, generator, args)).numpy())
            disc_loss.append(tf.reduce_mean(disc_loss_n_critic).numpy())

        full_training_time += time.time()-start

    return gen_loss, disc_loss, images_while_training, full_training_time

