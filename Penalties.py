import tensorflow as tf


def wasserstein_gp(fake_data, real_data, discriminator):

    # Interpolation constant
    alpha = tf.random.uniform(shape=[real_data.shape[3],1,1,1], minval=0., maxval=1.)

    # Calculate interpolations
    differences = fake_data - real_data
    interpolated_images = real_data + (alpha * differences)

    with tf.GradientTape() as gTape:
        gTape.watch(interpolated_images)
        disc_interpolates = discriminator(interpolated_images)

    gradients = gTape.gradient(disc_interpolates, interpolated_images)
    gradients += 1e-8
    slopes = tf.sqrt(tf.math.reduce_sum(tf.math.square(gradients), 1))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

    return gradient_penalty


def calc_penalty(fake_data, real_data, discriminator, args):
    if args.penalty == 'wgan-gp':
        return wasserstein_gp(fake_data, real_data, discriminator)
    elif args.penalty == 'none':
        return 0
