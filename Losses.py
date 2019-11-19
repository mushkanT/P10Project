import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def crossEntropy(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    D_loss = real_loss + fake_loss

    G_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    return D_loss, G_loss


def wasserstein(real_output, fake_output):
    D_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
    G_loss = -tf.reduce_mean(fake_output)

    return D_loss, G_loss


def wasserstein_gp(real_data, generated_data, real_output, fake_output, BATCH_SIZE, LAMBDA, discriminator):
    D_loss, G_loss = wasserstein(real_output, fake_output)

    alpha = tf.random.uniform(
        shape=[BATCH_SIZE, 1],
        minval=0.,
        maxval=1.
    )

    with tf.GradientTape() as gTape:
        interpolated_images = alpha * real_data + ((1-alpha) * generated_data)

        # WGAN-GP implementation does not correspond with what they wrote in their paper
        #differences = generated_data - real_data
        #interpolated_images = real_data + (alpha * differences)

        gTape.watch(interpolated_images)
        disc_interpolates = discriminator(interpolated_images)

    gradients = gTape.gradient(disc_interpolates, interpolated_images)
    slopes = tf.sqrt(tf.math.reduce_sum(tf.math.square(gradients), 1))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

    D_loss += gradient_penalty * LAMBDA

    return D_loss, G_loss