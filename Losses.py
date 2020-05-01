import tensorflow as tf

k_cross_entropy = tf.keras.losses.BinaryCrossentropy()


def cross_entropy_gen(fake_output):
    G_loss = k_cross_entropy(tf.ones_like(fake_output), fake_output)
    return G_loss


def cross_entropy_disc(fake_output, real_output):
    true_labels = tf.ones_like(real_output)
    false_labels = tf.zeros_like(fake_output)
    #true_labels += 0.05 * tf.random.uniform(tf.shape(true_labels))
    #false_labels += 0.05 * tf.random.uniform(tf.shape(false_labels))
    real_loss = k_cross_entropy(true_labels, real_output)
    fake_loss = k_cross_entropy(false_labels, fake_output)
    D_loss = real_loss + fake_loss
    return D_loss


def wasserstein_gen(fake_output):
    G_loss = -tf.reduce_mean(fake_output)
    return G_loss


def wasserstein_disc(fake_output, real_output):
    D_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
    return D_loss


def recon_criterion(input, target):
    return tf.math.reduce_mean(tf.math.abs(input - target))


def encoder_loss(generator_a, generator_b, encoder_a, encoder_b, noise):
    # generate
    x_a = generator_a(noise)
    x_b = generator_b(noise)
    # encode (within domain)
    latent_recon_x_a = encoder_a.encode(x_a)
    latent_recon_x_b = encoder_b.encode(x_b)
    # encode (cross domain)
    latent_recon_x_ba = encoder_a.encode(x_b)
    latent_recon_x_ab = encoder_b.encode(x_a)
    # generate again
    x_ba = generator_a(latent_recon_x_a)
    x_ab = generator_b(latent_recon_x_b)
    # encode again
    latent_recon_x_aba = encoder_a.encode(x_ba)
    latent_recon_x_bab = encoder_b.encode(x_ab)

    # reconstruction loss
    img_recon_a = recon_criterion(x_ba, x_a)
    img_recon_b = recon_criterion(x_ab, x_b)
    latent_recon_a = recon_criterion(latent_recon_x_a, noise)
    latent_recon_b = recon_criterion(latent_recon_x_b, noise)

    # questionable
    latent_recon_a_cross = recon_criterion(latent_recon_x_ba, noise)
    latent_recon_b_cross = recon_criterion(latent_recon_x_ab, noise)
    latent_cycrecon_aba = recon_criterion(latent_recon_x_aba, noise)
    latent_cycrecon_bab = recon_criterion(latent_recon_x_bab, noise)

    total_loss = img_recon_a + \
                 img_recon_b + \
                 latent_recon_a + \
                 latent_recon_b + \
                 latent_recon_a_cross + \
                 latent_recon_b_cross + \
                 latent_cycrecon_aba + \
                 latent_cycrecon_bab

    return total_loss


def set_losses(args):
    if args.loss == 'ce':
        return cross_entropy_disc, cross_entropy_gen
    elif args.loss == 'wgan':
        return wasserstein_disc, wasserstein_gen
