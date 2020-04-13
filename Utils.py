import tensorflow as tf
import numpy as np
import os.path
import Nets as nets
import Losses as l
import Penalties as p


def draw_2d_samples(generator, n_dim, seed=2019):
    noise = tf.random.normal([3000, n_dim], seed=seed)
    generated_image = generator(noise).numpy()
    return generated_image


def draw_samples(model, test_input):
    predictions = model(test_input, training=False)
    return predictions.numpy()


def write_config(args):
    file = open(os.path.join(args.dir, 'config.txt'), 'w')
    file.write(str(args))
    file.close()


def select_cogan_architecture(args):
    if args.g_arch == 'digit':
        generator1, generator2 = nets.cogan_generators_digit(args)
    elif args.g_arch == 'rotate':
        generator1, generator2 = nets.cogan_generators_rotate(args)
    elif args.g_arch == '256':
        generator1, generator2 = nets.cogan_generators_256(args)
    elif args.g_arch == 'face':
        generator1, generator2 = nets.cogan_generators_faces(args)
    elif args.g_arch == 'digit_noshare':
        generator1, generator2 = nets.cogan_generators_digit_noshare(args)

    if args.d_arch == 'digit':
        discriminator1, discriminator2 = nets.cogan_discriminators_digit(args)
    elif args.d_arch == 'rotate':
        discriminator1, discriminator2 = nets.cogan_discriminators_rotate(args)
    elif args.d_arch == '256':
        discriminator1, discriminator2 = nets.cogan_discriminators_256(args)
    elif args.d_arch == 'face':
        discriminator1, discriminator2 = nets.cogan_discriminators_faces(args)
    elif args.g_arch == 'digit_noshare':
        discriminator1, discriminator2 = nets.cogan_discriminators_digit_noshare(args)

    return generator1, generator2, discriminator1, discriminator2


def set_losses(args):
    if args.loss == 'ce':
        return l.cross_entropy_disc, l.cross_entropy_gen
    elif args.loss == 'wgan':
        return l.wasserstein_disc, l.wasserstein_gen


def select_gan_architecture(args):
    if args.dataset == 'toy':
        generator = nets.toy_gen(args.noise_dim)
        discriminator = nets.toy_disc(args)
    elif args.gan_type == '64':
        generator = nets.gan64_gen(args)
        discriminator = nets.gan64_disc(args)
    elif args.gan_type == '128':
        generator = nets.gan128_gen(args)
        discriminator = nets.gan128_disc(args)
    elif args.gan_type == 'cifargan':
        generator = nets.cifargan_gen(args)
        discriminator = nets.cifargan_disc(args)

    else:
        raise NotImplementedError()

    return generator, discriminator


