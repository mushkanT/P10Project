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
    if args.g_arch == 'conv':
        generator1, generator2 = nets.cogan_generators_conv(args)
    elif args.g_arch == 'fc':
        generator1, generator2 = nets.cogan_generators_fc(args)
    elif args.g_arch == 'cross':
        generator1, generator2 = nets.cross_cogan_generators(args)

    if args.d_arch == 'conv':
        discriminator1, discriminator2 = nets.cogan_discriminators_conv(args)
    elif args.d_arch == 'fc':
        discriminator1, discriminator2 = nets.cogan_discriminators_fc(args)
    elif args.d_arch == 'cross':
        discriminator1, discriminator2 = nets.cross_cogan_discriminators()

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
    elif args.gan_type == 'infogan':
        generator = nets.infogan_gen(args)
        discriminator, auxiliary = nets.infogan_disc(args)
    elif args.gan_type == 'tfgan':
        generator = nets.tfgan_gen(args)
        discriminator = nets.tfgan_disc(args)
    elif args.gan_type == 'dcgan':
        generator = nets.dcgan_gen(args)
        discriminator = nets.dcgan_disc(args)
    elif args.gan_type == 'cifargan':
        generator = nets.cifargan_gen(args)
        discriminator = nets.cifargan_disc(args)
    elif args.gan_type == 'cifargan_u':
        generator = nets.cifargan_ups_gen(args)
        discriminator = nets.cifargan_disc(args)
    elif args.gan_type == 'cifargan_bn':
        generator = nets.cifargan_bn_gen(args)
        discriminator = nets.cifargan_bn_disc(args)
    elif args.gan_type == 'cifargan_bn_g':
        generator = nets.cifargan_bn_gen(args)
        discriminator = nets.cifargan_disc(args)
    else:
        raise NotImplementedError()

    return generator, discriminator


