import tensorflow as tf
import numpy as np
import os.path
import Nets as nets


def draw_2d_samples(generator, n_dim, seed=2019):
    noise = tf.random.normal([3000, n_dim], seed=seed)
    generated_image = generator(noise).numpy()
    return generated_image


def draw_samples(model, test_input):
    predictions = model(test_input, training=False)
    return predictions.numpy()


def generate_latent_vector_infogan(args):
    noise = tf.random.normal([args.batch_size, args.noise_dim])
    c = np.random.randint(0, args.c_dim, args.batch_size)
    c = tf.keras.utils.to_categorical(c, num_classes=args.c_dim)
    latent_vector = np.hstack((noise, c))
    return latent_vector, c


def write_config(args):
    file = open(os.path.join(args.dir, 'config.txt'), 'w')
    file.write(str(args))
    file.close()


def select_models(args):
    if args.dataset == 'toy':
        generator = nets.toy_gen(args.noise_dim)
        discriminator = nets.toy_disc(args)
        auxiliary = None
    elif args.gan_type == 'infogan':
        generator = nets.infogan_gen(args)
        discriminator, auxiliary = nets.infogan_disc(args)
    elif args.gan_type == 'tfgan':
        generator = nets.tfgan_gen(args)
        discriminator = nets.tfgan_disc(args)
        auxiliary = None
    elif args.gan_type == 'dcgan':
        generator = nets.dcgan_gen(args)
        discriminator = nets.dcgan_disc(args)
        auxiliary = None
    elif args.gan_type == 'cifargan':
        generator = nets.cifargan_gen(args)
        discriminator = nets.cifargan_disc(args)
        auxiliary = None
    elif args.gan_type == 'cifargan_u':
        generator = nets.cifargan_ups_gen(args)
        discriminator = nets.cifargan_disc(args)
        auxiliary = None
    else:
        raise NotImplementedError()
    #generator._name = 'gen'
    #discriminator._name = 'disc'
    return generator, discriminator, auxiliary


'''
#Test generator and discriminator - before training.
generator = nets.generator_dcgan()

noise = tf.random.normal([1, 10])
generated_image = generator(noise, training=False)

plt.imshow(generated_image, cmap='gray')
plt.show()

discriminator = nets.discriminator_dcgan()
decision = discriminator(generated_image)
print (decision)
'''


