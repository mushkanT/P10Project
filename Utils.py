import tensorflow as tf
import numpy as np
import os.path


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


