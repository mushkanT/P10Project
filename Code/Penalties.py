import tensorflow as tf
import numpy as np


class DiscriminatorPenalties:
    def wasserstein_gp(self, fake_data, real_data, discriminator):
        # Interpolation constant
        alpha = tf.random.uniform(shape=[real_data.shape[0],1,1,1], minval=0., maxval=1.)

        # Calculate interpolations
        differences = fake_data - real_data
        interpolated_images = real_data + (alpha * differences)

        with tf.GradientTape() as gTape:
            gTape.watch(interpolated_images)
            disc_interpolates = discriminator(interpolated_images, training=True)

        gradients = gTape.gradient(disc_interpolates, interpolated_images)
        gradients += 1e-8
        #slopes = tf.sqrt(tf.math.reduce_sum(tf.math.square(gradients), 1))
        slopes = tf.sqrt(tf.math.reduce_sum(tf.math.square(gradients), axis=np.arange(1, len(gradients.shape))))

        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        return gradient_penalty

    def calc_penalty(self, fake_data, real_data, discriminator, args):
        if args.disc_penalty == 'wgan-gp':
            return self.wasserstein_gp(fake_data, real_data, discriminator)
        elif args.disc_penalty == 'none':
            return 0
        else:
            raise NotImplementedError()


class GeneratorPenalties:
    def weight_regularizer(self, g1, g2, shared_layers):
        distance = 0
        for idx in range(len(g1.trainable_variables)):
            distance = distance + tf.reduce_mean(tf.math.squared_difference(g1.trainable_variables[idx], g2.trainable_variables[idx]))
        return distance

    def feature_regularizer(self, g1_batch, g2_batch, shared_layers):
        distance = 0
        for idx in range(shared_layers):
            distance = distance + tf.reduce_mean(tf.math.squared_difference(g1_batch[idx], g2_batch[idx]))
        return distance

    def calc_penalty(self, g1, g2, shared_layers, args, g1_batch=None, g2_batch=None):
        if args.gen_penalty == 'weight':
            return self.weight_regularizer(g1, g2, shared_layers)
        if args.gen_penalty == 'feature':
            return self.feature_regularizer(g1_batch, g2_batch, shared_layers)
        if args.gen_penalty == 'none':
            return 0
