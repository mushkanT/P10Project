import tensorflow as tf
import time
from Code import Utils as u, Losses as l, Penalties as p
import matplotlib.pyplot as plt
import os

TINY = 1e-8


class GANTrainer(object):

    def __init__(self,
                 generator,
                 discriminator,
                 dataset
                 ):
        self.generator = generator
        self.discriminator = discriminator
        self.dataset = dataset
        self.gen_loss = []
        self.disc_loss = []
        self.images_while_training = []
        self.full_training_time = 0
        self.discPenal = p.DiscriminatorPenalties()
        self.d_loss_fn = None
        self.g_loss_fn = None

    def train_discriminator(self, real_data, args):
        noise = u.gen_noise(args)
        generated_images = self.generator(noise, training=True)

        with tf.GradientTape() as disc_tape:
            fake_output = self.discriminator(generated_images, training=True)
            real_output = self.discriminator(real_data, training=True)
            disc_loss = self.d_loss_fn(fake_output, real_output)
            gp = self.discPenal.calc_penalty(generated_images, real_data, self.discriminator, args)  # if loss is not wgan-gp then gp=0
            disc_loss = disc_loss + (gp * args.penalty_weight_d)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        args.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Clip weights if wgan loss function
        if args.loss == "wgan":
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -args.clip, args.clip))
        return disc_loss

    def train_generator(self, args):
        noise = u.gen_noise(args)

        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self.g_loss_fn(fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        args.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return gen_loss

    def train(self, args):
        if args.dataset != 'lsun':
            it = iter(self.dataset)
        else:
            it = self.dataset

        # Set loss functions
        self.d_loss_fn, self.g_loss_fn = l.set_losses(args)

        for epoch in range(args.epochs):
            start = time.time()
            disc_iters_loss = []

            # take x steps with disc before training generator
            for i in range(args.disc_iters):
                if args.dataset in ['celeba', 'lsun']:
                    batch = next(it)
                else:
                    batch = next(it)[0]

                d_loss = self.train_discriminator(batch, args)
                disc_iters_loss.append(d_loss)

            g_loss = self.train_generator(args)

            self.full_training_time += time.time() - start
            self.disc_loss.append(tf.reduce_mean(disc_iters_loss).numpy())
            self.gen_loss.append(g_loss.numpy())
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss,))

            # Generate samples and save
            if args.images_while_training != 0:
                if epoch % args.images_while_training == 0:
                    if args.dataset == "toy":
                        self.images_while_training.append(u.draw_2d_samples(self.generator, args.noise_dim))
                    else:
                        self.sample_images(epoch, args.seed, args.dir, args.dataset_dim[3])

        self.plot_losses(args.dir, self.disc_loss, self.gen_loss)
        self.sample_images(epoch, args.seed, args.dir, args.dataset_dim[3])
        return self.full_training_time

    def sample_images(self, epoch, seed, dir, channels):
        r, c = 2, 4
        gen_batch1 = self.generator.predict(seed)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_batch1 + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        # black/white images
        if channels == 1:
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig(os.path.join(dir, "images/%d.png" % epoch))
            plt.close()
        # color images
        else:
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig(os.path.join(dir, "images/%d.png" % epoch))
            plt.close()

    def plot_losses(self, dir, d_loss, gen_loss):
        plt.plot(gen_loss, label='Generator loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(dir, 'losses/gen_loss.png'))
        plt.close()

        plt.plot(d_loss, label='Discriminator loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(dir, 'losses/disc_loss.png'))
        plt.close()
