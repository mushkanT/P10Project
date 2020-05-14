from __future__ import print_function, division
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import numpy as np
import os
import Penalties as p
import Losses as l
import Nets as n


class CoGANTrainer(object):

    def __init__(self, g1, g2, d1, d2, domain1, domain2):
        self.hist_g1 = []
        self.hist_g2 = []
        self.hist_d1 = []
        self.hist_d2 = []
        self.hist_weight_similarity = []
        self.X1 = domain1
        self.X2 = domain2
        self.full_training_time = 0
        self.discPenal = p.DiscriminatorPenalties()
        self.genPenal = p.GeneratorPenalties()

        self.d1, self.d2 = d1, d2
        self.g1, self.g2 = g1, g2


    def train(self, args):
        self.encoder = n.encoder(args)
        if args.semantic_loss:
            self.classifier = tf.keras.models.load_model(args.classifier_path)

        it1 = iter(self.X1)
        it2 = iter(self.X2)

        # Set loss functions
        d_loss_fn, g_loss_fn = l.set_losses(args)

        for epoch in range(args.epochs):
            start = time.time()

            # ----------------------
            #  Train Discriminators
            # ----------------------

            for i in range(args.disc_iters):
                noise = tf.random.normal([args.batch_size, args.noise_dim])
                # Select a random batch of images
                if args.cogan_data in ['mnist2edge', 'Eyeglasses']:
                    batch1 = next(it1)
                    batch2 = next(it2)
                else:
                    batch1 = next(it1)[0]
                    batch2 = next(it2)[0]

                # Sample noise as generator input
                noise = tf.random.normal([args.batch_size, args.noise_dim])

                # Generate a batch of new images
                gen_batch1 = self.g1(noise, training=True)

                # d1
                with tf.GradientTape() as tape:
                    # Disc response
                    disc_real1 = self.d1(batch1, training=True)
                    disc_fake1 = self.d1(gen_batch1[-1], training=True)

                    # Calc loss and penalty
                    d1_loss = d_loss_fn(disc_fake1, disc_real1)
                    gp1 = self.discPenal.calc_penalty(gen_batch1[-1], batch1, self.d1, args)  # if loss is not wgan-gp then gp=0
                    d1_loss = d1_loss + (gp1 * args.penalty_weight_d)
                gradients_of_discriminator = tape.gradient(d1_loss, self.d1.trainable_variables)
                args.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.d1.trainable_variables))

                # Generate a batch of new images
                gen_batch2 = self.g2(noise, training=True)

                # d2
                with tf.GradientTape() as tape:
                    # Disc response
                    disc_real2 = self.d2(batch2, training=True)
                    disc_fake2 = self.d2(gen_batch2[-1], training=True)

                    # Calc loss and penalty
                    d2_loss = d_loss_fn(disc_fake2, disc_real2)
                    gp2 = self.discPenal.calc_penalty(gen_batch2[-1], batch2, self.d2, args)  # if loss is not wgan-gp then gp=0
                    d2_loss = d2_loss + (gp2 * args.penalty_weight_d)
                gradients_of_discriminator = tape.gradient(d2_loss, self.d2.trainable_variables)
                args.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.d2.trainable_variables))

                if args.loss == 'wgan' and args.disc_penalty == 'none':
                    self.clip_weights(args.clip)

            # ------------------
            #  Train Generators
            # ------------------
            noise = tf.random.normal([args.batch_size, args.noise_dim])
            
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3:
                # Adv loss
                gen1_fake = self.g1(noise, training=True)
                disc1_fake = self.d1(gen1_fake[-1], training=True)
                g1_loss = g_loss_fn(disc1_fake)

                gen2_fake = self.g2(noise, training=True)
                disc2_fake = self.d2(gen2_fake[-1], training=True)
                g2_loss = g_loss_fn(disc2_fake)
                
                if args.semantic_loss:
                    domain1_pred = self.classifier(gen1_fake[-1])
                    domain2_pred = self.classifier(gen2_fake[-1])
                    diff = tf.reduce_mean(tf.math.squared_difference(domain1_pred, domain2_pred))
                    g1_loss = g1_loss + diff * args.semantic_weight
                    g2_loss = g2_loss + diff * args.semantic_weight

                penalty = self.genPenal.calc_penalty(self.g1, self.g2, 4, args, gen1_fake, gen2_fake)
                g1_loss = g1_loss + (penalty * args.penalty_weight_g)
                g2_loss = g2_loss + (penalty * args.penalty_weight_g)

                if args.use_cycle:
                    # Recon loss
                    noise_recon1 = self.encoder(gen1_fake[-1])
                    noise_recon2 = self.encoder(gen2_fake[-1])

                    fake_recon1 = self.g1(noise_recon1, training=False)
                    fake_recon2 = self.g2(noise_recon2, training=False)

                    noise_recon_loss1 = l.recon_criterion(noise_recon1, noise)
                    noise_recon_loss2 = l.recon_criterion(noise_recon2, noise)

                    fake_recon_loss1 = l.recon_criterion(fake_recon1[-1], gen1_fake[-1])
                    fake_recon_loss2 = l.recon_criterion(fake_recon2[-1], gen2_fake[-1])

                    total_recon_loss = noise_recon_loss1 + noise_recon_loss2

                    g1_loss = g1_loss + total_recon_loss
                    g2_loss = g2_loss + total_recon_loss

            gradients_of_generator1 = tape1.gradient(g1_loss, self.g1.trainable_variables)
            args.gen_optimizer.apply_gradients(zip(gradients_of_generator1, self.g1.trainable_variables))
            gradients_of_generator2 = tape2.gradient(g2_loss, self.g2.trainable_variables)
            args.gen_optimizer.apply_gradients(zip(gradients_of_generator2, self.g2.trainable_variables))
            if args.use_cycle:
                gradients_of_encoder = tape3.gradient(total_recon_loss, self.encoder.trainable_variables)
                args.gen_optimizer.apply_gradients(zip(gradients_of_encoder, self.encoder.trainable_variables))
            weight_sim = self.genPenal.weight_regularizer(self.g1, self.g2, 21)
            self.full_training_time += time.time() - start

            '''
            # Check if shared weights are equal between generators
            a = self.g1.trainable_variables
            b = self.g2.trainable_variables
            mask = []

            for i in range(8):
                if np.array_equal(a[i].numpy(), b[i].numpy()):
                    mask.append(1)
                else:
                    mask.append(0)
            if 0 in mask:
                print("ERROR - weight sharing failure:" + mask)
            '''

            # Collect loss values
            self.hist_d1.append(d1_loss)
            self.hist_d2.append(d2_loss)
            self.hist_g1.append(g1_loss)
            self.hist_g2.append(g2_loss)
            self.hist_weight_similarity.append(weight_sim)

            print("%d [D1 loss: %f] [D2 loss: %f] [G1 loss: %f] [G2 loss: %f] [WeightSim: %f]}" % (epoch, d1_loss, d2_loss, g1_loss, g2_loss, weight_sim))

            # If at save interval => save generated image samples
            if epoch % args.images_while_training == 0:
                self.sample_images(epoch, args.seed, args.dir, args.dataset_dim[3])
        self.plot_losses(args.dir)
        return self.full_training_time

    def sample_images(self, epoch, seed, dir, channels):
        r, c = 4, 4
        gen_batch1 = self.g1.predict(seed)[-1]
        gen_batch2 = self.g2.predict(seed)[-1]

        gen_imgs = np.concatenate([gen_batch1, gen_batch2])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
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

    def plot_losses(self, dir):
        plt.plot(self.hist_g1, label='Generator 1 loss')
        plt.plot(self.hist_g2, label='Generator 2 loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(dir, 'losses/gen_loss.png'))
        plt.close()

        plt.plot(self.hist_d1, label='Discriminator 1 loss')
        plt.plot(self.hist_d2, label='Discriminator 2 loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(dir, 'losses/disc_loss.png'))
        plt.close()

        plt.plot(self.hist_weight_similarity, label='weight differences')
        plt.xlabel('Epochs')
        plt.ylabel('Difference')
        plt.legend()
        plt.savefig(os.path.join(dir, 'losses/weight_diff.png'))
        plt.close()

    def clip_weights(self, clip):
        for i, var in enumerate(self.d1.trainable_variables):
            self.d1.trainable_variables[i].assign(tf.clip_by_value(var, -clip, clip))
            #if not np.array_equiv(self.d1.trainable_variables[i].numpy(), self.d2.trainable_variables[i].numpy()):
                #print(i)
        for i, var in enumerate(self.d2.trainable_variables[6:]):
            self.d2.trainable_variables[i + 6].assign(tf.clip_by_value(var, -clip, clip))
            #if not np.array_equiv(self.d1.trainable_variables[i].numpy(), self.d2.trainable_variables[i].numpy()):
                #print(i)
