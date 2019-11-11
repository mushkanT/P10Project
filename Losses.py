import tensorflow as tf
import keras.losses as loss

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss_CE(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss_CE(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)



def wasserstein():
    return 0

def wasserstein_gp():
    return wasserstein()-1