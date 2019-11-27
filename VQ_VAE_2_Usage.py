import tensorflow as tf
import VQ_VAE_2
import sonnet as snt
import numpy as np
import matplotlib.pyplot as plt
import DataHandler


train_generator = DataHandler.custom_data('C:/Users/user/Desktop/1024_images/',1,(1024,1024))

x_train = next(train_generator)

model = VQ_VAE_2.VQVAEModel(1024)
optimizer = snt.optimizers.Adam(learning_rate=1e-4)

#@tf.function
def train_step(data):
    with tf.GradientTape() as tape:
        model_output = model(data, is_training=True)
    trainable_variables = model.trainable_variables
    grads = tape.gradient(model_output['loss'], trainable_variables)
    optimizer.apply(grads, trainable_variables)

    return model_output


train_losses = []
train_recon_errors = []
train_vqvae_loss = []

for i in range(10000):
    train_results = train_step(x_train[0])
    train_losses.append(train_results['loss'])
    train_recon_errors.append(train_results['recon_error'])
    train_vqvae_loss.append(train_results['mean_latent_loss'])

    if i % 100 == 0:
        print('%d. train loss: %f ' % (0 + 1,
                                       np.mean(train_losses[-100:])) +
              ('recon_error: %.3f ' % np.mean(train_recon_errors[-100:])) +
              ('vqvae loss: %.3f' % np.mean(train_vqvae_loss[-100:])))
