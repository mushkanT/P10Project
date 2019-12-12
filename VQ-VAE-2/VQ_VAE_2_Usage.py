import tensorflow as tf
import VQ_VAE_Model
import numpy as np
from Utils import DataHandler


def train_step(data, optimizer, model):
    with tf.GradientTape() as tape:
        model_output = model(data, is_training=True)
    trainable_variables = model.trainable_variables
    grads = tape.gradient(model_output['loss'], trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    return model_output

def train_loop(optimizer, num_images, batch_size, epochs, train_data, model, data_generator=None):
    train_losses = []
    train_recon_errors = []
    train_vqvae_loss = []
    train_recons = []
    for i in range(epochs):
        print('Epoch %d' % (i))
        if data_generator is not None:
            data_generator.reset()
        iter_count = 0
        for begin in range(0, num_images, batch_size):
            iter_count += 1
            print('iteration - %d' % (iter_count))
            if data_generator is not None:
                train_data = next(data_generator)[0]
                train_results = train_step(train_data, optimizer, model)
            else:
                end = min(begin + batch_size, num_images)
                train_results = train_step(train_data[begin:end], optimizer, model)
            train_losses.append(train_results['loss'])
            train_recon_errors.append(train_results['recon_error'])
            train_vqvae_loss.append(train_results['mean_latent_loss'])
            if iter_count % 100 == 0:
                train_recons.append(train_results['x_recon'])
                print('%d. train loss: %f ' % (0 + 1,
                                               np.mean(train_losses[-100:])) +
                      ('recon_error: %.3f ' % np.mean(train_recon_errors[-100:])) +
                      ('vqvae loss: %.3f' % np.mean(train_vqvae_loss[-100:])))

    return train_losses, train_recon_errors, train_vqvae_loss, train_recons


def train_vq_vae(optimizer, image_size, epochs=500, batch_size=100, data_path='mnist'):
    model = VQ_VAE_Model.VQVAEModel(image_size)


    if data_path == 'mnist':
        train_data, test_data = DataHandler.mnist()
        train_data = tf.pad(train_data, [[0,0], [2,2], [2,2], [0,0]])
        num_images = train_data.shape[0]
        return train_loop(optimizer, num_images, batch_size, epochs, train_data, model)
    elif data_path == 'cifar10':
        train_data, test_data = DataHandler.cifar10()
        num_images = train_data.shape[0]
        return train_loop(optimizer, num_images, batch_size, epochs, train_data, model)
    else:
        data_generator = DataHandler.custom_data(data_path, batch_size, (image_size, image_size))
        num_images = data_generator.n
        return train_loop(optimizer, num_images, batch_size, epochs, None, model, data_generator=data_generator)

    model.save('VQ_VAE_001.h5')


if __name__ == '__main__':
    train_vq_vae(tf.keras.optimizers.Adam(learning_rate=1e-4), 1024, batch_size=1, data_path='C:/users/user/desktop/1024_images/')

