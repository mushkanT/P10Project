import tensorflow as tf
import vq_vae_model
import numpy as np
import DataHandler
import argparse
import os
from tensorflow.python.keras.models import Model, load_model

def train_step(data, optimizer, model):
    with tf.GradientTape() as tape:
        output = model.model(data)
        recon_error = tf.reduce_mean((data - output[0]) ** 2)
        quant_top = output[1]
        quant_bottom = output[2]
        mean_latent_loss = (quant_top['loss'] + quant_bottom['loss']) / 2
        loss = recon_error + mean_latent_loss

    trainable_variables = model.model.trainable_variables
    grads = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    return {'loss': loss, 'mean_latent_loss': mean_latent_loss, 'recon_error': recon_error, 'x_recon': output[0]}


def train_loop(optimizer, print_n_batches, epochs, dataset, model):
    train_losses = []
    train_recon_errors = []
    train_vqvae_loss = []
    train_recons = []
    for i in range(epochs):
        for iteration, batch in enumerate(dataset):
            train_results = train_step(batch, optimizer, model)

            train_losses.append(train_results['loss'])
            train_recon_errors.append(train_results['recon_error'])
            train_vqvae_loss.append(train_results['mean_latent_loss'])

            if iteration % print_n_batches == 0:
                train_recons.append([train_results['x_recon'][:16], batch[:16]])
                print('%d. train loss: %f ' % (0 + iteration, np.mean(train_losses)) +
                        ('recon_error: %.6f ' % np.mean(train_recon_errors)) +
                        ('vqvae loss: %.6f' % np.mean(train_vqvae_loss)))

    return [train_losses, train_recon_errors, train_vqvae_loss, train_recons]


def train_vq_vae(optimizer, image_size, output_path, print_n_batches, epochs, batch_size, data_path):
    SECTION = 'VQvae'
    RUN_FOLDER = output_path
    RUN_FOLDER += SECTION + '/'
    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
    channels = 1 if data_path == 'mnist' else 3

    #Build or load model
    model = vq_vae_model.VQVAEModel(image_size, channels)
    model.build_model()

    #Get dataset for training
    dataset = DataHandler.get_dataset(batch_size=batch_size, data_name=data_path, pad_to_32=True)

    #Perform training
    train_metrics = train_loop(optimizer, print_n_batches, epochs, dataset, model)

    # Save losses and np images
    loss_file = os.path.join(RUN_FOLDER, 'loss')
    r_loss_file = os.path.join(RUN_FOLDER, 'r_loss')
    vq_loss_file = os.path.join(RUN_FOLDER, 'vq_loss')
    recons_file = os.path.join(RUN_FOLDER, 'recons')

    print('Saving loss and model...')
    np.save(loss_file, train_metrics[0])
    np.save(r_loss_file, train_metrics[1])
    np.save(vq_loss_file, train_metrics[2])
    np.save(recons_file, train_metrics[3])

    #save model
    model_file = os.path.join(RUN_FOLDER, 'model')
    os.mkdir(model_file)
    model.model.save(model_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='Can be mnist|cifar10')
    parser.add_argument('--lr', type=float, default='2e-4', help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--print_n_batches', type=int, default=1200, help='Prints status every n\'th batch.')
    parser.add_argument('--img_size', type=int, default=32, help='Size of images in the given dataset. NxN')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--run_id', type=str, help='ID of current run')
    parser.add_argument('--run_folder', type=str, help='folder that contains run generated items (images, weights etc.)')

    args = parser.parse_args()

    print(args)


    train_vq_vae(
        tf.keras.optimizers.Adam(learning_rate=args.lr),
        image_size=args.img_size,
        batch_size=args.batch_size,
        data_path=args.dataset,
        epochs=args.epochs,
        output_path=args.run_folder,
        print_n_batches=args.print_n_batches
    )
