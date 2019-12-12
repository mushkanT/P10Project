import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from keras.utils.vis_utils import plot_model


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).

    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig('images/generated/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


def draw_samples_and_plot_2d(generator, epoch, n_dim, seed=2019):
    a = []
    noise = tf.random.normal([3000, n_dim], seed=seed)
    generated_image = generator(noise)

    df = pd.DataFrame(generated_image, columns=["x", "y"])
    sns.jointplot(x="x", y="y", data=df, kind="kde")
    plt.savefig('images/toy/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def plot_toy_distribution(dat):
    df = pd.DataFrame(dat, columns=["x", "y"])
    sns.jointplot(x="x", y="y", data=df, kind="kde")
    plt.savefig('C:/Users/marku/Desktop/GAN_training_output/old/Toy_distribution.png')
    plt.close()


def plot_2d_data(path, samples, epoch_interval):
    samples = tf.convert_to_tensor(samples)
    counter = 0
    for i in range(samples.shape[0]):
        df = pd.DataFrame(samples[i], columns=["x", "y"])
        sns.jointplot(x="x", y="y", data=df, kind="kde")
        plt.savefig(path+'/image_'+str(counter)+'.png')
        counter = counter + epoch_interval
        plt.close()


def plot_loss(gen_loss, disc_loss, path):
    gen_loss = tf.convert_to_tensor(gen_loss)
    disc_loss = tf.convert_to_tensor(disc_loss)

    # Plot both
    plt.plot(gen_loss, label='Generator loss')
    plt.plot(disc_loss, label='Discriminator loss')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path+'/loss_both.png')
    plt.close()

    # Plot gen
    plt.plot(gen_loss, label='Generator loss')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path+'/loss_gen.png')
    plt.close()

    # Plot disc
    plt.plot(disc_loss, label='Discriminator loss', color='tab:orange')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path+'/loss_disc.png')
    plt.close()


def load_images(path):
    for folder in os.listdir(path):
        if folder == 'old':
            continue
        folder_path = path+'/'+str(folder)
        config_file = open(folder_path + '/config.txt', 'r').read()
        dataset = config_file.split(',')[3].split('\'')[1]
        epoch_interval = config_file.split(',')[8].split('=')[1]
        itw_data = np.load(folder_path + '/itw.npy')
        d_loss = np.load(folder_path + '/d_loss.npy')
        g_loss = np.load(folder_path + '/g_loss.npy')
        produce_images_itw(dataset, folder_path, itw_data, int(epoch_interval))
        produce_images_loss(folder_path, d_loss, g_loss)


def produce_images_itw(dataset, folder_path, data, epoch_interval):
    if dataset == 'toy':
        save_path = folder_path + '/images_itw'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        plot_2d_data(save_path, data, epoch_interval)
    elif dataset == 'mnist':
        save_path = folder_path + '/images_itw'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        counter = 0
        for x in data:
            for i in range(x.shape[0]):
                plt.subplot(4, 4, i + 1)
                plt.imshow(x[i, :, :, 0]*127.5+127.5, cmap='gray')
                plt.axis('off')
            plt.savefig(save_path + '/itw_' + str(counter) + '.png')
            counter = counter + epoch_interval
        plt.close()

    elif dataset == 'cifar10':
        save_path = folder_path + '/images_itw'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        counter = 0
        for x in data:
            for i in range(x.shape[0]):
                plt.subplot(4, 4, i + 1)
                plt.imshow(x[i, :, :, :])
                plt.axis('off')
            plt.savefig(save_path + '/itw_' + str(counter) + '.png')
            counter = counter + epoch_interval
        plt.close()


def produce_images_loss(folder_path, g_loss, d_loss):
    save_path = folder_path + '/images_loss'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    plot_loss(g_loss, d_loss, save_path)

def test_trunc_trick(args):
    seed = tf.random.normal([args.num_samples_to_gen, args.noise_dim])
    truncated_seed = tf.random.truncated_normal([args.num_samples_to_gen, args.noise_dim])

    gen = tf.keras.models.load_model('C:/Users/marku/Desktop/GAN_training_output/old/2224/generator')
    noise = tf.random.normal([5000, 100])
    mean_noise = tf.reduce_mean(noise, axis=0)

    for z in range(2,10,2):
        z=z/10
        #z_hat=[]
        #for v in range(truncated_seed.shape[0]):
        z_hat=(mean_noise + z * (seed - mean_noise))
        truncated_images = gen(z_hat)
        for i in range(truncated_images.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(truncated_images[i, :, :, 0] * 127.5 + 127.5)
            plt.axis('off')
        plt.savefig('C:/Users/marku/Desktop/GAN_training_output/old/trunc_images'+str(z)+'.png')
        plt.close()

    images = gen(seed)
    #truncated_images = gen(truncated_seed)

    for i in range(images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')
    plt.savefig('C:/Users/marku/Desktop/GAN_training_output/old/truncTrickTest/reg_images.png')
    plt.close()
