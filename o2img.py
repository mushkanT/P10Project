import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


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


def plot_toy_distribution(samples):
    df = pd.DataFrame(samples, columns=["x", "y"])
    sns.jointplot(x="x", y="y", data=df, kind="kde")
    plt.savefig('images/toy/ring_distribution.png')


def draw_samples_and_plot_2d(generator, epoch, n_dim, seed=2019):
    a = []
    noise = tf.random.normal([3000, n_dim], seed=seed)
    generated_image = generator(noise)

    df = pd.DataFrame(generated_image, columns=["x", "y"])
    sns.jointplot(x="x", y="y", data=df, kind="kde")
    plt.savefig('images/toy/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def plot_2d_data(path, samples):
    samples = tf.convert_to_tensor(samples)
    df = pd.DataFrame(samples, columns=["x", "y"])
    sns.jointplot(x="x", y="y", data=df, kind="kde")
    plt.savefig(path+'/image_.png')
    plt.close()


def plot_loss(gen_loss, disc_loss, path):
    gen_loss = tf.convert_to_tensor(gen_loss)
    disc_loss = tf.convert_to_tensor(disc_loss)
    plt.plot(gen_loss, label='Generator loss')
    plt.plot(disc_loss, label='Discriminator loss')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path+'/loss.png')
    plt.close()


def load_images(path, dataset):
    for folder in os.listdir(path):
        folder_path = path+'/'+str(folder)
        itw_data = np.load(folder_path + '/itw.npy')
        d_loss = np.load(folder_path + '/d_loss.npy')
        g_loss = np.load(folder_path + '/g_loss.npy')
        produce_images_itw(dataset, folder_path, itw_data)
        produce_images_loss(folder_path, d_loss, g_loss)


def produce_images_itw(dataset, folder_path, data):
    if dataset == 'toy':
        save_path = folder_path + '/images_itw'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        plot_2d_data(save_path, data)
    elif dataset == 'mnist':
        # data = data.reshape([16, 28, 28, 1])
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
            counter = counter + 1
        plt.close()


def produce_images_loss(folder_path, g_loss, d_loss):
    save_path = folder_path + '/images_loss'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    plot_loss(g_loss, d_loss, save_path)
