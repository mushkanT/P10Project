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


def plot_2d_data(path, samples, nr):
    samples = tf.convert_to_tensor(samples)
    df = pd.DataFrame(samples, columns=["x", "y"])
    sns.jointplot(x="x", y="y", data=df, kind="kde")
    plt.savefig(path+'/image_'+str(nr)+'.png')
    plt.close()


def plot_loss(gen_loss, disc_loss, path):
    plt.plot(gen_loss, label='Generator loss')
    plt.plot(disc_loss, label='Discriminator loss')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path+'/loss.png')
    plt.close()


def create_images(path, dataset):
    for folder in os.listdir(path):
        folder_path = path+'/'+str(folder)
        for file in os.listdir(folder_path):
            file_path = folder_path+'/'+str(file)
            if file == 'losses.txt' or file == 'itw.txt':
                open_file = open(file_path)
                data = open_file.readline()
                data = data.split('|')
                fix_formats_and_produce_images(file, data, dataset, folder_path)


def fix_formats_and_produce_images(file, data, dataset, folder_path):
    if file == 'itw.txt':
        final_list = []
        for i in data:
            new_list = []
            i = i.replace('[ ', '')
            i = i.replace(' ]', '')
            i = i.replace('[', '')
            i = i.replace(']', '')
            i = i.split(',')
            for k in i:
                k = k.split(' ')
                temp_list = []
                for x in k:
                    if x != '':
                        temp_list.append(float(x))
                new_list.append(temp_list)
            final_list.append(new_list[:-1])
        produce_images_itw(dataset, folder_path, file, final_list)

    elif file == 'losses.txt':
        g_loss = []
        d_loss = []
        data[0] = data[0].replace(']', '')
        data[0] = data[0].replace('[', '')
        data[1] = data[1].replace(']', '')
        data[1] = data[1].replace('[', '')
        data[0] = data[0].split(',')
        data[1] = data[1].split(',')
        for i in range(len(data[0])):
            g_loss.append(float(data[0][i]))
            d_loss.append(float(data[1][i]))
        produce_images_loss(dataset, folder_path, file, g_loss, d_loss)


def produce_images_itw(dataset, folder_path, file, final_list):
    if dataset == 'toy':
        save_path = folder_path + '/images_' + file
        os.mkdir(save_path)
        counter = 0
        for i in final_list:
            a = np.array(i)
            plot_2d_data(save_path, a, counter)
            counter = counter + 1


def produce_images_loss(dataset, folder_path, file, g_loss, d_loss):
    if dataset == 'toy':
        save_path = folder_path + '/images_' + file
        os.mkdir(save_path)
        plot_loss(g_loss, d_loss, save_path)
