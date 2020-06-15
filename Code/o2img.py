import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import math

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
    plt.plot(gen_loss[:], label='Generator loss')
    plt.plot(disc_loss[:], label='Discriminator loss')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    #plt.xlim(0,5)
    #plt.axis([0,30000,0,5])
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


    # -----ZOOM PLOTS-----
    zoom = 12000
    # Plot both zoom
    plt.plot(gen_loss[zoom:], label='Generator loss')
    plt.plot(disc_loss[zoom:], label='Discriminator loss')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    #plt.xlim(0,5)
    #plt.axis([0,30000,0,5])
    plt.legend()
    plt.savefig(path+'/loss_both_zoom.png')
    plt.close()

    # Plot gen
    plt.plot(gen_loss[zoom:], label='Generator loss')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path+'/loss_gen_zoom.png')
    plt.close()

    # Plot disc
    plt.plot(disc_loss[zoom:], label='Discriminator loss', color='tab:orange')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path+'/loss_disc_zoom.png')
    plt.close()


def plot_acc(fakes, reals, path):
    save_path = path + '/graphs'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    fakes = tf.convert_to_tensor(fakes)
    reals = tf.convert_to_tensor(reals)

    # Plot both
    plt.plot(fakes, label='Fakes')
    plt.plot(reals, label='Reals')
    plt.xlabel('Batches')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_path+'/acc_both.png')
    plt.close()

    # Plot gen
    plt.plot(fakes, label='Fakes')
    plt.xlabel('Batches')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_path+'/acc_fakes.png')
    plt.close()

    # Plot disc
    plt.plot(reals, label='Reals', color='tab:orange')
    plt.xlabel('Batches')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_path+'/acc_reals.png')
    plt.close()


def load_images(path):
    for folder in os.listdir(path):
        if folder == 'old':
            continue
        folder_path = path+'/'+str(folder)
        config_file = open(folder_path + '/config.txt', 'r').read()

        dataset = config_file.split(',')[7].split('\'')[1]
        if dataset == 'toy':
            epoch_interval = config_file.split(',')[18].split('=')[1]
        else:
            epoch_interval = config_file.split(',')[20].split('=')[1]
        input_scale = config_file.split(',')[23].split('=')[1]

        itw_data = np.load(folder_path + '/itw.npy')
        d_loss = np.load(folder_path + '/d_loss.npy')
        g_loss = np.load(folder_path + '/g_loss.npy')
        acc_fakes = np.load(folder_path + '/acc_fakes.npy')
        acc_reals = np.load(folder_path + '/acc_reals.npy')
        produce_images_itw(dataset, folder_path, itw_data, int(epoch_interval), input_scale)
        produce_loss_graphs(folder_path, d_loss, g_loss)
        plot_acc(acc_fakes, acc_reals, folder_path)


def produce_images_itw(dataset, folder_path, data, epoch_interval, input_scale):
    if dataset == 'toy':
        save_path = folder_path + '/images_itw'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        plot_2d_data(save_path, data, epoch_interval)
    elif dataset in ['mnist', 'frey', 'mnist-f']:
        save_path = folder_path + '/images_itw'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        counter = 1
        for x in data:
            for i in range(x.shape[0]):
                plt.subplot(6, 6, i + 1)
                if input_scale == 'True':
                    plt.imshow((x[i, :, :, 0]+1)/2, cmap='gray')
                else:
                    plt.imshow((x[i, :, :, 0]), cmap='gray')
                #plt.imshow(x[i, :, :, 0]*127.5+127.5, cmap='gray')
                plt.axis('off')
            plt.savefig(save_path + '/itw_' + str(counter) + '.png')
            counter = counter + epoch_interval
        plt.close()
    elif dataset in ['lsun', 'cifar10']:
        save_path = folder_path + '/images_itw'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        counter = 0
        for x in data:
            for i in range(x.shape[0]):
                plt.subplot(6, 6, i + 1)
                if input_scale == 'True':
                    plt.imshow((x[i, :, :, :]+1)/2)
                else:
                    plt.imshow((x[i, :, :, :]))
                plt.axis('off')
            plt.savefig(save_path + '/itw_' + str(counter) + '.png')
            counter = counter + epoch_interval
        plt.close()


def produce_loss_graphs(folder_path, d_loss, g_loss):
    save_path = folder_path + '/graphs'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    plot_loss(g_loss, d_loss, save_path)


def test_trunc_trick(args):
    seed = tf.random.normal([args.num_samples_to_gen, args.noise_dim])
    truncation = 0.5
    truncated_seed = truncation * tf.random.truncated_normal([args.num_samples_to_gen, args.noise_dim])

    gen = tf.keras.models.load_model('C:/Users/marku/Desktop/GAN_training_output/3905/generator')
    # noise = tf.random.normal([10000, 100])
    # mean_noise = tf.reduce_mean(noise, axis=0)

    '''
    counter = 0
    a = gen(seed)
    for i in a:
        plt.imshow((i+1)/2)
        plt.savefig('C:/Users/marku/Desktop/GAN_training_output/old/truncTrickTest/trunc_images'+str(counter)+'.png')
        counter += 1
    '''
    for z in range(2,10,5):
        z=z/10
        #z_hat=[]
        #for v in range(truncated_seed.shape[0]):
        #z_hat=(mean_noise + z * (seed - mean_noise))
        truncated_images = gen(truncated_seed)
        for i in range(truncated_images.shape[0]):
            plt.subplot(6, 6, i + 1)
            plt.imshow((truncated_images[i, :, :, :]+1)/2)
            plt.axis('off')
        plt.savefig('C:/Users/marku/Desktop/GAN_training_output/old/truncTrickTest/trunc_images'+str(z)+'.png')
        plt.close()

    images = gen(seed)
    #truncated_images = gen(truncated_seed)

    for i in range(images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((images[i, :, :, :] +1)/2)
        plt.axis('off')
    plt.savefig('C:/Users/marku/Desktop/GAN_training_output/old/truncTrickTest/reg_images.png')
    plt.close()


def latent_walk(args, steps=16):
    gen = tf.keras.models.load_model('C:/Users/marku/Desktop/GAN_training_output/old/cifar10/3187/generator')
    # interpolate points
    p0 = args.seed[0]
    p1 = args.seed[6]
    ratios = np.linspace(0, 1, num=steps)
    vectors = []
    # linear interpolate vectors
    for ratio in ratios:
        #Spherical interpolation
        v = slerp(ratio, p0, p1)
        #Lin interpolaiton between points
        #v = (1.0 - ratio) * p0 + ratio * p1
        vectors.append(v)
    vectors = np.asarray(vectors)
    images = gen(vectors)
    images = (images+1)/2.0
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((images[i, :, :, :] + 1) / 2)
        plt.axis('off')


def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        # L'Hopital's rule/LERP
        return (1.0-val) * low + val * high
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high