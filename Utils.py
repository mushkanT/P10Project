import tensorflow as tf
from seaborn import jointplot
import matplotlib.pyplot as plt
import pandas as pd
from keras import backend as K


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).

    predictions = model(test_input, training=False)
    return predictions.numpy()

    '''
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig('images/generated/image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()
    '''

'''
def plot_toy_distribution(samples, title='', cmap='Blues',x="x",y="y"):

    df = pd.DataFrame(samples, columns=["x", "y"])
    jointplot(x="x", y="y", data=df, kind="kde")
    plt.savefig('images/toy/ring_distribution.png')

    # From 'Are all GANS created equal':
    samples = samples.cpu().numpy()
    sns.set(font_scale=2)
    f, ax = plt.subplots(figsize=(4, 4))
    sns.kdeplot(samples[:, 0], samples[:, 1], cmap=cmap, ax=ax, n_levels=20, shade=True)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.axis('off')
    plt.title(title)
    plt.show()
'''

def draw_samples_and_plot_2d(generator, epoch, n_dim, seed=2019):
    a = []

    noise = tf.random.normal([3000, n_dim], seed=seed)
    generated_image = generator(noise).numpy()
    a.append(generated_image)
    return generated_image

    '''
    samples = tf.convert_to_tensor(tf.reshape(a, [3000, 2]))
    df = pd.DataFrame(samples, columns=["x", "y"])
    jointplot(x="x", y="y", data=df, kind="kde")
    plt.savefig('images/toy/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()
    '''

'''
def plot_loss(gen_loss, disc_loss):
    plt.plot(gen_loss, label='Generator loss')
    plt.plot(disc_loss, label='Discriminator loss')
    plt.legend()
    plt.savefig('images/plots/oss_plot.png')
    plt.close()
'''


def weight_init():
    return 0

