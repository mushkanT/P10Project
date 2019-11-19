import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def plot_distribution(samples, title='', cmap='Blues',x="x",y="y"):

    df = pd.DataFrame(samples, columns=["x", "y"])
    sns.jointplot(x="x", y="y", data=df, kind="kde")
    #plt.savefig('ring_distribution.png')
    plt.show()

    ''' # From 'Are all GANS created equal':
    samples = samples.cpu().numpy()
    sns.set(font_scale=2)
    f, ax = plt.subplots(figsize=(4, 4))
    sns.kdeplot(samples[:, 0], samples[:, 1], cmap=cmap, ax=ax, n_levels=20, shade=True)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.axis('off')
    plt.title(title)
    plt.show()'''


def draw_samples_and_plot_2d(generator, epoch, n_dim):
    a = []
    for c in range(2000):
        noise = tf.random.normal([1, n_dim])
        generated_image = generator(noise, training=False)
        a.append(generated_image)

    samples = tf.convert_to_tensor(tf.reshape(a,[2000,2]))

    df = pd.DataFrame(samples, columns=["x", "y"])
    sns.jointplot(x="x", y="y", data=df, kind="kde")
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()

def weight_init():
    return 0