import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds


def createToyDataRing(n_mixtures=10, radius=3, Ntrain=5120, std=0.05): #50176
    delta_theta = 2 * np.pi / n_mixtures
    centers_x = []
    centers_y = []
    for i in range(n_mixtures):
        centers_x.append(radius * np.cos(i * delta_theta))
        centers_y.append(radius * np.sin(i * delta_theta))

    centers_x = np.expand_dims(np.array(centers_x), 1)
    centers_y = np.expand_dims(np.array(centers_y), 1)
    centers = np.concatenate([centers_x, centers_y], 1)

    p = [1. / n_mixtures for _ in range(n_mixtures)]

    ith_center = np.random.choice(n_mixtures, Ntrain, p=p)
    sample_centers = centers[ith_center, :]

    sample_points = np.random.normal(loc=sample_centers, scale=std).astype('float32')

    dat = tf.convert_to_tensor(sample_points)
    return dat


def mnist(restrict=False):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    if restrict:
        selected_ix = train_labels == 7
        selected_ix_test = test_labels == 7
        train_images = train_images[selected_ix]
        test_images = test_images[selected_ix_test]
        train_images = np.concatenate([train_images, test_images])
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # Transform from 28x28 to 32x32
    padding = tf.constant([[0,0], [2,2], [2,2], [0,0]])
    train_images = tf.pad(train_images, padding, "CONSTANT")
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return train_images


def cifar10(restrict=False):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    if restrict:
        train_mask = [y[0] == 8 for y in train_labels]
        test_mask = [y[0] == 8 for y in test_labels]
        train_images = train_images[train_mask]
        test_images = test_images[test_mask]
        train_images = np.concatenate([train_images, test_images])
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [0, 1]
    return train_images


def lsun(restrict=False):
    train_dataset = tfds.load(name="lsun/bedroom")
    #train_dataset = tfds.load(name="mnist")['train']
    return train_dataset
