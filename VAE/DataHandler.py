import tensorflow as tf
import numpy as np


def mnist(restrict=False, norm_setting=0, pad_to_32=False):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    if restrict:
        selected_ix = train_labels == 7
        selected_ix_test = test_labels == 7
        train_images = train_images[selected_ix]
        test_images = test_images[selected_ix_test]
        train_images = np.concatenate([train_images, test_images])
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

    # Transform from 28x28 to 32x32
    if pad_to_32:
        padding = tf.constant([[0,0], [2,2], [2,2], [0,0]])
        train_images = tf.pad(train_images, padding, "CONSTANT")

    if norm_setting == 0:
        train_images /= 255.0 # Normalize to [0,1]
    elif norm_setting == 1:
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    else:
        raise NotImplementedError('Only supports norm_setting 0|1')
    return train_images

def get_dataset(batch_size, data_name='mnist', restrict=False, pad_to_32=False, norm_setting=0, shuffle=True, drop_remainder=True):
    if data_name == 'mnist':
        dat = mnist(restrict, norm_setting, pad_to_32)
        dataset = tf.data.Dataset.from_tensor_slices(dat)
        if shuffle:
            dataset = dataset.shuffle(dat.shape[0])
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).repeat()
        return dataset
    elif data_name == 'cifar10':
        dat = cifar10(restrict,norm_setting)


def cifar10(restrict=True, norm_setting=0):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if norm_setting == 0:
        x_train /= 255.0
        x_test /= 255.0
    elif norm_setting == 1:
        x_train = (x_train - 127.5) / 127.5
        x_test = (x_test - 127.5) / 127.5
    return x_train, x_test

def custom_data(path, batch_size, target_size, shuffle=True, normalise=True, norm_setting=0):
    if normalise:
        if norm_setting == 0:
            image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.0)

            train_data_gen = image_generator.flow_from_directory(directory=str(path),
                                                         batch_size=batch_size,
                                                         target_size=target_size,
                                                         shuffle=shuffle)
            test = next(train_data_gen)
            return train_data_gen
        elif norm_setting == 1:
            image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=(1. - 127.5) / 127.5)

            train_data_gen = image_generator.flow_from_directory(directory=str(path),
                                                                 batch_size=batch_size,
                                                                 target_size=target_size,
                                                                 shuffle=shuffle)
            return train_data_gen
