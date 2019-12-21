import tensorflow as tf
import numpy as np
import os


def mnist(restrict=-1, norm_setting=0, pad_to_32=False, fashion=False):
    if fashion:
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    if restrict != -1:
        train_mask = [y[0] == restrict for y in train_labels]
        test_mask = [y[0] == restrict for y in test_labels]
        train_images = train_images[train_mask]
        test_images = test_images[test_mask]
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
    elif norm_setting is not None:
        raise NotImplementedError('Only supports norm_setting 0|1')
    return train_images

def get_dataset(batch_size, data_name='mnist', restrict=-1, pad_to_32=False, norm_setting=0, shuffle=True, drop_remainder=True, target_size=32):
    if data_name == 'mnist':
        dat = mnist(restrict, norm_setting, pad_to_32)
    elif data_name == 'mnist_fashion':
        dat = mnist(restrict, norm_setting, pad_to_32, True)
    elif data_name == 'cifar10':
        dat = cifar10(restrict,norm_setting)
    elif os.path.exists(data_name):
        return custom_data(data_name, batch_size,target_size,shuffle,norm_setting=norm_setting)



    dataset = tf.data.Dataset.from_tensor_slices(dat)
    if shuffle:
        dataset = dataset.shuffle(dat.shape[0])
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset




def get_encodings(batch_size, shuffle, drop_remainder, path):
    data = np.load(path)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        dataset = dataset.shuffle(data.shape[0])
    return dataset.batch(batch_size, drop_remainder=drop_remainder)





def cifar10(restrict=-1, norm_setting=0):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    if restrict != -1:
        train_mask = [y[0] == restrict for y in train_labels]
        test_mask = [y[0] == restrict for y in test_labels]
        train_images = train_images[train_mask]
        test_images = test_images[test_mask]
        train_images = np.concatenate([train_images, test_images])
    if norm_setting == 0:
        train_images /= 255.0  # Normalize to [0,1]
    elif norm_setting == 1:
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    elif norm_setting is not None:
        raise NotImplementedError('Only supports norm_setting 0|1')
    return train_images

def custom_data(path, batch_size, target_size, shuffle=True, norm_setting=0):
    if norm_setting == 0:
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.0)

    elif norm_setting == 1:
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=(1. - 127.5) / 127.5)

    elif norm_setting is None:
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator()
    else:
        raise NotImplementedError('Only support norm_setting = 0|1|None')

    train_data_gen = image_generator.flow_from_directory(directory=str(path),
                                                         batch_size=batch_size,
                                                         target_size=target_size,
                                                         shuffle=shuffle)
    return train_data_gen
