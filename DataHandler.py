import tensorflow as tf


def mnist(normalise=True, norm_setting=0):
    (x_train, x_test), (_, _) = tf.keras.datasets.mnist.load_data()
    if normalise:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        if norm_setting == 0:
            x_train /= 255.0
            x_test /= 255.0
        elif norm_setting == 1:
            x_train = (x_train - 127.5) / 127.5
            x_test = (x_test - 127.5) / 127.5
    return x_train,x_test


def cifar10(normalise=True, norm_setting=0):
    (x_train, x_test), (_, _) = tf.keras.datasets.cifar10.load_data()
    if normalise:
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
