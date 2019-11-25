import tensorflow as tf

class datahandler():
    def __init__(self, batch_size, normalise=True, Shuffle=True):


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

def custom_data(path, normalise=True, norm_setting=0):
    if normalise:
        if norm_setting == 0:
            image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

            train_data_gen = image_generator.flow_from_directory(directory=str(path),
                                                         batch_size=1,
                                                         target_size=(1024, 1024),
                                                         shuffle=True)