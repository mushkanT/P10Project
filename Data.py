import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from scipy.io import loadmat
import scipy
import cv2
import matplotlib.pyplot as plt
import glob


def select_dataset_gan(args):
    shape = None
    if args.dataset == "toy":
        dat = createToyDataRing()
        # o2i.plot_toy_distribution(dat)
        train_dat = tf.data.Dataset.from_tensor_slices(dat).shuffle(dat.shape[0]).batch(args.batch_size).repeat()
    elif args.dataset == "mnist":
        data, info = tfds.load('mnist', with_info=True, as_supervised=True)
        X2, test = data['train'], data['test']
        X2 = X2.map(format_example_to32)
        num_examples = info.splits['train'].num_examples
        train_dat = X2.shuffle(num_examples).batch(args.batch_size).repeat()
    elif args.dataset == "svhn":
        data, info = tfds.load('svhn_cropped', with_info=True, as_supervised=True)
        X2, test = data['train'], data['test']
        X2 = X2.map(format_example_to32)
        num_examples = info.splits['train'].num_examples
        train_dat = X2.shuffle(num_examples).batch(args.batch_size).repeat()
    elif args.dataset == "apple2orange":
        data, info = tfds.load('cycle_gan/'+args.dataset, with_info=True, as_supervised=True)
        X1, X2 = data['trainA'], data['trainB']
        X1 = X1.concatenate(X2)
        X1 = X1.map(format_example_to128_2)
        num_examples = info.splits['trainA'].num_examples
        shape = (None,X1.element_spec[0].shape[0],X1.element_spec[0].shape[1],X1.element_spec[0].shape[2])
        train_dat = X1.shuffle(num_examples).batch(args.batch_size).repeat()
    elif args.dataset == "celeba":
        images = glob.glob('C:/Users/marku/Desktop/img_align_celeba/*.jpg')
        #images = glob.glob('/user/student.aau.dk/mjuuln15/img_align_celeba/*.jpg')
        dataset = []
        for i in images:
            image = plt.imread(i)
            dataset.append(image)
        dataset = np.array(dataset)
        X1 = dataset.reshape(dataset.shape[0], dataset.shape[1], dataset.shape[2], dataset.shape[3]).astype('float32')
        X1 = tf.data.Dataset.from_tensor_slices(X1).shuffle(len(X1)).batch(args.batch_size).repeat()
        train_dat = X1.map(format_example_to128)
        shape = train_dat.element_spec.shape
    elif args.dataset == "mnist-f":
        dat = mnist_f(args.input_scale, args.limit_dataset)
        if args.scale_data != 0:
            dat = tf.image.resize(dat, [args.scale_data, args.scale_data])
        train_dat = tf.data.Dataset.from_tensor_slices(dat).shuffle(dat.shape[0]).batch(args.batch_size).repeat()
    elif args.dataset == 'cifar10':
        dat = cifar10(args.input_scale, args.limit_dataset)
        if args.scale_data != 0:
            dat = tf.image.resize(dat, [args.scale_data, args.scale_data])
        if args.grayscale:
            dat = tf.image.rgb_to_grayscale(dat)
        train_dat = tf.data.Dataset.from_tensor_slices(dat).shuffle(dat.shape[0]).batch(args.batch_size).repeat()
    elif args.dataset == 'lsun':
        ImgDataGen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess, dtype=tf.dtypes.float32)
                
        train_dat = ImgDataGen.flow_from_directory('/user/student.aau.dk/mjuuln15/lsun_data/', target_size=(args.scale_data, args.scale_data), batch_size=args.batch_size, seed=2019, class_mode=None, interpolation="nearest")
               
        amount = 2554932
        shape = (amount, train_dat.image_shape[0], train_dat.image_shape[1], train_dat.image_shape[2])
    elif args.dataset == 'frey':
        img_size = (28, 20, 1)
        # data = loadmat('/user/student.aau.dk/mjuuln15/frey_rawface.mat')
        data = loadmat(args.dir + '/frey_rawface.mat')
        data = data['ff']
        data = data.transpose()
        data = data.reshape((-1, *img_size))
        data = np.pad(data, [(0, 0), (2, 2), (6, 6), (0, 0)], 'constant')
        # dat = data / 255.
        dat = (data - 127.5) / 127.5
        train_dat = tf.data.Dataset.from_tensor_slices(dat).shuffle(dat.shape[0]).batch(args.batch_size).repeat()
    else:
        raise NotImplementedError()
    if args.dataset in ['lsun']:
        shape = dat.shape
    return train_dat, shape


def select_dataset_cogan(args):
    # Same dataset
    if args.cogan_data in ['mnist2edge', 'mnist2rotate', 'mnist2negative']:
        X1, X2 = mnist_cogan(args.batch_size, args.cogan_data)
        if args.cogan_data == 'mnist2edge':
            shape = X1.element_spec.shape
        else:
            shape = X1.element_spec[0].shape
    elif args.cogan_data == 'mnist2svhn':
        # Domain 1
        data, info = tfds.load('mnist', with_info=True, as_supervised=True)
        X1, test = data['train'], data['test']
        X1 = X1.map(format_example_g2rgb)
        num_examples = info.splits['train'].num_examples
        X1 = X1.shuffle(num_examples).batch(args.batch_size).repeat()

        # Domain 2
        data, info = tfds.load('svhn_cropped', with_info=True, as_supervised=True)
        X2, test = data['train'], data['test']
        X2 = X2.map(format_example_to32)
        num_examples = info.splits['train'].num_examples
        X2 = X2.shuffle(num_examples).batch(args.batch_size).repeat()
        shape = X1.element_spec[0].shape
    elif args.cogan_data in ['apple2orange', 'horse2zebra', 'vangogh2photo']:
        # Domains
        data, info = tfds.load('cycle_gan/'+args.cogan_data, with_info=True, as_supervised=True)
        X1, X2 = data['trainA'], data['trainB']
        X1 = X1.map(format_example_scale)
        X2 = X2.map(format_example_scale)
        num_examples = info.splits['trainA'].num_examples

        X1 = X1.shuffle(num_examples).batch(args.batch_size).repeat()
        X2 = X2.shuffle(num_examples).batch(args.batch_size).repeat()
        shape = (None, X1.element_spec[0].shape[0], X1.element_spec[0].shape[1], X1.element_spec[0].shape[2])
    elif args.cogan_data in ['Eyeglasses']:
        #lines = [line.rstrip() for line in open('C:/Users/marku/Desktop/list_attr_celeba.txt', 'r')]
        lines = [line.rstrip() for line in open('/user/student.aau.dk/mjuuln15/list_attr_celeba.txt', 'r')]
        all_attr_names = lines[1].split()
        attr2idx = {}
        idx2attr = {}
        mask = []
        dataset = []
        for i, attr_name in enumerate(all_attr_names):
            attr2idx[attr_name] = i
            idx2attr[i] = attr_name
        lines = lines[2:]
        for i, line in enumerate(lines):
            split = line.split()
            values = split[1:]
            for attr_name in ['Eyeglasses']:
                idx = attr2idx[attr_name]
                label = (values[idx] == '1')
            mask.append(label)

        #images = glob.glob('C:/Users/marku/Desktop/img_align_celeba/*.jpg')
        images = glob.glob('/user/student.aau.dk/mjuuln15/img_align_celeba/*.jpg')
        for i in images:
            image = plt.imread(i)
            dataset.append(image)

        mask = np.array(mask)
        dataset = np.array(dataset)
        X1 = dataset[mask]
        X2 = dataset[np.invert(mask)]

        X1 = X1.reshape(X1.shape[0], X1.shape[1], X1.shape[2], X1.shape[3]).astype('float32')
        X2 = X2.reshape(X2.shape[0], X2.shape[1], X2.shape[2], X2.shape[3]).astype('float32')

        X1 = tf.data.Dataset.from_tensor_slices(X1).shuffle(len(X1)).batch(args.batch_size).repeat()
        X2 = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(X2)).shuffle(len(X2)).batch(args.batch_size).repeat()

        X1 = X1.map(format_example_to128)
        X2 = X2.map(format_example_to128)
        shape = X2.element_spec.shape

    return X1, X2, shape


def format_example_g2rgb(image, label):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = (image - 127.5) / 127.5
    # Resize the image
    image = tf.image.resize(image, (32, 32))
    image = tf.image.grayscale_to_rgb(image)
    return (image, label)


def format_example_edge(image, label):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = (image - 127.5) / 127.5
    # Resize the image
    image = tf.image.resize(image, (32, 32))
    image = np.squeeze(image)
    dilation = cv2.dilate(image, np.ones((3, 3), np.uint8), iterations=1)
    image = dilation - image
    return (image, label)


def format_example_rotate(image, label):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = (image - 127.5) / 127.5
    # Resize the image
    image = tf.image.rot90(image)
    image = tf.image.resize(image, (32, 32))
    return (image, label)


def format_example_to32(image, label):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = (image - 127.5) / 127.5
    # Resize the image
    image = tf.image.resize(image, (32, 32))
    return (image, label)


def format_example_to128(image):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = (image - 127.5) / 127.5
    # Resize the image
    image = tf.image.resize(image, (128, 128))
    return image


def format_example_to128_2(image,label):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = (image - 127.5) / 127.5
    # Resize the image
    image = tf.image.resize(image, (128, 128))
    return image,label


def format_example_scale(image, label):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = (image - 127.5) / 127.5
    # Resize the image
    return (image, label)


def format_example_negative(image, label):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = (image - 127.5) / 127.5
    # Resize the image
    image = tf.image.resize(image, (32, 32))
    image = tf.math.negative(image)
    return (image, label)


# Regular GAN data loaders
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


def mnist(input_scale, restrict=False):
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
    if input_scale:
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    else:
        train_images = train_images / 255 # Normalize the images to [0, 1]    
    return train_images


def mnist_f(input_scale, restrict=False):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
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
    if input_scale:
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    else:
        train_images = train_images / 255 # Normalize the images to [0, 1]
    return train_images


def cifar10(input_scale, restrict=False):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    if restrict:
        train_mask = [y[0] == 8 for y in train_labels]
        test_mask = [y[0] == 8 for y in test_labels]
        train_images = train_images[train_mask]
        test_images = test_images[test_mask]
    train_images = np.concatenate([train_images, test_images])
    train_images.astype('float32')
    if input_scale:
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    else:
        train_images = train_images / 255 # Normalize the images to [0, 1]
    return train_images


def preprocess(img):
    img = (img - 127.5) / 127.5
    return img


# CoGAN data loaders
def mnist_cogan(batch_size, data):
    d2 = data.split('2')[1]

    if d2 in ['rotate', 'negative']:
        X1, info = tfds.load('mnist', split='train[:50%]', with_info=True, as_supervised=True)
        X2, info2 = tfds.load('mnist', split='train[50%:]', with_info=True, as_supervised=True)
        X1 = X1.map(format_example_to32)
        if d2 == 'rotate':
            X2 = X2.map(format_example_rotate)
        else:
            X2 = X2.map(format_example_negative)
        num_examples = info.splits['train'].num_examples
        num_examples2 = info2.splits['train'].num_examples
        X1 = X1.shuffle(num_examples).batch(batch_size).repeat()
        X2 = X2.shuffle(num_examples2).batch(batch_size).repeat()
    elif d2 == 'edge':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        # 28x28 -> 32x32
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        padding = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
        train_images = tf.pad(train_images, padding, "CONSTANT")

        # Split dataset
        X1 = train_images[:int(train_images.shape[0] / 2)]
        X2 = train_images[int(train_images.shape[0] / 2):]

        edges = np.zeros((X2.shape[0], 32, 32, 1))
        for idx, i in enumerate(X2):
            i = np.squeeze(i)
            dilation = cv2.dilate(i, np.ones((3, 3), np.uint8), iterations=1)
            edge = dilation - i
            edges[idx - X2.shape[0], :, :, 0] = edge
        X2 = tf.convert_to_tensor(edges)

        X1 = (X1 - 127.5) / 127.5  # Normalize the images to [-1, 1]
        X2 = (X2 - 127.5) / 127.5  # Normalize the images to [-1, 1]

        X1 = tf.data.Dataset.from_tensor_slices(X1).shuffle(X1.shape[0]).batch(
            batch_size).repeat()
        X2 = tf.data.Dataset.from_tensor_slices(X2).shuffle(X2.shape[0]).batch(
            batch_size).repeat()
    return X1, X2
