import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from scipy.io import loadmat
import cv2
import matplotlib.pyplot as plt
import glob


def select_dataset_gan(args):
    shape = None
    if args.dataset == "toy":
        dat = createToyDataRing()
        # o2i.plot_toy_distribution(dat)
        train = tf.data.Dataset.from_tensor_slices(dat).shuffle(dat.shape[0]).batch(args.batch_size).repeat()

    elif args.dataset == "mnist":
        data, info = tfds.load('mnist', with_info=True, as_supervised=True)
        train, test = data['train'], data['test']
        train = train.map(format_example_to32)

    elif args.dataset == "svhn":
        data, info = tfds.load('svhn_cropped', with_info=True, as_supervised=True)
        train, test = data['train'], data['test']
        train = train.map(format_example_to32)

    elif args.dataset in ['apple2orange', 'horse2zebra', 'vangogh2photo']:
        data, info = tfds.load('cycle_gan/'+args.dataset, with_info=True, as_supervised=True)
        trainA, trainB = data['trainA'], data['trainB']
        train = trainA.concatenate(trainB)
        train = train.map(format_example_scale)
        shape = (None,256,256,3)
        num_examples = info.splits['trainA'].num_examples+info.splits['trainB'].num_examples
        train = train.shuffle(num_examples).repeat().batch(args.batch_size)

    elif args.dataset == "celeba":
        #images = glob.glob('C:/Users/marku/Desktop/img_align_celeba/*.jpg')
        images = glob.glob('/user/student.aau.dk/mjuuln15/img_align_celeba/*.jpg')
        dataset = []
        for i in images:
            image = plt.imread(i)
            dataset.append(image)
        X1 = np.array(dataset)
        num_examples = len(X1)
        X1 = tf.data.Dataset.from_tensor_slices(X1)
        train = X1.map(format_example_to128)
        train = train.shuffle(num_examples).repeat().batch(args.batch_size)
        shape = train.element_spec.shape

    elif args.dataset == "mnist-f":
        data, info = tfds.load('fashion_mnist', with_info=True, as_supervised=True)
        train, test = data['train'], data['test']
        train = train.map(format_example_to32)

    elif args.dataset == 'cifar10':
        data, info = tfds.load(args.dataset, with_info=True, as_supervised=True)
        train, test = data['train'], data['test']
        train = train.map(format_example_scale)

    elif args.dataset == 'lsun':
        images = glob.glob('/user/student.aau.dk/mjuuln15/lsun_data/bedroom*.jpg')
        dataset=[]
        for i in images:
            image = plt.imread(i)
            dataset.append(image)
        X1 = np.array(dataset)
        X1 = tf.data.Dataset.from_tensor_slices(X1).shuffle(50000).repeat().batch(args.batch_size)
        train = X1.map(format_example_to128)
        shape = (64,256,256,3)

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

    if args.dataset not in ['lsun', 'celeba', 'apple2orange', 'horse2zebra', 'vangogh2photo']:
        if args.limit_dataset:
            train = train.filter(class_filter)
        num_examples = info.splits['train'].num_examples
        train = train.shuffle(num_examples).repeat().batch(args.batch_size)

    return train, shape


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
        X1, test1 = data['train'], data['test']
        X1 = X1.map(format_example_g2rgb)
        #test1 = test1.map(format_example_g2rgb)
        num_examples = info.splits['train'].num_examples
        X1 = X1.shuffle(num_examples).repeat().batch(args.batch_size)

        # Domain 2
        data, info = tfds.load('svhn_cropped', with_info=True, as_supervised=True)
        X2, test2 = data['train'], data['test']
        X2 = X2.map(format_example_to32)
        #test2 = test2.map(format_example_to32)
        num_examples = info.splits['train'].num_examples
        X2 = X2.shuffle(num_examples).repeat().batch(args.batch_size)
        shape = X1.element_spec[0].shape

    elif args.cogan_data in ['apple2orange', 'horse2zebra', 'vangogh2photo', 'cityscapes']:
        # Domains
        data, info = tfds.load('cycle_gan/'+args.cogan_data, with_info=True, as_supervised=True)
        X1, X2 = data['trainA'], data['trainB']
        X1 = X1.map(format_example_scale)
        X2 = X2.map(format_example_scale)
        num_examples = info.splits['trainA'].num_examples

        X1 = X1.shuffle(num_examples).repeat().batch(args.batch_size)
        X2 = X2.shuffle(num_examples).repeat().batch(args.batch_size)
        shape = (None, 256, 256, 3)

    elif args.cogan_data in ['Eyeglasses']:
        lines = [line.rstrip() for line in open('C:/Users/marku/Desktop/list_attr_celeba.txt', 'r')]
        #lines = [line.rstrip() for line in open('/user/student.aau.dk/mjuuln15/list_attr_celeba.txt', 'r')]
        all_attr_names = lines[1].split()
        attr2idx = {}
        idx2attr = {}
        mask = []
        dataset = []

        for i, attr_name in enumerate(all_attr_names):
            attr2idx[attr_name] = i
            idx2attr[i] = attr_name
        lines = lines[2:]
        for i, line in enumerate(lines[:3700]):
            split = line.split()
            values = split[1:]

            for attr_name in ['Eyeglasses']:
                idx = attr2idx[attr_name]
                has_attribute = (values[idx] == '1')
            mask.append(has_attribute)

        images = glob.glob('C:/Users/marku/Desktop/img_align_celeba/*.jpg')
        #images = glob.glob('/user/student.aau.dk/mjuuln15/img_align_celeba/*.jpg')
        for i in images:
            image = plt.imread(i)
            dataset.append(image)

        mask = np.array(mask)
        dataset = np.array(dataset)
        X1 = dataset[mask]
        X2 = dataset[np.invert(mask)]
        X1_num_examples = len(X1)
        X2_num_examples = len(X2)

        X1 = tf.data.Dataset.from_tensor_slices(X1)
        X2 = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(X2))

        X1 = tf.data.Dataset.from_tensor_slices(X1)
        X2 = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(X2))

        X1 = X1.map(format_example_to128).shuffle(X1_num_examples).repeat().batch(args.batch_size)
        X2 = X2.map(format_example_to128).shuffle(X2_num_examples).repeat().batch(args.batch_size)
        shape = X2.element_spec.shape

    else:
        raise NotImplementedError()

    return X1, X2, shape


def load_celeba_data_classifier():
    #lines = [line.rstrip() for line in open('C:/Users/marku/Desktop/list_attr_celeba.txt', 'r')]
    lines = [line.rstrip() for line in open('/user/student.aau.dk/mjuuln15/list_attr_celeba.txt', 'r')]
    all_attr_names = lines[1].split()
    attr2idx = {}
    idx2attr = {}
    mask = []
    dataset = []
    labels = []

    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name
    lines = lines[2:]
    for i, line in enumerate(lines[:3700]):
        split = line.split()
        values = split[1:]

        temp_label = []
        has_attribute = False
        for attr_name in ['Arched_Eyebrows', 'Attractive', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                          'Mouth_Slightly_Open', 'No_Beard', 'Oval_Face', 'Pointy_Nose', 'Smiling', 'Wavy_Hair',
                          'Wearing_Lipstick', 'Young']:
            idx = attr2idx[attr_name]
            if not has_attribute:
                has_attribute = (values[idx] == '1')
            temp_label.append(int(values[attr2idx[attr_name]]))

        if has_attribute:
            labels.append(temp_label)
        mask.append(has_attribute)

    #images = glob.glob('C:/Users/marku/Desktop/img_align_celeba/*.jpg')
    images = glob.glob('/user/student.aau.dk/mjuuln15/img_align_celeba/*.jpg')
    for i in images:
        image = plt.imread(i)
        dataset.append(image)

    mask = np.array(mask)
    dataset = np.array(dataset)
    X1 = dataset[mask]
    cast = lambda image: tf.cast(image, tf.float32)
    scale = lambda image: (image - 127.5) / 127.5
    crop = lambda image: tf.image.central_crop(image, 0.7)
    resize = lambda image: tf.image.resize(image, [128, 128], antialias=True)
    X1 = cast(X1)
    X1 = scale(X1)
    X1 = crop(X1)
    X1 = resize(X1)

    X1 = X1[5000:]
    X2 = X1[:5000]
    L1 = np.asarray(labels[5000:])
    L2 = np.asarray(labels[:5000])
    L1[L1 == -1] = 0
    L2[L2 == -1] = 0
    X1 = [X1, L1]
    X2 = [X2, L2]

    return X1, X2

# Dataset augments
def class_filter(image, label, allowed_labels=tf.constant([1.])):
    isallowed = tf.equal(allowed_labels, tf.cast(label, tf.float32))
    reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
    return tf.greater(reduced, tf.constant(0.))


def format_example_g2rgb(image, label):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = (image - 127.5) / 127.5
    # Resize the image
    image = tf.image.resize(image, (32, 32))
    image = tf.image.grayscale_to_rgb(image)
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


def format_example_to32_2(image):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = (image - 127.5) / 127.5
    # Resize the image
    image = tf.image.resize(image, (32, 32))
    return image


def format_example_to128(image):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = (image - 127.5) / 127.5
    # Resize the image
    image = tf.image.resize(image, [128, 128])
    return image


def format_example_scale2(image):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = (image - 127.5) / 127.5
    # Resize the image
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
        X1 = X1.shuffle(num_examples).repeat().batch(batch_size)
        X2 = X2.shuffle(num_examples2).repeat().batch(batch_size)
    elif d2 == 'edge':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        # 28x28 -> 32x32
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

        # Split dataset
        X1 = train_images[:int(train_images.shape[0] / 2)]
        X2 = train_images[int(train_images.shape[0] / 2):]

        edges = np.zeros((X2.shape[0], 28, 28, 1))
        for idx, i in enumerate(X2):
            i = np.squeeze(i)
            dilation = cv2.dilate(i, np.ones((3, 3), np.uint8), iterations=1)
            edge = dilation - i
            edges[idx - X2.shape[0], :, :, 0] = edge
        X2 = tf.convert_to_tensor(edges)

        X1 = tf.data.Dataset.from_tensor_slices(X1).shuffle(X1.shape[0]).repeat().batch(
            batch_size)
        X2 = tf.data.Dataset.from_tensor_slices(X2).shuffle(X2.shape[0]).repeat().batch(
            batch_size)
        X1 = X1.map(format_example_to32_2)
        X2 = X2.map(format_example_to32_2)

    return X1, X2
