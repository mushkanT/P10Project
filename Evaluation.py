import pickle
import os
import dnnlib
from dnnlib.tflib import tfutil
import numpy as np
from precision_recall import knn_precision_recall_features
import argparse
import tensorflow as tf



def init_tf(random_seed=1234):
    """Initialize TF"""
    print('Initializing TensorFlow...\n')
    np.random.seed(random_seed)
    tfutil.init_tf({'graph_options.place_pruned_graph': True,
                    'gpu_options.allow_growth': True})


def initialize_feature_extractor():
    """Load VGG-16 network pickle (returns features from FC layer with shape=(n, 4096))."""
    print('Initializing VGG-16 model...')
    url = 'https://drive.google.com/uc?id=1fk6r8vetqpRShtEODXm9maDytbMkHLfa' # vgg16.pkl
    with dnnlib.util.open_url(url, cache_dir=os.path.join('run', '_cache')) as f:
        _, _ , net = pickle.load(f)
    return net


def initialize_feature_extractor_incept():
    print('Initializing InceptionV3 model...')
    with open('c:/users/palmi/desktop/inception_v3_features.pkl','rb') as file:
        net = pickle.load(file)
    return net


def evaluate(real_images, generated_images, batch_size=100, feature_model=0):
    """
    Evaluates precision and recall of estimated manifold on two sets of image samples
    :param real_images: Tensor of real image samples NCHW
    :param generated_images: Tensor of generated image samples NCHW
    :param batch_size: Number of features calculated at a time (limited by GPU ram)
    :param feature_model: Determines whether to use VGG16 (arg 0) or inceptionV3(arg 1)
    :return: state: Dictionary containing precision and recall metrics
    """
    # Ensures that same amount of images in both samples
    assert(real_images.shape[0] == generated_images.shape[0])

    init_tf()

    num_images = real_images.shape[0]

    if feature_model == 0:
        feature_net = initialize_feature_extractor()
    elif feature_model == 1:
        feature_net = initialize_feature_extractor_incept()

    # Calculate features vectors for real image samples
    print('Calculating feature vectors for real images...')
    ref_features = np.zeros([num_images, feature_net.output_shape[1]], dtype=np.float32)
    for begin in range(0, num_images, batch_size):
        end = min(begin + batch_size, num_images)
        ref_features[begin:end] = feature_net.run(real_images[begin:end], num_gpus=1, assume_frozen=True)

    # Calculate feature vectors for generated image samples
    print('Calculating feature vector for generated images...')
    eval_features = np.zeros([num_images, feature_net.output_shape[1]], dtype=np.float32)
    for begin in range(0, num_images, batch_size):
        end = min(begin+batch_size, num_images)
        eval_features[begin:end] = feature_net.run(generated_images[begin:end], num_gpus=1, assume_froze=True)

    print('Estimating manifold of feature vectors and calculating precision/recall...')
    state = knn_precision_recall_features(ref_features, eval_features)
    return state

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--norm_setting', type=int, help='0 for 0-1 normalising, 1 for -1,1 normalising')
    parser.add_argument('--sample_size', type=int, default=50000, help='Number of sample used for manifold estimation')
    parser.add_argument('--dataset', type=str, help='mnist|cifar10|freyface|lsun')
    parser.add_argument('--datapath', type=str, help='path to real datasets in case of dataset=freyface|lsun')
    parser.add_argument('--gen_data', type=str, help='')
    parser.add_argument('--feature_net', type=str, default='vgg', help='feature extractor - options: vgg|incepv3')
    parser.add_argument('--mask', type=int, default=None, help='Optional mask for cifar dataset option')

    args = parser.parse_args()

    if args.dataset == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    elif args.dataset == 'cifar10':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()


    #Mask images by labels -> Create a single class
    if args.mask is not None:
        train_mask = [y[0] == args.mask for y in train_labels]
        test_mask = [y[0] == args.mask for y in test_labels]
        train_images = train_images[train_mask]
        test_images = test_images[test_mask]

    #Get sample size images or atleast all train+test images
    if train_images.shape[0] >= args.sample_size:
        train_images = train_images[:args.sample_size]
    else:
        train_images = np.concatenate([train_images, test_images])
        end = min(train_images.shape[0], args.sample_size)
        train_images = train_images[:end]



    #(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    real_images = np.expand_dims(train_images,-1)
    # real_images = real_images / 255.


    sess = tf.Session()
    with sess.as_default():
        real_images = tf.transpose(real_images, perm=[0, 3, 1, 2]).eval()

    generated_images = np.load(args.gen_data)
    if args.norm_setting == 1:
        generated_images = (generated_images + 1) / 2
    generated_images = (generated_images * 255).astype(int)

    with sess.as_default():
        generated_images = tf.transpose(generated_images, perm=[0, 3, 1, 2]).eval()

    result = evaluate(real_images, generated_images, batch_size=50, feature_model=0)

    print('Recall (Variance): ' + str(result['recall'][0]))
    print('Precision (quality): ' + str(result['precision'][0]))
