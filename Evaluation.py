import pickle
import os
import dnnlib
from dnnlib.tflib import tfutil
import numpy as np
from precision_recall import knn_precision_recall_features



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
