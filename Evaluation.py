import pickle
import os
import dnnlib
from dnnlib.tflib import tfutil
import numpy as np

def init_tf(random_seed=1234):
    """Initialize TF."""
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
    with open('inception_v3_features.pkl','rb') as file:
        net = pickle.load(file)
    return net

def evaluate(input_image):
    init_tf()

    feature_net = initialize_feature_extractor()

    feature = feature_net.run(input_image, num_gpus=1, assume_frozen=True)

    print(feature)