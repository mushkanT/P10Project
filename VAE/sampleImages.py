import tensorflow as tf
import numpy as np
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--zdim', type=int, default=100)
    args = parser.parse_args()
    n_samples = args.samples
    z = np.random.normal(size=(n_samples, args.zdim))
    model = tf.keras.models.load_model(args.path + 'model', compile=False)
    decoder = model.get_layer('model_3')
    out = decoder.predict(z)
    np.save(args.path + str(args.samples) +'_samples', out)



