import numpy as np
import tensorflow as tf
import argparse
import os
import Data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='Path to directory containing models')
    parser.add_argument('--out_dir', type=str, help='Path to create image outputs')
    parser.add_argument('--num_samples', type=int, help='Number of samples to be generated from each model')
    args = parser.parse_args()
    args.cogan_data = 'Smiling'
    args.batch_size = 50000
    x1, x2, shape = Data.select_dataset_cogan(args)

    it1 = iter(x1)
    it2 = iter(x2)

    batch1 = next(it1)
    batch2 = next(it2)

    batch1 = batch1.numpy()
    batch2 = batch2.numpy()


    np.save(os.path.join(args.out_dir, "celebaNoSmile_50K_samples"), batch1)
    np.save(os.path.join(args.out_dir, "smiling_30K_samples"), batch2)
'''
    gen1 = tf.keras.models.load_model(os.path.join(args.model_dir, 'generator1'))
    gen2 = tf.keras.models.load_model(os.path.join(args.model_dir, 'generator2'))

    noise_dim = gen1.inputs[0].shape[1]

    noise = tf.random.normal([args.num_samples, noise_dim])

    samples1 = gen1.predict(noise)
    samples2 = gen2.predict(noise)

    #samples1 = samples1[-1]
    #samples2 = samples2[-1]


    np.save(os.path.join(args.out_dir, "samples1"), samples1)
    np.save(os.path.join(args.out_dir, "samples2"), samples2)
'''

