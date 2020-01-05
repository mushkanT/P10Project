import tensorflow as tf
import numpy as np
from PixelSNAIL import PixelSNAIL
import os


if __name__ == '__main__':
    batch = 16

    model = PixelSNAIL([32,32],256, 128, 5, 4, 2, 128)
    dummy_input = tf.zeros([1,32,32], dtype=tf.int64)
    model(dummy_input)
    model.load_weights('/user/student.aau.dk/palmin15/mnist_model/')

    row = tf.Variable(tf.zeros([batch,32,32], dtype=tf.int64))
    cache = {}

    for i in range(32):
        print('Row' + str(i))
        for j in range(32):
            out, cache = model(row[:, : i + 1, :])
            prob = tf.nn.softmax(out[:, i, j, :] / 1.0, 1)
            sample = tf.random.categorical(prob, 1)
            sample = tf.squeeze(sample, axis=-1)
            row[:, i, j].assign(sample)

    print('Done')
    np.save('/user/student.aau.dk/palmin15/mnist_samples', row.numpy())