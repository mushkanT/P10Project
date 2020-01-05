import tensorflow as tf
import numpy as np
import os
from PixelSNAIL import PixelSNAIL
import DataHandler


def train(model, dataset, optimizer):
    losses = []
    cross_entropy = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    for i, batch in enumerate(dataset):
        batch = tf.cast(tf.squeeze(batch, axis=3), tf.int64)
        with tf.GradientTape() as tape:
            out = model(batch)
            loss = cross_entropy(batch, out)
            print(loss)
            losses.append(loss)
        trainable_variables = model.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))

    return losses


if __name__ == '__main__':
    epoch = 5
    dataset = DataHandler.get_dataset(batch_size=32, data_name='mnist_fashion', pad_to_32=True)

    model = PixelSNAIL([32,32], 256, 128, 5, 2, 4, 128)

    optimizer = tf.optimizers.Adam(0.0002)

    for i in range(epoch):
        train(model, dataset, optimizer)

    model.save_weights('home/palminde/Desktop/mnist_model/', save_format='tf')
