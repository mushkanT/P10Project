
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_images = train_images / 255.0

test_images = test_images / 255.0
# Display image 55 from test set
plt.figure()
plt.imshow(train_images[55])
plt.colorbar()
plt.grid(False)
plt.show()




