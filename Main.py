
import tensorflow as tf
import numpy as np
import keras as K
import Nets as nets
import Data as dt
import Train as t

# Settings
args = {}
args["gen_optimizer"] = tf.keras.optimizers.Adam(1e-3)
args["disc_optimizer"] = tf.keras.optimizers.Adam(1e-3)
args["loss"]                            = "wgan-gp"  # ce, wgan
args["batch_size"]                      = 256
args["epochs"]                          = 500
args["n_critic"]                        = 2  # update critic 'n_critic' times pr gen update
args["noise_dim"]                       = 10
args["buffer_size"]                     = 60000
args["num_of_samples_to_generate"]   = 16
args["lambda"]                          = 10  # wgan-gp penalty scaling factor

# Toy data set
dat = dt.createToyDataRing()
toy_train_dataset = tf.data.Dataset.from_tensor_slices(dat).shuffle(args["buffer_size"]).batch(args["batch_size"])
#dt.plot_distribution(dat, "Toy data distribution")

# Mnist
'''
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
'''

# Nets
generator = nets.generator_toy()
discriminator = nets.discriminator_toy()

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([args["num_of_samples_to_generate"], args["noise_dim"]])

if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    with tf.device('/GPU:0'):
        t.train(toy_train_dataset, discriminator, generator, args)
else:
    t.train(toy_train_dataset, discriminator, generator, args)


'''
#Test generator and discriminator - before training.
generator = nets.generator_dcgan()

noise = tf.random.normal([1, 10])
generated_image = generator(noise, training=False)

plt.imshow(generated_image, cmap='gray')
plt.show()

discriminator = nets.discriminator_dcgan()
decision = discriminator(generated_image)
print (decision)
'''