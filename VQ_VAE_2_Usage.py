import tensorflow as tf
import VQ_VAE_2
import sonnet as snt
import numpy as np
import matplotlib.pyplot as plt

# Set hyper-parameters.
batch_size = 32
image_size = 32

# 100k steps should take < 30 minutes on a modern (>= 2017) GPU.
num_training_updates = 100000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
# These hyper-parameters define the size of the model (number of parameters and layers).
# The hyper-parameters in the paper were (For ImageNet):
# batch_size = 128
# image_size = 128
# num_hiddens = 128
# num_residual_hiddens = 32
# num_residual_layers = 2

# This value is not that important, usually 64 works.
# This will not change the capacity in the information-bottleneck.
embedding_dim = 64

# The higher this value, the higher the capacity in the information bottleneck.
num_embeddings = 512

# commitment_cost should be set appropriately. It's often useful to try a couple
# of values. It mostly depends on the scale of the reconstruction cost
# (log p(x|z)). So if the reconstruction cost is 100x higher, the
# commitment_cost should also be multiplied with the same amount.
commitment_cost = 0.25

# Use EMA updates for the codebook (instead of the Adam optimizer).
# This typically converges faster, and makes the model less dependent on choice
# of the optimizer. In the VQ-VAE paper EMA updates were not used (but was
# developed afterwards). See Appendix of the paper for more details.
vq_use_ema = True

# This is only used for EMA updates.
decay = 0.99

learning_rate = 3e-4

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data_gen = image_generator.flow_from_directory(directory='C:/Users/User/Desktop/1024_images/',
                                                     batch_size=1,
                                                     target_size=(1024,1024),
                                                     shuffle=True)

h = next(train_data_gen)
(x_train, y_train), (x_test, y_test) =tf.keras.datasets.cifar10.load_data()
# Data Loading.

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train=h[0]

# # Build modules.
encoder = VQ_VAE_2.Encoder8Stride(num_hiddens, num_residual_layers, num_residual_hiddens)
decoder = VQ_VAE_2.Decoder(num_hiddens, num_residual_layers, num_residual_hiddens,2)
pre_vq_conv1 = snt.Conv2D(output_channels=embedding_dim,
                          kernel_shape=(1, 1),
                          stride=(1, 1),
                          name="to_vq")

if vq_use_ema:
    vq_vae = snt.nets.VectorQuantizerEMA(
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        decay=decay)
else:
    vq_vae = snt.nets.VectorQuantizer(
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost)

model = VQ_VAE_2.VQVAEModel(encoder, decoder, vq_vae, pre_vq_conv1,
                   data_variance=0.063369)

optimizer = snt.optimizers.Adam(learning_rate=learning_rate)


#@tf.function
def train_step(data):
    with tf.GradientTape() as tape:
        model_output = model(data, is_training=True)
    trainable_variables = model.trainable_variables
    plt.imshow(model_output['x_recon'][0])
    plt.show()
    grads = tape.gradient(model_output['loss'], trainable_variables)
    optimizer.apply(grads, trainable_variables)

    return model_output


train_losses = []
train_recon_errors = []
train_perplexities = []
train_vqvae_loss = []

for i in range(10000):
    train_results = train_step(x_train[0:1])
    train_losses.append(train_results['loss'])
    train_recon_errors.append(train_results['recon_error'])
    train_perplexities.append(train_results['vq_output']['perplexity'])
    train_vqvae_loss.append(train_results['vq_output']['loss'])

    if i % 100 == 0:
        print('%d. train loss: %f ' % (0 + 1,
                                       np.mean(train_losses[-100:])) +
              ('recon_error: %.3f ' % np.mean(train_recon_errors[-100:])) +
              ('perplexity: %.3f ' % np.mean(train_perplexities[-100:])) +
              ('vqvae loss: %.3f' % np.mean(train_vqvae_loss[-100:])))
