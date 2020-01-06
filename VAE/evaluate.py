import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import VAE_Model

def dummy_loss(y_true, y_pred):
    return y_true

if __name__ == '__main__':

    model = tf.keras.models.load_model('c:/users/user/desktop/output_data/3727vae/3727_MNIST/model', custom_objects={'vae_loss', tf.keras.losses.binary_crossentropy})
    print(model.summary)
    nx = ny = 15
    x_values = np.linspace(.05, .95, nx)
    y_values = np.linspace(.05, .95, ny)

    canvas = np.empty((28 * ny, 28 * nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[norm.ppf(xi), norm.ppf(yi)]]).astype('float32')
            x_mean = model.decode_new(z_mu)
            canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()