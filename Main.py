import tensorflow as tf
import numpy as np
from Evaluation import evaluate
import matplotlib.pyplot as plt

(train_images, train_labels) , (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

train_mask = [y[0] == 8 for y in train_labels]
test_mask = [y[0] == 8 for y in test_labels]

train_images = train_images[train_mask]
test_images = test_images[test_mask]

sess = tf.Session()

real_images = np.concatenate([train_images, test_images])
#real_images = real_images / 255.

with sess.as_default():
    real_images = tf.transpose(real_images, perm=[0,3,1,2]).eval()

generated_images = np.load('c:/users/palmi/desktop/gen_images.npy')

generated_images = (generated_images+1)/2

generated_images = (generated_images * 255).astype(int)


do_print = False

if do_print:
    for i in range(100):
        for h in range(25):
                plt.subplot(5, 5, h+1)
                plt.imshow(generated_images[h*(i+1)])
                plt.axis('off')

        plt.show()



with sess.as_default():
    generated_images = tf.transpose(generated_images, perm=[0,3,1,2]).eval()

#generated_images = np.concatenate([real_images[:3000],generated_images[:3000]])


result = evaluate(real_images,generated_images, batch_size=50, feature_model=1)


print('Recall (Variance): ' + str(result['recall'][0]))
print('Precision (quality): '+ str(result['precision'][0]))