import numpy as np
import tensorflow as tf
import argparse
import os
import Data
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='Path to directory containing models')
    parser.add_argument('--out_dir', type=str, help='Path to create image outputs')
    parser.add_argument('--num_samples', type=int, help='Number of samples to be generated from each model')
    args = parser.parse_args()

    '''
    args.cogan_data = 'apple2orange'
    args.batch_size = 1000

    x1, x2, shape = Data.select_dataset_cogan(args)
    it1 = iter(x1)
    it2 = iter(x2)

    batch1 = next(it1)[0]
    batch2 = next(it2)[0]

    batch1 = batch1.numpy()
    batch2 = batch2.numpy()

    np.save('C:/users/palmi/desktop/lpips/apple_big_1000.npy', batch1)
    np.save('C:/users/palmi/desktop/lpips/orange_big_1000.npy', batch2)

    r, c = 1, 2
    gen_batch1 = np.load('C:/users/palmi/desktop/cyclegan_datasets/percep_a2o/samples1.npy')
    gen_batch2 = np.load('C:/users/palmi/desktop/cyclegan_datasets/percep_a2o/samples2.npy')

    gen_batch1 = gen_batch1[:950]
    gen_batch2 = gen_batch2[:950]

    gen_batch1 = 0.5 * gen_batch1 + 0.5
    gen_batch2 = 0.5 * gen_batch2 + 0.5

    img11 = gen_batch1[841]
    img12 = gen_batch2[841]
    img21 = gen_batch1[234]
    img22 = gen_batch2[234]

    #cnt = 0
    fig, axs = plt.subplots(r, c)
    #for i in range(r):
    #    for j in range(c):
    #        if i < 2:
    #            axs[i, j].imshow(gen_batch1[cnt, :, :, :])
    #            axs[i, j].axis('off')
    #            cnt += 1
    #        else:
    #            axs[i, j].imshow(gen_batch2[cnt-8, :, :, :])
    #            axs[i, j].axis('off')
    #            cnt += 1
    #plt.show()
    #fig.savefig(os.path.join(dir, "images/%d.png" % epoch))
    #plt.close()

    axs[0].imshow(img21)
    axs[0].axis('off')
    axs[1].imshow(img22)
    axs[1].axis('off')
    plt.show()
    #fig.savefig('C:/users/palmi/desktop/comp1_svhn_3.png')
    #plt.close()

    '''
    args.model_dir = 'G:/experiments/40602/'
    args.out_dir = 'G:/experiments/40602'
    args.num_samples = 50000
    imgs1 = np.empty((args.num_samples,32,32,3),dtype=np.float32)
    imgs2 = np.empty((args.num_samples,32,32,3),dtype=np.float32)



    gen1 = tf.keras.models.load_model(os.path.join(args.model_dir, 'generator1'))
    gen2 = tf.keras.models.load_model(os.path.join(args.model_dir, 'generator2'))

    noise_dim = gen1.inputs[0].shape[1]

    div_num = 1000
    rg = (int)(args.num_samples / div_num)
    for idx in range(div_num):
        noise = tf.random.uniform(shape=(rg, noise_dim), minval=-1., maxval=1.)
        #noise = tf.random.normal(shape=(rg,noise_dim))

        samples1 = gen1(noise)[-1]
        samples2 = gen2(noise)[-1]
        for sample_idx in range(samples1.shape[0]):
            imgs1[sample_idx + (50 * idx)] = samples1[sample_idx]
            imgs2[sample_idx + (50 * idx)] = samples2[sample_idx]

    #imgs1 = np.asarray(imgs1)
    #imgs2 = np.asarray(imgs2)
    np.save(os.path.join(args.out_dir, "samples1"), imgs1)
    np.save(os.path.join(args.out_dir, "samples2"), imgs2)