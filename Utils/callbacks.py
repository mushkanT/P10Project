from keras.callbacks import Callback, LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
import os


#### CALLBACKS
class CustomCallback(Callback):

    def __init__(self, run_folder, print_every_n_batches, initial_epoch, vae):
        self.epoch = initial_epoch
        self.run_folder = run_folder
        self.print_every_n_batches = print_every_n_batches
        self.vae = vae

    def on_batch_end(self, batch, logs={}):
        if batch % self.print_every_n_batches == 0:
            fig = plt.figure(figsize=(4, 4))
            z_new = np.random.normal(size=(16, self.vae.z_dim))
            reconst = self.vae.decoder.predict(z_new)
            for i in range(z_new.shape[0]):
                plt.subplot(4, 4, i+1)
                if reconst.shape[3] == 1:
                    plt.imshow(reconst[i, :, :, 0], cmap='gray')
                else:
                    plt.imshow(reconst[i, :, :, :])
                plt.axis('off')

            filepath = os.path.join(self.run_folder, 'images',
                                    'img_' + str(self.epoch).zfill(3) + '_' + str(batch) + '.png')
            print(len(reconst.shape))
            plt.savefig(filepath)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1


def step_decay_schedule(initial_lr, decay_factor=0.5, step_size=1):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''

    def schedule(epoch):
        new_lr = initial_lr * (decay_factor ** np.floor(epoch / step_size))

        return new_lr

    return LearningRateScheduler(schedule)