from keras.callbacks import Callback, LearningRateScheduler
import numpy as np
import os


#### CALLBACKS
class CustomCallback(Callback):

    def __init__(self, run_folder, print_every_n_batches, initial_epoch, vae):
        self.epoch = initial_epoch
        self.run_folder = run_folder
        self.print_every_n_batches = print_every_n_batches
        self.vae = vae
        self.loss = []
        self.r_loss = []
        self.kl_loss = []
        self.z = np.random.normal(size=(16, self.vae.z_dim))


    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.r_loss.append(logs.get('vae_r_loss'))
        self.kl_loss.append(logs.get('vae_kl_loss'))

        if batch % self.print_every_n_batches == 0:
            reconst = self.vae.decoder.predict(self.z)
            filepath = os.path.join(self.run_folder, 'images',
                                    'img_' + str(self.epoch).zfill(3) + '_' + str(batch) )
            np.save(filepath, reconst)


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