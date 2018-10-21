import numpy as np
import os
import pickle

class ModelOutput(object):
    def __init__(self, model_path):
        self.model_path = model_path

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    @property
    def history_path(self):
        return os.path.join(self.model_path, 'history.npz')

    @property
    def checkpt_path(self):
        return os.path.join(self.model_path, 'checkpts')

    @property
    def converged(self):
        if not os.path.exists(self.history_path):
            return False
        with np.load(self.history_path) as data:
            return bool(data['converged'])

    def load_history(self):
        return np.load(self.history_path)

    def save_history(self, train_loss_history, converged=False):
        np.savez(self.history_path,
                 train_loss_history=train_loss_history,
                 converged=converged)

    def get_epoch_checkpt_paths(self, epoch):
        checkpt_path = self.checkpt_path
        if not os.path.exists(checkpt_path):
            os.makedirs(checkpt_path)

        # Things saved by the model but not by tensorflow
        model_state_path = os.path.join(checkpt_path, 'model_state_{:d}'.format(epoch))
        # Things saved by tensorflow, especially model weights
        model_ckpt_path = os.path.join(checkpt_path, 'model_params_{:d}.ckpt'.format(epoch))
        # Dataset batching order state
        dataset_state_path = os.path.join(checkpt_path, 'dataset_state_{:d}'.format(epoch))

        return model_state_path, model_ckpt_path, dataset_state_path

    def save_checkpt(self, epoch, model, dataset):
        model_state_path, model_ckpt_path, dataset_state_path = \
            self.get_epoch_checkpt_paths(epoch)

        model.save_ckpt(model_ckpt_path)
        with open(model_state_path, 'w') as f:
            pickle.dump(model.get_state(), f)
        with open(dataset_state_path, 'w') as f:
            pickle.dump(dataset.get_state(), f)

    def load_checkpt(self, epoch, model, dataset):
        model_state_path, model_ckpt_path, dataset_state_path = \
            self.get_epoch_checkpt_paths(epoch)

        model.load_ckpt(model_ckpt_path)
        with open(model_state_path, 'r') as f:
            model.set_state(pickle.load(f))
        with open(dataset_state_path, 'r') as f:
            dataset.set_state(pickle.load(f))

