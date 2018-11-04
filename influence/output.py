import numpy as np
import os
import pickle
import logging

class ModelOutput(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.logger = logging.getLogger("ModelOutput")

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
        with self.load_history() as data:
            return bool(data['converged'])

    @property
    def checkpt_list_path(self):
        return os.path.join(self.checkpt_path, 'checkpt_list.npz')

    @property
    def last_checkpt_epoch(self):
        checkpt_list = self.load_checkpt_list()
        if len(checkpt_list) == 0:
            return None
        return checkpt_list[-1]

    def log(self, message):
        self.logger.info(message)

    def load_checkpt_list(self):
        checkpt_list_path = self.checkpt_list_path
        if not os.path.exists(checkpt_list_path):
            return []
        return list(np.load(checkpt_list_path)['checkpt_list'])

    def update_checkpt_list(self, epoch):
        cp_list = set(self.load_checkpt_list())
        cp_list.add(epoch)
        np.savez(self.checkpt_list_path, checkpt_list=list(cp_list))
        self.log("Updated checkpoint list with epoch {}".format(epoch))

    def load_history(self):
        return np.load(self.history_path)

    def save_history(self, train_loss_history, test_loss_history, converged=False):
        np.savez(self.history_path,
                 train_loss_history=train_loss_history,
                 test_loss_history=test_loss_history,
                 converged=converged)
        self.log("Saved history, converged={}".format(converged))

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

        self.update_checkpt_list(epoch)
        self.log("Saved model before epoch {}".format(epoch))

    def load_checkpt(self, epoch, model, dataset):
        model_state_path, model_ckpt_path, dataset_state_path = \
            self.get_epoch_checkpt_paths(epoch)

        model.load_ckpt(model_ckpt_path)
        with open(model_state_path, 'r') as f:
            model.set_state(pickle.load(f))
        with open(dataset_state_path, 'r') as f:
            dataset.set_state(pickle.load(f))
        
        self.log("Loaded model before epoch {}".format(epoch))
