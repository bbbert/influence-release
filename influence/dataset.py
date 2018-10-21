# Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

import numpy as np
import copy, warnings

class DataSet(object):
    def __init__(self, x, labels, omits=[]):
        if len(x.shape) > 2:
            x = np.reshape(x, [x.shape[0], -1])
        assert(x.shape[0] == labels.shape[0])
        x = x.astype(np.float32)

        self._x = x
        self._labels = labels
        self._num_examples = x.shape[0]

        # run through the dataset by generating a series of random permutations
        self._rng = None
        self._epoch_indices = None
        self._index_in_epoch = None

        # labels to omit from each batch
        self._omits = set(omits)

    def init_state(self, seed):
        self._rng = np.random.RandomState(seed)
        self.new_epoch()

    def set_state(self, dataset_state):
        random_state, epoch_indices, index_in_epoch = dataset_state
        self._rng.set_state(random_state)
        self._epoch_indices = np.array(epoch_indices)
        self._index_in_epoch = index_in_epoch

    def get_state(self):
        if self._rng is None:
            raise ValueError("The dataset's random state is not initialized.")
        return (self._rng.get_state(), list(epoch_indices), index_in_epoch)

    def get_omits(self):
        return self._omits

    def set_omits(self, omits):
        self._omits = set(omits)

    def clone(self):
        d = DataSet(self._x, self._labels, self._labels_to_omit)
        if self._rng is not None:
            d.set_state(self.get_state())
        return d

    @property
    def x(self):
        return self._x

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def new_epoch(self):
        indices = np.arange(self._num_examples)
        self._epoch_indices = np._rng.shuffle(indices)
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        selected_indices = []

        while len(selected_indices) < batch_size:
            indices_to_take = min(batch_size - len(selected_indices), self.num_examples - self._index_in_epoch)
            selected_indices += self._epoch_indices[self._index_in_epoch : self._index_in_epoch + indices_to_take]
            self._index_in_epoch += indices_to_take

            if self._index_in_epoch >= self.num_examples:
                self.new_epoch()

        # filter after selecting indices so that batch order stays the same
        selected_indices = [i for i in selected_indices if self._labels[i] not in self._labels_to_omit]

        return self._x[selected_indices], self._labels[selected_indices]
