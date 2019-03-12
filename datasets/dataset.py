# Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

import numpy as np
import copy, warnings

class DataSet(object):

    def reset_rng(self):
        self._rng.seed(self._randomState)
        self.reset_indices('normal')

    def reset_orig(self):
        self._orig_rng.seed(self._randomState)
        self.reset_indices('orig')

    def reset_clone(self):
        self._clone_rng = copy.deepcopy(self._rng)
        self.reset_indices('clone')

    def reset_rngs(self):
        self.reset_rng()
        self.reset_orig()
        self.reset_clone()

    def reset_indices(self, which_rng):
        rngs = ['normal', 'orig', 'clone']
        assert which_rng in ['all'] + rngs
        if which_rng == 'all':
            for rng in rngs:
                self._indices_in_epoch[rng] = 0
                self._batch_indices[rng] = np.arange(self._num_examples)
        else:
            self._indices_in_epoch[which_rng] = 0
            self._batch_indices[which_rng] = np.arange(self._num_examples)

    # N.B.: omits is a boolean vector where True = omit, False = keep
    def __init__(self, x, labels, randomState, omits):

        if len(x.shape) > 2:
            x = np.reshape(x, [x.shape[0], -1])

        assert(x.shape[0] == labels.shape[0])

        x = x.astype(np.float32)

        self._x = x

        self._labels = labels
        self._num_examples = x.shape[0]
        self._indices_in_epoch = {}
        self._batch_indices = {}
        self.reset_indices('all')
        self._randomState = randomState
        self._rng = np.random.RandomState(self._randomState)
        self._orig_rng = np.random.RandomState(self._randomState)
        self.reset_clone()
        self._omits = omits

    @property
    def x(self):
        return self._x

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def omits(self):
        return self._omits

    @property
    def randomState(self):
        return self._randomState

    def set_randomState_and_reset_rngs(self,randomState):
        self._randomState = randomState
        self.reset_rngs()
        self.reset_indices('all')

    def reset_omits(self):
        self._omits = np.zeros(self._num_examples, dtype=bool)

    def set_omits(self, new_omits):
        self._omits = new_omits

    def reset_batch(self):
        raise DeprecationWarning("You probably don't want to reset all: reset_orig for eval funcs and set_omits for overriding/updating")
    self._index_in_epoch = 0
    self.reset_indices('all')
    self.reset_rngs()
    self.reset_omits()

    def next_batch(self, batch_size, which_rng, verbose=False):
        assert batch_size <= self._num_examples

        start = self._indices_in_epoch[which_rng]
        self._indices_in_epoch[which_rng] += batch_size
        if self._indices_in_epoch[which_rng] > self._num_examples:

            # Shuffle the data
            perm = np.arange(self._num_examples)
            if which_rng == "clone":
                self._clone_rng.shuffle(perm)
            elif which_rng == "orig":
                self._orig_rng.shuffle(perm)
            elif which_rng == "normal":
                self._rng.shuffle(perm)
            else:
                raise ValueError("Invalid rng type")

            self._batch_indices[which_rng] = self._batch_indices[which_rng][perm]

            # Start next epoch
            start = 0
            self._indices_in_epoch[which_rng] = batch_size

        end = self._indices_in_epoch[which_rng]

        # Extract x's and labels from batch_indices

        selected_indices = self._batch_indices[which_rng][start:end]

        selected_indices = selected_indices[~self._omits[selected_indices]]
        if verbose:
            print(selected_indices[0])
        #if len(selected_indices) != batch_size:
        #    print("Omitted something")


        return self._x[selected_indices], self._labels[selected_indices]


def filter_dataset(X, Y, pos_class, neg_class):
    """
    Filters out elements of X and Y that aren't one of pos_class or neg_class
    then transforms labels of Y so that +1 = pos_class, -1 = neg_class.
    """
    assert(X.shape[0] == Y.shape[0])
    assert(len(Y.shape) == 1)

    Y = Y.astype(int)

    pos_idx = Y == pos_class
    neg_idx = Y == neg_class
    Y[pos_idx] = 1
    Y[neg_idx] = -1
    idx_to_keep = pos_idx | neg_idx
    X = X[idx_to_keep, ...]
    Y = Y[idx_to_keep]
    return (X, Y)


def find_distances(target, X, theta=None):
    assert len(X.shape) == 2, "X must be 2D, but it is currently %s" % len(X.shape)
    target = np.reshape(target, -1)
    assert X.shape[1] == len(target), \
        "X (%s) and target (%s) must have same feature dimension" % (X.shape[1], len(target))

    if theta is None:
        return np.linalg.norm(X - target, axis=1)
    else:
        theta = np.reshape(theta, -1)

        # Project onto theta
        return np.abs((X - target).dot(theta))
