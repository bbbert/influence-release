from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import datasets as ds
import datasets.mnist
import datasets.spam
import datasets.hospital
import datasets.cifar10
import datasets.common

DATASETS = {
    'mnist': lambda: ds.mnist.load_mnist(),
    'mnist_small': lambda: ds.mnist.load_small_mnist(),
    'spam': lambda: ds.spam.load_spam(),
    'hospital': lambda: ds.hospital.load_hospital(),
    'cifar10': lambda: ds.cifar10.load_cifar10(),
    'cifar10_small': lambda: ds.cifar10.load_small_cifar10(),
}

def load_dataset(config):
    dataset_id = config.get('dataset_id', None)
    if dataset_id not in DATASETS:
        raise ValueError('Unknown dataset_id {}'.format(dataset_id))

    datasets = DATASETS[dataset_id]()

    center_data = config.get('center_data', False)
    if center_data:
        datasets = ds.common.center_data(datasets)

    append_bias = config.get('append_bias', False)
    if append_bias:
        datasets = ds.common.append_bias(datasets)

    return datasets
