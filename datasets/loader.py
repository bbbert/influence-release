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
    'mnist': ds.mnist.load_mnist,
    'mnist_small': ds.mnist.load_small_mnist,
    'spam': ds.spam.load_spam,
    'hospital': ds.hospital.load_hospital,
    'hospital_preprocess': ds.hospital.load_hospital_preprocess,
    'cifar10': ds.cifar10.load_cifar10,
    'cifar10_small': ds.cifar10.load_small_cifar10,
}

def load_dataset(dataset_id,
                 center_data=False,
                 append_bias=False,
                 data_dir=None):
    if dataset_id not in DATASETS:
        raise ValueError('Unknown dataset_id {}'.format(dataset_id))

    datasets = DATASETS[dataset_id](data_dir=data_dir)

    if center_data:
        datasets = ds.common.center_data(datasets)

    if append_bias:
        datasets = ds.common.append_bias(datasets)

    return datasets
