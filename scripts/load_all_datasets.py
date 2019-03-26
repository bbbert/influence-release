from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import datasets.mnist
import datasets.spam
import datasets.hospital
import datasets.cifar10

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute subset self and test influences')

    # Environment args
    parser.add_argument('--data-dir', default=None, type=str,
                        help="The base dataset directory")

    args = parser.parse_args()
    data_dir = args.data_dir

    print("Downloading mnist")
    dataset = datasets.mnist.load_mnist(data_dir=data_dir)
    print("mnist: train={}, val={}, test={}".format(
        None if dataset.train is None else "{}".format(dataset.train.x.shape),
        None if dataset.validation is None else "{}".format(dataset.validation.x.shape),
        None if dataset.test is None else "{}".format(dataset.test.x.shape)))

    print("Downloading small_mnist")
    dataset = datasets.mnist.load_small_mnist(data_dir=data_dir)
    print("small_mnist: train={}, val={}, test={}".format(
        None if dataset.train is None else "{}".format(dataset.train.x.shape),
        None if dataset.validation is None else "{}".format(dataset.validation.x.shape),
        None if dataset.test is None else "{}".format(dataset.test.x.shape)))

    print("Downloading spam")
    dataset = datasets.spam.load_spam(data_dir=data_dir)
    print("spam: train={}, val={}, test={}".format(
        None if dataset.train is None else "{}".format(dataset.train.x.shape),
        None if dataset.validation is None else "{}".format(dataset.validation.x.shape),
        None if dataset.test is None else "{}".format(dataset.test.x.shape)))

    print("Downloading hospital")
    dataset = datasets.hospital.load_hospital(data_dir=data_dir)
    print("hospital: train={}, val={}, test={}".format(
        None if dataset.train is None else "{}".format(dataset.train.x.shape),
        None if dataset.validation is None else "{}".format(dataset.validation.x.shape),
        None if dataset.test is None else "{}".format(dataset.test.x.shape)))

    print("Downloading cifar10")
    dataset = datasets.cifar10.load_cifar10(data_dir=data_dir)
    print("cifar10: train={}, val={}, test={}".format(
        None if dataset.train is None else "{}".format(dataset.train.x.shape),
        None if dataset.validation is None else "{}".format(dataset.validation.x.shape),
        None if dataset.test is None else "{}".format(dataset.test.x.shape)))
