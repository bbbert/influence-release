from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import datasets.mnist
import datasets.spam
import datasets.hospital
import datasets.cifar10

if __name__ == "__main__":
    print("Downloading mnist")
    dataset = datasets.mnist.load_mnist()
    print("mnist: train={}, val={}, test={}".format(
        None if dataset.train is None else "{}".format(dataset.train.x.shape),
        None if dataset.validation is None else "{}".format(dataset.validation.x.shape),
        None if dataset.test is None else "{}".format(dataset.test.x.shape)))

    print("Downloading small_mnist")
    dataset = datasets.mnist.load_small_mnist()
    print("small_mnist: train={}, val={}, test={}".format(
        None if dataset.train is None else "{}".format(dataset.train.x.shape),
        None if dataset.validation is None else "{}".format(dataset.validation.x.shape),
        None if dataset.test is None else "{}".format(dataset.test.x.shape)))

    print("Downloading spam")
    dataset = datasets.spam.load_spam()
    print("spam: train={}, val={}, test={}".format(
        None if dataset.train is None else "{}".format(dataset.train.x.shape),
        None if dataset.validation is None else "{}".format(dataset.validation.x.shape),
        None if dataset.test is None else "{}".format(dataset.test.x.shape)))

    print("Downloading hospital")
    dataset = datasets.hospital.load_hospital()
    print("hospital: train={}, val={}, test={}".format(
        None if dataset.train is None else "{}".format(dataset.train.x.shape),
        None if dataset.validation is None else "{}".format(dataset.validation.x.shape),
        None if dataset.test is None else "{}".format(dataset.test.x.shape)))

    print("Downloading cifar10")
    dataset = datasets.cifar10.load_cifar10()
    print("cifar10: train={}, val={}, test={}".format(
        None if dataset.train is None else "{}".format(dataset.train.x.shape),
        None if dataset.validation is None else "{}".format(dataset.validation.x.shape),
        None if dataset.test is None else "{}".format(dataset.test.x.shape)))
