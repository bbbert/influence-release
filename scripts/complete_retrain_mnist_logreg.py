from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import pandas as pd

import IPython
import tensorflow as tf

from influence.logisticRegressionWithLBFGS import LogisticRegressionWithLBFGS
import influence.experiments as experiments

from load_mnist import load_mnist, load_small_mnist



data_sets = load_small_mnist('data')

num_classes = 10
seed = 0

input_dim = data_sets.train.x.shape[1]
weight_decay = 0.01
batch_size = 1400
initial_learning_rate = 0.001 
keep_probs = None
max_lbfgs_iter = 1000
decay_epochs = [1000, 10000]

tf.reset_default_graph()

tf_model = LogisticRegressionWithLBFGS(
    seed=seed,
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='mnist_logreg_lbfgs')

tf_model.train()
test_idx=6558

actual_loss_diffs, predicted_loss_diffs, indices_to_remove = experiments.test_retraining(
        tf_model,
        test_idx,
        iter_to_load=0,
        force_refresh=True,
        num_to_remove=6,
        remove_type='manual',
        indices_to_remove=[1173,4644,1891,4936,1735,3562],
        random_seed=seed,
        do_sanity_checks=True)

