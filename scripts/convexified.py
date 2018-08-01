from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import time, os.path

from tensorflow.contrib.learn.python.learn.datasets import base

from influence.dataset import DataSet
from influence.all_CNN_c import All_CNN_C
from influence.logisticRegressionWithLBFGS import LogisticRegressionWithLBFGS
from configMaker import make_config, get_model_name

seeds = range(300)#range(18,24)#range(300)
num_seeds = len(seeds)
dataset_type = 'mnist_small'#'cifar10'#'mnist_small'
model_type = 'all_cnn_c_hidden'
num_units = 2#3
out = '../output-week3'#'../output-week4'#'../output-week3'
nametag = 'find_distribs'#'find-distribs-deeper'#'find_distribs'
force_refresh = False
num_steps = 300000#1000000#300000
test_idx = 6558

def convexify(seed):
    tf.reset_default_graph()
    print('Starting seed {}'.format(seed))

    # Load old model
    config_dict = make_config(dataset_type = dataset_type, seed=seed, model_type=model_type, out=out, nametag=nametag, num_steps=num_steps, save=False)
    full_model = All_CNN_C(config_dict)
    full_model.load_checkpoint(num_steps-1,False)

    # Process datasets
    updated_train = full_model.sess.run(full_model.h_d, feed_dict=full_model.all_train_feed_dict)
    updated_test = full_model.sess.run(full_model.h_d, feed_dict=full_model.all_test_feed_dict)

    # Get last layer
    full_params = full_model.get_all_params()
    b = full_model.sess.run(full_params[-1])
    W = full_model.sess.run(full_params[-2])

    # Create new model
    tf.reset_default_graph()
    config_dict = make_config(dataset_type = dataset_type, seed=seed, model_type='logreg_lbfgs', out=out, nametag='convexified', save=True)
    config_dict['spec']['input_dim'] = int(W.shape[0]/config_dict['gen']['num_classes'])
    config_dict['spec']['has_biases'] = True
    convex_model = LogisticRegressionWithLBFGS(config_dict)

    # Load processed datasets
    train = DataSet(updated_train, convex_model.data_sets.train.labels, 0, np.zeros(convex_model.num_train_examples,dtype=bool))
    validation = convex_model.data_sets.validation
    test = DataSet(updated_test, convex_model.data_sets.test.labels, 0, np.zeros(convex_model.num_test_examples,dtype=bool))
    convex_model.data_sets = base.Datasets(train=train, validation=validation, test=test)
    convex_model.all_train_feed_dict = convex_model.fill_feed_dict_with_all_ex(convex_model.data_sets.train)
    convex_model.all_test_train_feed_dict = convex_model.fill_feed_dict_with_all_ex(convex_model.data_sets.test)

    # Load last layer
    params_feed_dict = {}
    params_feed_dict[convex_model.b_placeholder] = b
    params_feed_dict[convex_model.W_placeholder] = W
    convex_model.sess.run(convex_model.set_params_op, feed_dict=params_feed_dict)

    # Try influence calculation
    convex_pred_infl = convex_model.get_influence_on_test_loss(
            [test_idx],
            np.arange(convex_model.num_train_examples),
            force_refresh = True,
            batch_size = 'default'
            )
    np.savez('{}/{}_approx_pred_infl'.format(out, convex_model.model_name),
        convex_pred_infl = convex_pred_infl)

for seed in range(25):
    convexify(seed)
