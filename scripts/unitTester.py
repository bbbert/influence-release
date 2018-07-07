from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import cPickle as pickle

from influence.logisticRegressionWithLBFGS import LogisticRegressionWithLBFGS
from influence.all_CNN_c import All_CNN_C
import influence.experiments as experiments

def load_config(model_name):
    pickle_in = open('configs/config_dict_{}.pickle'.format(model_name), 'rb')
    return pickle.load(pickle_in)

test_idx=6558

def train_retrain_logreg():
    tf.reset_default_graph()
    model_name = 'mnist_small_logreg_lbfgs_seed0'

    model = LogisticRegressionWithLBFGS(load_config(model_name))
    #model.mini_batch = False####

    model.train()
    
    actual_loss_diffs, predicted_loss_diffs, indices_to_remove = experiments.test_retraining(
            model,
            test_idx,
            iter_to_load=0,
            force_refresh=True,
            num_to_remove=6,
            num_steps=0,
            remove_type='manual',
            indices_to_remove=[1173,4644,1891,4936,1735,3562],
            random_seed=None,
            do_sanity_checks=True)
    """
    predicted_loss_diffs_lissa = model.get_influence_on_test_loss(
                [test_idx],
                indices_to_remove,
                batch_size='default',
                approx_type='lissa',
                approx_params={'scale':25, 'recursion_depth':5000, 'damping':0, 'batch_size':1, 'num_samples':10},
                force_refresh=True
                )

    print('CG: {}'.format(predicted_loss_diffs))
    print('LiSSA: {}'.format(predicted_loss_diffs_lissa))
    """

def small_train_retrain_hidden2():
    tf.reset_default_graph()
    num_steps = 10000 #300000
    model_name = 'mnist_small_all_cnn_c_hidden2_seed0_iter-{}'.format(num_steps)
    model = All_CNN_C(load_config(model_name))

    model.train(num_steps=num_steps,
            iter_to_switch_to_batch=10000000,
            iter_to_switch_to_sgd=1000000,
            save_checkpoints=True,
            verbose=True,
            track_losses=True)
    iter_to_load = num_steps - 1
    num_to_remove=2
    

    actual_loss_diffs = experiments.test_only_retraining(
    #actual_loss_diffs, predicted_loss_diffs, indices_to_remove = experiments.test_retraining(
            model,
            num_to_remove=2,
            test_idx=test_idx,
            iter_to_load=iter_to_load,
            num_steps=5000,
            remove_type='manual',
            #remove_type='maxinf',
            force_refresh=True,
            random_seed=None,
            indices_to_remove=[1173,851],
            #indices_to_remove=None,
            do_sanity_checks=True
            )

train_retrain_logreg()
#small_train_retrain_hidden2()

