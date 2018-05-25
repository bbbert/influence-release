from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython

import tensorflow as tf

import influence.experiments as experiments
from influence.all_CNN_c_hidden2 import All_CNN_C_Hidden2
#from influence.all_CNN_c import All_CNN_C

from load_mnist import load_small_mnist, load_mnist



data_sets = load_small_mnist('data')    

num_classes = 10
input_side = 28
input_channels = 1
input_dim = input_side * input_side * input_channels 
weight_decay = 0.001 ##########
batch_size = 500

initial_learning_rate = 0.0001#0.001 
decay_epochs = [5000,10000]#[1500, 2500]
hidden1_units = 8
hidden2_units = 8
#hidden3_units = 8
conv_patch_size = 3
keep_probs = [1.0, 1.0]

seeds = [39,40]
retrain_seed = 17 ## this is default in experiments.py; doesn't matter since we're not doing random retraining
train_dir = '../scr/output'

def train_with_seed(seed):
 
    num_steps = 300000
    damping = 2e-2#1e-1#1e-2 ###########
    model_name = 'mnist_small_all_cnn_c_hidden2_seed{}_wd{}_damping{}_iter-{}'.format(seed, int(weight_decay*1000), int(damping*100), num_steps)

    model = All_CNN_C_Hidden2(
        input_side=input_side, 
        input_channels=input_channels,
        conv_patch_size=conv_patch_size,
        hidden1_units=hidden1_units, 
        hidden2_units=hidden2_units,
        #hidden3_units=hidden3_units,
        weight_decay=weight_decay,
        num_classes=num_classes, 
        batch_size=batch_size,
        data_sets=data_sets,
        initial_learning_rate=initial_learning_rate,
        damping=damping,
        decay_epochs=decay_epochs,
        mini_batch=True,
        train_dir=train_dir, 
        log_dir='log',
        model_name=model_name,
        seed=seed)

    model.train(
        num_steps=num_steps, 
        iter_to_switch_to_batch=10000000,
        iter_to_switch_to_sgd=10000000)
    iter_to_load = num_steps - 1

    test_idx = 6558

    num_to_remove=100
    actual_loss_diffs, predicted_loss_diffs, indices_to_remove = experiments.test_retraining(
        model, 
        test_idx=test_idx, 
        iter_to_load=iter_to_load, 
        num_to_remove=num_to_remove,
        num_steps=30000, 
        remove_type='maxinf',
        force_refresh=True,
        random_seed=retrain_seed)

    save_name = '{}/{}_retraining-{}.npz'.format(train_dir,model_name,num_to_remove)

    np.savez(
        save_name, 
        actual_loss_diffs=actual_loss_diffs, 
        predicted_loss_diffs=predicted_loss_diffs, 
        indices_to_remove=indices_to_remove
        )

for seed in seeds:
    print("Starting seed {}".format(seed))
    tf.reset_default_graph()
    train_with_seed(seed)
    print("Ending seed {}".format(seed))


