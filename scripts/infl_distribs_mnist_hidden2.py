from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython

import tensorflow as tf
import time

import influence.experiments as experiments
from influence.all_CNN_c_hidden2 import All_CNN_C_Hidden2


from load_mnist import load_small_mnist, load_mnist



data_sets = load_small_mnist('data')    

num_classes = 10
input_side = 28
input_channels = 1
input_dim = input_side * input_side * input_channels 
weight_decay = 0.001
batch_size = 500

initial_learning_rate = 0.0001
decay_epochs = [5000,10000]
hidden1_units = 8
hidden2_units = 8
conv_patch_size = 3
keep_probs = [1.0, 1.0]

#num_seeds = 17
#seeds = range(num_seeds)
seeds = [17]
train_dir = '../scr/output'

def get_infl_with_seed(seed):
    num_steps = 300000
    damping = 2e-2
    model_name = 'mnist_small_all_cnn_c_hidden2_seed{}_wd{}_damping{}_iter-{}'.format(seed, int(weight_decay*1000), int(damping*100), num_steps)

    model = All_CNN_C_Hidden2(
        input_side=input_side, 
        input_channels=input_channels,
        conv_patch_size=conv_patch_size,
        hidden1_units=hidden1_units, 
        hidden2_units=hidden2_units,
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

    iter_to_load = num_steps - 1
    test_idx = 6558
    model.load_checkpoint(iter_to_load)

    predicted_loss_diffs = model.get_influence_on_test_loss(
            [test_idx],
            np.arange(len(model.data_sets.train.labels)),
            force_refresh=False)

    save_name = '{}/{}_predicted_loss_diffs-test-{}.npz'.format(train_dir,model_name,[test_idx])

    np.savez(
        save_name,
        predicted_loss_diffs=predicted_loss_diffs)

for seed in seeds:
    print("Starting seed {}".format(seed))
    tf.reset_default_graph()
    get_infl_with_seed(seed)
    print("Ending seed {}".format(seed))


