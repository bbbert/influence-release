from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython

import tensorflow as tf

import influence.experiments as experiments
from influence.all_CNN_c_hidden0 import All_CNN_C_Hidden0
#from influence.all_CNN_c import All_CNN_C

from load_mnist import load_small_mnist, load_mnist

import os.path
import time

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

seeds = [0]
retrain_seed = 17 ## this is default in experiments.py; doesn't matter since we're not doing random retraining
train_dir = '../scr/output'

def retrain_with_seed(seed):
 
    num_steps = 300000
    damping = 2e-2#1e-1#1e-2 ###########
    model_name = 'mnist_small_all_cnn_c_logreg_seed{}_iter-{}'.format(seed, num_steps)

    for dataset in data_sets:
        if dataset is not None:
            dataset.set_randomState_and_reset_rngs(seed)
    data_sets.train.set_omits(np.zeros(len(data_sets.train.labels),dtype=bool))

    model = All_CNN_C_Hidden0(
        input_side=input_side, 
        input_channels=input_channels,
        conv_patch_size=conv_patch_size,
        #hidden1_units=hidden1_units, 
        #hidden2_units=hidden2_units,
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
    
    #model.train(
    #    num_steps=num_steps, 
    #    iter_to_switch_to_batch=10000000,
    #    iter_to_switch_to_sgd=10000000)
    iter_to_load = num_steps-1#100000 - 1
    test_idx = 6558

    model.load_checkpoint(iter_to_load)
    sess = model.sess
    for step in xrange(30000):
        start_time = time.time()
        feed_dict = model.fill_feed_dict_with_batch(model.data_sets.train, which_rng="normal")
        _ = sess.run(model.train_op, feed_dict=feed_dict)
        feed_dict = model.fill_feed_dict_with_one_ex(model.data_sets.test,test_idx)
        train_loss_val = sess.run(model.loss_no_reg,feed_dict=feed_dict)
        duration = time.time() - start_time
        if step % 1000 == 0:
            print('Step %d: loss = %.8f (%.3f sec)' % (step, train_loss_val,duration))    

for seed in seeds:
    print("Starting seed {}".format(seed))
    tf.reset_default_graph()
    retrain_with_seed(seed)
    print("Ending seed {}".format(seed))


