from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import time, os.path

from influence.all_CNN_c import All_CNN_C
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
num_points = 5500

# Look for inverse HVPs in 'modelname-cg-normal_loss-test-[6558].npz'

train_grad_loss_vals = [None] * num_seeds

idx = 1173#0

for i, seed in enumerate(seeds):
    tf.reset_default_graph()
    print('Starting seed {}'.format(seed))
    model_name = get_model_name(nametag=nametag, dataset_type=dataset_type, model_type=model_type, seed=seed, num_units=num_units, num_steps=num_steps)
    config_dict = make_config(dataset_type = dataset_type, seed=seed, model_type=model_type, out=out, nametag=nametag, num_steps=num_steps, save=False)
    model = All_CNN_C(config_dict)
    model.load_checkpoint(num_steps-1,False)
    feed_dict = model.fill_feed_dict_with_one_ex(model.data_sets.train, idx)
    train_grad_loss_vals[i] = np.array(model.sess.run(model.grad_total_loss_op, feed_dict=feed_dict))
    print('Ending seed {}'.format(seed))
    np.savez('{}/{}_{}-seed_{}_train_grad_loss_vals'.format(out,model_name,num_seeds,idx), train_grad_loss_vals=train_grad_loss_vals)

