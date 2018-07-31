from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import time, os.path

from influence.all_CNN_c import All_CNN_C
from configMaker import make_config, get_model_name

seeds = range(18,24)#range(300)
num_seeds = len(seeds)
dataset_type = 'cifar10'#'mnist_small'
model_type = 'all_cnn_c_hidden'
num_units = 3# 2
out = '../output-week4'#'../output-week3'
nametag = 'find-distribs-deeper'#'find_distribs'
force_refresh = False
num_steps = 1000000#300000
test_idx = 6558

params = [None] * num_seeds

for i, seed in enumerate(seeds):
    tf.reset_default_graph()
    print('Starting seed {}'.format(seed))
    model_name = get_model_name(nametag=nametag, dataset_type=dataset_type, model_type=model_type, seed=seed, num_units=num_units, num_steps=num_steps)
    config_dict = make_config(dataset_type = dataset_type, seed=seed, model_type=model_type, out=out, nametag=nametag, num_steps=num_steps, save=False)
    model = All_CNN_C(config_dict)
    model.load_checkpoint(num_steps-1,False)
    params[i] = np.array(model.sess.run(model.get_all_params()))
    print('Ending seed {}'.format(seed))

np.savez('{}/{}_{}-seed_all_params'.format(out,model_name,num_seeds), params=params)

