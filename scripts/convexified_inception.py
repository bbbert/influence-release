from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from keras.layers import Flatten, AveragePooling2D
from keras.utils.data_utils import get_file
from keras import backend as K

import numpy as np
import tensorflow as tf
import time, os.path

from tensorflow.contrib.learn.python.learn.datasets import base
from load_animals import load_animals

from influence.dataset import DataSet
#from influence.all_CNN_c import All_CNN_C
from influence.logisticRegressionWithLBFGS import LogisticRegressionWithLBFGS
from configMaker import make_config, get_model_name

seeds = [0]
num_seeds = len(seeds)
dataset_type = 'processed_imageNet'
model_type = 'logreg_lbfgs'
#num_units = 2#3
out = '../output-week8'
nametag = 'convexified_inception'
force_refresh = False
num_steps = 1
test_idx = 0

def convexify(seed):
    tf.reset_default_graph()
    print('Starting seed {}'.format(seed))

    print('Creating new model')

    config_dict = make_config(dataset_type = dataset_type, seed=seed, model_type='logreg_lbfgs', out=out, nametag=nametag, save=True)
    # Not sure how to get input_dim, need to 
    config_dict['spec']['input_dim'] = int(W.shape[0]/config_dict['gen']['num_classes'])
    convex_model = LogisticRegressionWithLBFGS(config_dict)
    convex_model.train()

    print('Calculating influence')
    convex_pred_infl = convex_model.get_influence_on_test_loss(
            [test_idx],
            np.arange(convex_model.num_train_examples),
            force_refresh = True,
            batch_size = 'default'
            )
    np.savez('{}/{}_approx_pred_infl'.format(out, convex_model.model_name),
        convex_pred_infl = convex_pred_infl)

for seed in seeds:
    convexify(seed)
