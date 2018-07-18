from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython
import cPickle as pickle

import tensorflow as tf
import time
import os.path

#import influence.experiments as experiments
from influence.all_CNN_c import All_CNN_C
from configMaker import make_config, get_model_name

from scipy.stats import pearsonr

import argparse

#parser = argparse.ArgumentParser(description='Input seed and point.')
#parser.add_argument('seed', type=int)
#parser.add_argument('point', type=int)
#args=parser.parse_args()

#point = args.point
#seed = args.seed

seeds = [0]
dataset_type = 'cifar10_small'#'mnist'#'mnist_small'
model_type = 'all_cnn_c_hidden'
num_units = 2#3#2
out = '../output-week4'#'../output-week3'
nametag = 'find_distribs'
force_refresh=True
num_steps = 300000#1000000#300000

test_idx = 6558

for seed in seeds:
    tf.reset_default_graph()

    print("Starting seed {}".format(seed))
    model_name = get_model_name(nametag=nametag, dataset_type=dataset_type, model_type=model_type, seed=seed, num_units=num_units, num_steps=num_steps)
    model = All_CNN_C(make_config(dataset_type=dataset_type, seed=seed, model_type=model_type, out=out, nametag=nametag, num_steps=num_steps, save=True))
    lossespathname = '{}/{}_test_losses_over_time'.format(out, model_name)

    print('Model {}'.format(model_name))

    # Training
    if os.path.exists('{}.npz'.format(lossespathname)) and not force_refresh:
        f = np.load('{}.npz'.format(lossespathname))
        losses = f['losses']
        losses_fine = f['losses_fine']
    else:
        model.train(num_steps=num_steps,
            iter_to_switch_to_batch=10000000,
            iter_to_switch_to_sgd=10000000)
        losses,losses_fine = model.get_all_losses()
    np.savez(lossespathname, losses=losses, losses_fine=losses_fine)
    
    model.load_checkpoint(num_steps-1,True)

    train_losses = model.sess.run(model.indiv_loss_no_reg, feed_dict=model.all_train_feed_dict)
    test_losses = model.sess.run(model.indiv_loss_no_reg, feed_dict=model.all_test_feed_dict)
    pred_infl = model.get_influence_on_test_loss(
            [test_idx],#[8],#[test_idx],
            np.arange(len(model.data_sets.train.labels)),
            force_refresh=True,
            batch_size='default'
            )

    #np.savez('{}/{}_train_test_losses_pred_infl_on_8'.format(out,model_name),
    np.savez('{}/{}_train_test_losses_pred_infl'.format(out,model_name),
            train_losses=train_losses,
            test_losses=test_losses,
            pred_infl=pred_infl)
