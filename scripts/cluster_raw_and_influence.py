from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import time, os.path

#from influence.all_CNN_c import All_CNN_C
from influence.logisticRegressionWithLBFGS import LogisticRegressionWithLBFGS
from configMaker import make_config, get_model_name
from influence.hessians import hessians

seeds = [0]
num_seeds = len(seeds)
dataset_type = 'mnist_small'
model_type = 'logreg_lbfgs'
#num_units = 2#3
out = '../output-week6'
nametag = 'cluster'
force_refresh = True
#num_steps = 300000#1000000#300000
test_idx = 6558
num_points = 5500

def save_influence_vectors(seed, ignore_hess):
    tf.reset_default_graph()
    print('Starting seed {}'.format(seed))
    model_name = get_model_name(nametag=nametag, dataset_type=dataset_type, model_type=model_type, seed=seed)
    config_dict = make_config(seed=seed, dataset_type=dataset_type, model_type=model_type, out=out, test_idx=test_idx)
    model = LogisticRegressionWithLBFGS(config_dict)
    model.train()
    
    infl_vectors = [None] * num_points
    if not ignore_hess:
        print('Loading hessian')
        hess = hessians(ys=model.total_loss, xs=model.params)
        print(hess.shape, hess)
        print('Inverting hessian')
        inv_hess = np.linalg.inv(model.sess.run(hess[0]))
        print('Multiplying hessian by gradients')
    for point in range(num_points):
        train_grad_val = np.concatenate(model.sess.run(model.grad_total_loss_op, feed_dict=model.fill_feed_dict_with_one_ex(model.data_sets.train, point)))
        if ignore_hess:
            infl_vectors[point] = train_grad_val
        else:
            infl_vectors[point] = - np.dot(inv_hess, train_grad_val)
    infl_vectors = np.array(infl_vectors)
    print(infl_vectors.shape)

    print('Ending seed {}'.format(seed))
    np.savez('{}/{}_influence_vectors-{}-ignoring-hess-{}'.format(out, model_name, test_idx, ignore_hess), infl_vectors=infl_vectors)

for seed in seeds:
    save_influence_vectors(seed, ignore_hess=True)

