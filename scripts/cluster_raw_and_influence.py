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
nametag = 'cluster'#-no-damping'
force_refresh = True
#num_steps = 300000#1000000#300000
test_idx = 6558
test_indices = [6651, 3906, 1790, 5734, 5888, 7859, 3853, 9009, 1530, 2293]
num_points = 5500

def get_test_losses(model, test_pts):
    arr = []
    for pt in test_pts:
        arr.append(model.sess.run(model.loss_no_reg, feed_dict=\
                model.fill_feed_dict_with_one_ex(model.data_sets.test, pt)))
    return arr

def save_influence_vectors(seed, ignore_hess):
    tf.reset_default_graph()
    print('Starting seed {}'.format(seed))
    model_name = get_model_name(nametag=nametag, dataset_type=dataset_type, model_type=model_type, seed=seed)
    config_dict = make_config(seed=seed, dataset_type=dataset_type, model_type=model_type, out=out, test_idx=test_idx, nametag=nametag)
    #config_dict['spec']['weight_decay'] = 0.000001
    #config_dict['gen']['damping'] = 0
    model = LogisticRegressionWithLBFGS(config_dict)
    model.train()

    for test_pt in test_indices:
        model.get_influence_on_test_loss([test_pt], [0], False, 'default')

    train_losses = []
    for pt in range(num_points):
        train_losses.append(model.sess.run(model.loss_no_reg, feed_dict=\
                model.fill_feed_dict_with_one_ex(model.data_sets.train, pt)))
    np.savez('{}/{}_train_losses'.format(out, model_name), train_losses=train_losses)

    np.savez('{}/{}_test_losses'.format(out, model_name), test_losses=get_test_losses(model,\
            range(model.num_test_examples)))
    
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

def remove_cluster_and_retrain(seed, test_indices, removed_indices, name):
    tf.reset_default_graph()
    print('Starting seed {}'.format(seed))
    model_name = get_model_name(nametag=nametag, dataset_type=dataset_type, model_type=model_type, seed=seed)
    config_dict = make_config(seed=seed, dataset_type=dataset_type, model_type=model_type, out=out, test_idx=test_idx, nametag=nametag)
    model = LogisticRegressionWithLBFGS(config_dict)
    model.train()

    if test_indices == 'all':
        before_test_losses = model.sess.run(model.indiv_loss_no_reg, feed_dict=model.all_test_feed_dict)
    else:
        before_test_losses = get_test_losses(model, test_indices)

    print('Getting influences')
    influences = []
    # influences on the test points of the removed points
    if test_indices != 'all': # It's more expensive to get infl on all test points
        for test_pt in test_indices:
            influences.append(model.get_influence_on_test_loss([test_pt], np.array(removed_indices), False, 'default'))

    print('Doing retraining')
    tf.reset_default_graph()
    model_name = get_model_name(nametag=nametag+'retrain', dataset_type=dataset_type, model_type=model_type, seed=seed)
    config_dict = make_config(seed=seed, dataset_type=dataset_type, model_type=model_type, out=out, test_idx=test_idx, nametag=nametag+'retrain')
    model = LogisticRegressionWithLBFGS(config_dict)
    kept_indices = [i for i in range(num_points) if i not in removed_indices]
    model.all_train_feed_dict = model.fill_feed_dict_with_some_ex(model.data_sets.train, kept_indices)
    model.num_train_examples -= len(removed_indices)
    model.train()

    if test_indices == 'all':
        after_test_losses = model.sess.run(model.indiv_loss_no_reg, feed_dict=model.all_test_feed_dict)
    else:
        after_test_losses = get_test_losses(model, test_indices)
    np.savez('{}/{}_before_after_test_losses_vs_influence_on_{}_removing_{}'.format(out, model_name,\
            test_indices, name), before_test_losses=before_test_losses, after_test_losses=\
            after_test_losses, influences=influences)

for seed in seeds:
    #save_influence_vectors(seed, ignore_hess=True)
    model_name = get_model_name(nametag=nametag, dataset_type=dataset_type, model_type=model_type, seed=seed)
    f = np.load('{}/{}_fives_nines_to_remove.npz'.format(out, model_name))
    # These are hard test points according to their losses
    remove_cluster_and_retrain(seed, 'all', f['fives'][0], 'fives')
    remove_cluster_and_retrain(seed, 'all', f['nines'][0], 'nines')
