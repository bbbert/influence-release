from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import tensorflow as tf
import time
import os.path

import math
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

from influence.logisticRegressionWithLBFGS import LogisticRegressionWithLBFGS
from configMaker import make_config, get_model_name

seed = 0 # irrelevant for convex model
dataset_type = 'hospital' 
# hospital is binary
# processed_imageNet is 10-class
model_type = 'logreg_lbfgs'
out = '../output-break-infl-logreg'
nametag = 'break-infl-logreg'

if dataset_type == 'processed_imageNet':
    default_prop = 0.09 # Doing 10% messes up the single-class subset in imageNet since an entire class is removed; the training breaks
else:
    default_prop = 0.1
default_num_subsets = 100

random_seed = 2
np.random.seed(random_seed) # intended for subset choices

def get_losses(model):
    train_losses = model.sess.run(model.indiv_loss_no_reg, feed_dict=model.all_train_feed_dict)
    test_losses = model.sess.run(model.indiv_loss_no_reg, feed_dict=model.all_test_feed_dict)
    return train_losses, test_losses

# test_idx: choose a test point to measure scalar infl against
def orig_train(test_idx=None):
    tf.reset_default_graph()

    model_name = get_model_name(nametag, dataset_type, model_type, seed)
    config_dict = make_config(seed, dataset_type, model_type, out, nametag='break_infl_logreg', test_idx=test_idx)
    # For some reason, it takes 100 iter, which is the default param.
    # When I change it to max_iter of 20, it takes 20 iter. Is this a bug?
    config_dict['spec']['max_lbfgs_iter'] = 400
    model = LogisticRegressionWithLBFGS(config_dict)

    model.train()
    train_losses, test_losses = get_losses(model)
    print('Trained original model.')

    if test_idx is None:
        test_idx = np.argmax(test_losses)
    num_train_pts = model.num_train_examples
    pred_infl = model.get_influence_on_test_loss([test_idx], np.arange(num_train_pts), force_refresh=True, batch_size='default')
    print('Calculated scalar infl for all training points on test_idx {}.'.format(test_idx))

    grad_loss = []
    for point in range(num_train_pts):
        grad_loss.append(np.concatenate(model.sess.run(model.grad_total_loss_op, feed_dict=model.fill_feed_dict_with_one_ex(model.data_sets.train, point))))
    grad_loss = np.array(grad_loss)
    print('Calculated training gradients for all training points.')

    return model_name, config_dict, model, train_losses, test_losses, pred_infl, grad_loss, test_idx

def retrained_losses(model, remove_indices):
    num_train_pts = model.num_train_examples
    included_indices = [i for i in range(num_train_pts) if i not in remove_indices]
    assert len(included_indices) == num_train_pts - len(remove_indices)
    model.retrain(0, feed_dict=model.fill_feed_dict_with_some_ex(model.data_sets.train, included_indices), verbose=False)
    return get_losses(model)

def retrain(model, remove_subsets):
    train_losses, test_losses = [], []
    n = len(remove_subsets)
    n20 = n//20
    for i, remove_indices in enumerate(remove_subsets):
        if (i % n20 == 0):
            print('Retraining model {} out of {}'.format(i, n))
        train_loss, test_loss = retrained_losses(model, remove_indices)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    return np.array(train_losses), np.array(test_losses)

def get_random_subset(num_train_pts, proportion=default_prop, num=default_num_subsets):
    subsets = []
    for i in range(num):
        subsets.append(np.random.choice(num_train_pts, int(num_train_pts * proportion), replace=False))
    return np.array(subsets)

def get_scalar_infl_tails(num_train_pts, pred_infl, proportion=default_prop, num=default_num_subsets):
    assert proportion < 0.5
    window = int(2*proportion*num_train_pts)
    points = int(proportion*num_train_pts)
    scalar_infl_indices = np.argsort(pred_infl)
    pos_subsets, neg_subsets = [], []
    for i in range(num):
        neg_subsets.append(np.random.choice(scalar_infl_indices[:window], points, replace=False))
        pos_subsets.append(np.random.choice(scalar_infl_indices[-window:], points, replace=False))
    return np.array(neg_subsets), np.array(pos_subsets)

def get_same_grad_dir(num_train_pts, grad_loss, proportion=default_prop, num=default_num_subsets):
    # Using Pigeonhole to guarantee we get a sufficiently large cluster
    points = int(proportion*num_train_pts)
    n_clusters = int(math.floor(1 / proportion))
    km = KMeans(n_clusters=n_clusters)
    km.fit(grad_loss)
    labels, centroids = km.labels_, km.cluster_centers_
    _, counts = np.unique(labels, return_counts=True)
    
    best = None
    for i in range(len(counts)):
        if counts[i] >= points and (best is None or counts[i] < counts[best]):
            best = i

    subsets = []
    cluster_indices = np.where(labels == best)[0]
    for i in range(num):
        subsets.append(np.random.choice(cluster_indices, points, replace=False))
    return np.array(subsets), best, labels

def get_same_class_subset(num_train_pts, labels, proportion=default_prop, num=default_num_subsets):
    points = int(proportion*num_train_pts)
    label_vals, counts = np.unique(labels, return_counts=True)
    valid_labels = []
    valid_indices = []
    for i in range(len(label_vals)):
        if counts[i] >= points:
            valid_labels.append(label_vals[i])
            valid_indices.append(np.where(labels == label_vals[i])[0])
    assert len(valid_indices) > 0
    flat = np.ndarray.flatten(np.array(valid_indices))
    label_to_ind = dict(zip(valid_labels, range(len(valid_indices))))

    subsets = []
    for i in range(num):
        sample = np.random.choice(flat)
        sample_ind = label_to_ind[labels[sample]]
        subsets.append(np.random.choice(valid_indices[sample_ind], points, replace=False))
    return np.array(subsets)

model_name, config_dict, model, train_losses, test_losses, pred_infl, grad_loss, test_idx = orig_train()
num_train_pts = model.num_train_examples
print('Finished original training.')

random_subsets = get_random_subset(num_train_pts)
print('Found random subsets.')
neg_tail_subsets, pos_tail_subsets = get_scalar_infl_tails(num_train_pts, pred_infl)
print('Found scalar infl tail subsets.')
same_grad_subsets, cluster_label, cluster_labels = get_same_grad_dir(num_train_pts, grad_loss)
print('Found same gradient subsets.')
same_class_subsets = get_same_class_subset(num_train_pts, model.data_sets.train.labels)
print('Found same class subsets.')

random_train_losses, random_test_losses = retrain(model, random_subsets)
print('Finished random retraining.')
print(random_train_losses.shape)
neg_tail_train_losses, neg_tail_test_losses = retrain(model, neg_tail_subsets)
print('Finished neg tails retraining.')
pos_tail_train_losses, pos_tail_test_losses = retrain(model, pos_tail_subsets)
print('Finished pos tails retraining.')
same_grad_train_losses, same_grad_test_losses = retrain(model, same_grad_subsets)
print('Finished same grad retraining.')
same_class_train_losses, same_class_test_losses = retrain(model, same_class_subsets)
print('Finished same class retraining.')

np.savez(os.path.join(out, 'all-experiment-data-{}-prop-{}-subsets-{}-random_seed-{}'.format(dataset_type, default_prop, default_num_subsets, random_seed)),
        train_losses=train_losses,
        test_losses=test_losses,
        pred_infl=pred_infl,
        grad_loss=grad_loss,
        test_idx=test_idx,
        random_subsets=random_subsets,
        neg_tail_subsets=neg_tail_subsets,
        pos_tail_subset=pos_tail_subsets,
        same_grad_subsets=same_grad_subsets,
        cluster_label=cluster_label,
        cluster_labels=cluster_labels,
        same_class_subsets=same_class_subsets,
        random_train_losses=random_train_losses,
        random_test_losses=random_test_losses,
        neg_tail_train_losses=neg_tail_train_losses,
        neg_tail_test_losses=neg_tail_test_losses,
        pos_tail_train_losses=pos_tail_train_losses,
        pos_tail_test_losses=pos_tail_test_losses,
        same_grad_train_losses=same_grad_train_losses,
        same_grad_test_losses=same_grad_test_losses,
        same_class_train_losses=same_class_train_losses,
        same_class_test_losses=same_class_test_losses)
