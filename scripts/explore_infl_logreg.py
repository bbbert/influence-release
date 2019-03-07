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

seed = 10
subset_seed = 0
#dataset_type = 'processed_imageNet' # processed_imageNet is 10-class
#dataset_type = 'hospital' # hospital is binary
dataset_type = 'mnist_small'
center_data = False
model_type = 'logreg_lbfgs'
out = './output-explore-infl-logreg'
nametag = 'explore-infl-logreg'

if dataset_type == 'processed_imageNet':
    default_prop = 0.09 # Doing 10% messes up the single-class subset in imageNet since an entire class is removed; the training breaks
else:
    default_prop = 0.1
default_num_subsets = 30

use_hessian_lu = True

def get_losses(model):
    train_losses = model.sess.run(model.indiv_loss_no_reg, feed_dict=model.all_train_feed_dict)
    test_losses = model.sess.run(model.indiv_loss_no_reg, feed_dict=model.all_test_feed_dict)
    return train_losses, test_losses

def get_margins(model):
    train_margins = model.sess.run(model.margins, feed_dict=model.all_train_feed_dict)
    test_margins = model.sess.run(model.margins, feed_dict=model.all_test_feed_dict)
    return train_margins, test_margins

def get_cross_validated_weight_decay(initial_config_dict,
                                     min_weight_decay=0.0001,
                                     max_weight_decay=0.01,
                                     weight_decay_samples=5,
                                     num_folds=5):
    config_dict = initial_config_dict.copy()
    config_dict['spec'] = config_dict['spec'].copy()

    weight_decays = np.logspace(np.log10(min_weight_decay),
                                np.log10(max_weight_decay), weight_decay_samples)
    cv_errors = np.zeros(weight_decay_samples)
    for i, weight_decay in enumerate(weight_decays):
        config_dict['spec']['weight_decay'] = weight_decay
        model = LogisticRegressionWithLBFGS(config_dict)

        num_train_pts = model.num_train_examples
        fold_size = (num_train_pts + num_folds - 1) // num_folds
        cv_error = 0.0
        for k in range(num_folds):
            fold_begin, fold_end = k * num_folds, min(num_train_pts, (k + 1) * num_folds)
            fold_train_indices = np.concatenate((np.arange(0, fold_begin), np.arange(fold_end, num_train_pts)))

            model.all_train_feed_dict = model.fill_feed_dict_with_some_ex(model.data_sets.train, fold_train_indices)
            model.train()
            fold_feed_dict = model.fill_feed_dict_with_some_ex(model.data_sets.train, np.arange(fold_begin, fold_end))
            fold_loss = model.sess.run(model.loss_no_reg, feed_dict=fold_feed_dict)
            cv_error += fold_loss

        cv_errors[i] = cv_error
        print('Cross-validation error is {} for weight_decay={}.'.format(cv_error, weight_decay))

        model.sess.close()
        tf.reset_default_graph()

    best_i = np.argmin(cv_errors)
    best_weight_decay = weight_decays[best_i]
    print('Cross-validation errors: {}'.format(cv_errors))
    print('Selecting weight_decay {}, with error {}.'.format(best_weight_decay, cv_errors[best_i]))
    print('Cross-validation complete.')

    return best_weight_decay

# test_idx: choose a test point to measure scalar infl against
def initial_training():
    tf.reset_default_graph()

    model_name = get_model_name(nametag, dataset_type, model_type, seed)
    config_dict = make_config(seed, dataset_type, model_type, out, nametag=nametag, test_idx=0)
    # For some reason, it takes 100 iter, which is the default param.
    # When I change it to max_iter of 20, it takes 20 iter. Is this a bug?
    config_dict['spec']['max_lbfgs_iter'] = 1024
    config_dict['gen']['center_data'] = center_data

    weight_decay = 0.0003#get_cross_validated_weight_decay(config_dict)
    config_dict['spec']['weight_decay'] = weight_decay
    model = LogisticRegressionWithLBFGS(config_dict)

    model.train()
    train_losses, test_losses = get_losses(model)
    train_margins, test_margins = None, None
    if config_dict['gen']['num_classes'] == 2:
        train_margins, test_margins = get_margins(model)
    print('Trained original model.')

    grad_loss = []
    for point in range(model.num_train_examples):
        grad_loss.append(np.concatenate(model.sess.run(model.grad_total_loss_op, feed_dict=model.fill_feed_dict_with_one_ex(model.data_sets.train, point))))
    grad_loss = np.array(grad_loss)
    print('Calculated training gradients for all training points.')

    return model_name, config_dict, model, train_losses, test_losses, train_margins, test_margins, grad_loss

def pick_test_points(test_losses):
    argsort = np.argsort(test_losses)

    # Pick 3 high loss points
    high_loss = argsort[-3:]

    # Pick 3 random points
    random_loss = np.random.choice(argsort[:-3], 3, replace=False)

    return list(high_loss) + list(random_loss)

def get_fixed_test_influence(model, test_points):
    # Get predicted influences on a set of fixed test points

    num_train_pts = model.num_train_examples
    fixed_test_pred_infl = []
    fixed_test_pred_margin_infl = []
    for test_idx in test_points:
        pred_infl = model.get_influence_on_test_loss([test_idx], np.arange(num_train_pts), force_refresh=True, batch_size='default', use_hessian_lu=use_hessian_lu)
        fixed_test_pred_infl.append(pred_infl)
        if model.num_classes == 2:
            pred_margin_infl = model.get_influence_on_test_loss([test_idx], np.arange(num_train_pts), force_refresh=True, batch_size='default', margins=True, use_hessian_lu=use_hessian_lu)
            fixed_test_pred_margin_infl.append(pred_margin_infl)
        print('Calculated scalar infl for all training points on test_idx {}.'.format(test_idx))

    return fixed_test_pred_infl, fixed_test_pred_margin_infl

def retrained_losses(model, remove_indices):
    num_train_pts = model.num_train_examples
    included_indices = [i for i in range(num_train_pts) if i not in remove_indices]
    assert len(included_indices) == num_train_pts - len(remove_indices)
    model.retrain(0, feed_dict=model.fill_feed_dict_with_some_ex(model.data_sets.train, included_indices), verbose=False)
    return get_losses(model)

def retrain(model, remove_subsets, remove_tags):
    if remove_subsets is None:
        return None, None

    n = len(remove_subsets)
    n_report = max(n // 100, 1)

    # It is important that the influence gets calculated before the model is retrained,
    # so that the parameters are the original parameters
    start_time = time.time()
    self_pred_infls = []
    self_pred_margin_infls = []
    for i, remove_indices in enumerate(remove_subsets):
        if (i % n_report == 0):
            print('Computing self-influences for subset {} out of {} (tag={})'.format(i, n, remove_tags[i]))

        # get_influence_on_test_loss returns influence for the mean test gradient, we want actual self influences
        pred_infls = model.get_influence_on_test_loss(remove_indices, remove_indices, force_refresh=True, batch_size='default',
                                                      test_indices_from_train=True, # remove_indices refers to training points
                                                      test_description='subset-{}-{}'.format(i, remove_tags[i]), use_hessian_lu=use_hessian_lu)
        self_pred_infls.append(np.sum(pred_infls) * len(remove_indices))
        if model.num_classes == 2:
            pred_margin_infls = model.get_influence_on_test_loss(remove_indices, remove_indices, force_refresh=True, batch_size='default',
                                                                 test_indices_from_train=True, # remove_indices refers to training points
                                                                 margins=True, use_hessian_lu=use_hessian_lu,
                                                                 test_description='subset-{}-{}'.format(i, remove_tags[i]))
            self_pred_margin_infls.append(np.sum(pred_margin_infls) * len(remove_indices))

        if (i % n_report == 0):
            cur_time = time.time()
            time_per_retrain = (cur_time - start_time) / (i + 1)
            remaining_time = time_per_retrain * (n - i- 1)
            print('Each self-influence calculation takes {} s, {} s remaining'.format(time_per_retrain, remaining_time))

    start_time = time.time()
    train_losses, test_losses = [], []
    train_margins, test_margins = [], []
    for i, remove_indices in enumerate(remove_subsets):
        if (i % n_report == 0):
            print('Retraining model {} out of {} (tag={})'.format(i, n, remove_tags[i]))

        train_loss, test_loss = retrained_losses(model, remove_indices)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if model.num_classes == 2:
            train_margin, test_margin = get_margins(model)
            train_margins.append(train_margin)
            test_margins.append(test_margin)

        if (i % n_report == 0):
            cur_time = time.time()
            time_per_retrain = (cur_time - start_time) / (i + 1)
            remaining_time = time_per_retrain * (n - i- 1)
            print('Each retraining takes {} s, {} s remaining'.format(time_per_retrain, remaining_time))

    train_margins = np.array(train_margins) if model.num_classes == 2 else None
    test_margins = np.array(test_margins) if model.num_classes == 2 else None

    return np.array(train_losses), np.array(test_losses), train_margins, test_margins, np.array(self_pred_infls), np.array(self_pred_margin_infls)

def get_random_subset(subset_picker_rng, num_train_pts, proportion=default_prop, num=default_num_subsets):
    subsets = []
    for i in range(num):
        subsets.append(subset_picker_rng.choice(num_train_pts, int(num_train_pts * proportion), replace=False))
    return np.array(subsets)

def get_scalar_infl_tails(subset_picker_rng, num_train_pts, pred_infl, proportion=default_prop, num=default_num_subsets):
    assert proportion < 0.5
    window = int(2*proportion*num_train_pts)
    points = int(proportion*num_train_pts)
    scalar_infl_indices = np.argsort(pred_infl)
    pos_subsets, neg_subsets = [], []
    for i in range(num):
        neg_subsets.append(subset_picker_rng.choice(scalar_infl_indices[:window], points, replace=False))
        pos_subsets.append(subset_picker_rng.choice(scalar_infl_indices[-window:], points, replace=False))
    return np.array(neg_subsets), np.array(pos_subsets)

def get_same_grad_dir(subset_picker_rng, num_train_pts, grad_loss, proportion=default_prop, num=default_num_subsets):
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
        subsets.append(subset_picker_rng.choice(cluster_indices, points, replace=False))
    return np.array(subsets), best, labels

def get_same_class_subset(subset_picker_rng, num_train_pts, labels, proportion=default_prop, num=default_num_subsets, test_label=None):
    points = int(proportion*num_train_pts)
    label_vals, counts = np.unique(labels, return_counts=True)
    valid_labels = []
    valid_indices = []
    for i in range(len(label_vals)):
        if counts[i] >= points:
            valid_labels.append(label_vals[i])
            valid_indices.append(list(np.where(labels == label_vals[i])[0]))
    assert len(valid_indices) > 0
    flat = [i for sublist in valid_indices for i in sublist]
    label_to_ind = dict(zip(valid_labels, range(len(valid_indices))))

    subsets = []
    if test_label is not None and int(test_label) not in valid_labels:
        print('Couldn\'t use desired label.')
    for i in range(num):
        if (test_label is None) or (int(test_label) not in valid_labels):
            sample = subset_picker_rng.choice(flat)
            sample_ind = label_to_ind[labels[sample]]
        else:
            sample_ind = label_to_ind[int(test_label)]
        subsets.append(subset_picker_rng.choice(valid_indices[sample_ind], points, replace=False))
    return np.array(subsets)

def get_same_features_subset(subset_picker_rng, num_train_pts, features, labels, proportion=default_prop, num=default_num_subsets):
    if dataset_type == 'hospital' and not center_data:
        indices = np.where(features[:,9]==1)[0]
        indices = np.where(features[indices,1]>5)[0]
        indices = np.where(features[indices,26]==1)[0]
        indices = np.where(features[indices,14]==1)[0]
        indices = np.where(features[indices,3]>5)[0]
        subsets = []
        print('Features subset has {} examples total'.format(len(indices)))
        for i in range(num):
            subsets.append(subset_picker_rng.choice(indices, 4*len(indices)//5, replace=False))
        return subsets
    else:
        print("Warning: unimplemented method to get subsets with the same features")
        return []

initial_training_result = initial_training()
model_name, config_dict, model = initial_training_result[:3]
train_losses, test_losses, train_margins, test_margins, grad_loss = initial_training_result[3:]
print('Finished original training.')

if dataset_type == "hospital":
    test_points = [2267, 54826, 66678, 41567, 485, 25286]
elif dataset_type == "spam":
    test_points = [14, 7, 10, 6, 15, 3]
else:
    test_points = pick_test_points(test_losses)

print('Test points: {}'.format(test_points))
fixed_test_pred_infl, fixed_test_pred_margin_infl = get_fixed_test_influence(model, test_points)

subset_picker_rng = np.random.RandomState(subset_seed)

num_train_pts = model.num_train_examples
tagged_subsets = []

random_subsets = get_random_subset(subset_picker_rng, num_train_pts)
tagged_subsets += [('random', s) for s in random_subsets]
print('Found random subsets.')

same_class_subsets = get_same_class_subset(subset_picker_rng, num_train_pts, model.data_sets.train.labels, test_label=None)
same_class_subset_labels = [model.data_sets.train.labels[s[0]] for s in same_class_subsets]
tagged_subsets += [('random_same_class-{}'.format(label), s) for s, label in zip(same_class_subsets, same_class_subset_labels)]
print('Found same class subsets.')

for pred_infl, test_idx in zip(fixed_test_pred_infl, test_points):
    neg_tail_subsets, pos_tail_subsets = get_scalar_infl_tails(subset_picker_rng, num_train_pts, pred_infl)
    tagged_subsets += [('neg_tail_test-{}'.format(test_idx), s) for s in neg_tail_subsets]
    tagged_subsets += [('pos_tail_test-{}'.format(test_idx), s) for s in pos_tail_subsets]
    print('Found scalar infl tail subsets for test idx {}.'.format(test_idx))
print('Found scalar infl tail subsets.')

same_features_subsets = get_same_features_subset(subset_picker_rng, num_train_pts, model.data_sets.train.x, model.data_sets.train.labels)
tagged_subsets += [('same_features', s) for s in same_features_subsets]
print('Found same features subsets.')

same_grad_subsets, cluster_label, cluster_labels = get_same_grad_dir(subset_picker_rng, num_train_pts, grad_loss)
tagged_subsets += [('same_grad', s) for s in same_grad_subsets]
print('Found same gradient subsets.')

subset_tags = [tag for tag, subset in tagged_subsets]
subset_indices = [subset for tag, subset in tagged_subsets]
retrain_result = retrain(model, subset_indices, subset_tags)
subset_train_losses, subset_test_losses, subset_train_margins, subset_test_margins = retrain_result[:4]
subset_self_pred_infl, subset_self_pred_margin_infl = retrain_result[4:]
print('Finished retraining subsets')

# N = num_train_pt
# K = total_num_subsets
# D = dimension
name_template = 'explore-infl-logreg-data-{}-prop-{}-subsets-{}-subset_seed-{}-center-data-{}'
name_args = (dataset_type, default_prop, default_num_subsets, subset_seed, center_data)
np.savez(os.path.join(out, name_template.format(*name_args)),
         initial_train_losses=train_losses,                             # (N,)
         initial_test_losses=test_losses,                               # (N,)
         initial_train_margins=train_margins,                           # (N,) or None
         initial_test_margins=test_margins,                             # (N,) or None
         grad_loss=grad_loss,                                           # (N, D)
         test_points=test_points,                                       # (6,)
         fixed_test_pred_infl=fixed_test_pred_infl,                     # (6, N)
         fixed_test_pred_margin_infl=fixed_test_pred_margin_infl,       # (6, N)
         subset_tags=subset_tags,                                       # (K,) strings
         subset_indices=subset_indices,                                 # (K,)
         subset_train_losses=subset_train_losses,                       # (K, N)
         subset_test_losses=subset_test_losses,                         # (K, N)
         subset_train_margins=subset_train_margins,                     # (K, N) or None
         subset_test_margins=subset_test_margins,                       # (K, N) or None
         subset_self_pred_infl=subset_self_pred_infl,                   # (K,)
         subset_self_pred_margin_infl=subset_self_pred_margin_infl)     # (K,)
