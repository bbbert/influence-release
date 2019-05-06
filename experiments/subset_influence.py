from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import datasets as ds
import datasets.loader
import datasets.mnist
from datasets.common import DataSet
from experiments.common import Experiment, collect_phases, phase
from experiments.benchmark import benchmark
from experiments.plot import *
from influence.logistic_regression import LogisticRegression

import os
import time
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

@collect_phases
class SubsetInfluenceLogreg(Experiment):
    """
    Test the LogisticRegression model's functionality.
    """
    def __init__(self, config, out_dir=None):
        super(SubsetInfluenceLogreg, self).__init__(config, out_dir)
        self.datasets = ds.loader.load_dataset(**self.config['dataset_config'])
        self.train = self.datasets.train
        self.test = self.datasets.test
        self.validation = self.datasets.validation

        model_dir = os.path.join(self.base_dir, 'models')
        model_config = LogisticRegression.default_config()
        model_config['arch'] = LogisticRegression.infer_arch(self.datasets.train)
        model_config['arch']['fit_intercept'] = True

        # Heuristic for determining maximum batch evaluation sizes without OOM
        D = model_config['arch']['input_dim'] * model_config['arch']['num_classes']
        model_config['grad_batch_size'] =  max(1, self.config['max_memory'] // D)
        model_config['hessian_batch_size'] = max(1, self.config['max_memory'] // (D * D))

        # Set the method for computing inverse HVP
        model_config['inverse_hvp_method'] = self.config['inverse_hvp_method']

        self.model_dir = model_dir
        self.model_config = model_config

        # Convenience member variables
        self.dataset_id = self.config['dataset_config']['dataset_id']
        self.num_train = self.datasets.train.num_examples
        self.num_classes = self.model_config['arch']['num_classes']
        self.num_subsets = self.config['num_subsets']
        if self.subset_choice_type == "types":
            self.subset_size = int(self.num_train * self.config['subset_rel_size'])
        elif self.subset_choice_type == "range":
            self.subset_min_size = int(self.num_train * self.config['subset_min_rel_size'])
            self.subset_max_size = int(self.num_train * self.config['subset_max_rel_size'])

    experiment_id = "ss_logreg"

    @property
    def subset_choice_type(self):
        return self.config.get('subset_choice_type', 'types')

    @property
    def run_id(self):
        if self.subset_choice_type == "types":
            return "{}_ihvp-{}_seed-{}_size-{}_num-{}".format(
                self.config['dataset_config']['dataset_id'],
                self.config['inverse_hvp_method'],
                self.config['subset_seed'],
                self.config['subset_rel_size'],
                self.config['num_subsets'])
        elif self.subset_choice_type == "range":
            return "{}_ihvp-{}_seed-{}_sizes-{}-{}_num-{}".format(
                self.config['dataset_config']['dataset_id'],
                self.config['inverse_hvp_method'],
                self.config['subset_seed'],
                self.config['subset_min_rel_size'],
                self.config['subset_max_rel_size'],
                self.config['num_subsets'])

    def get_model(self):
        if not hasattr(self, 'model'):
            self.model = LogisticRegression(self.model_config, self.model_dir)
        return self.model

    @phase(0)
    def cross_validation(self):
        model = self.get_model()
        res = dict()

        reg_min, reg_max, reg_samples = self.config['normalized_cross_validation_range']
        reg_min *= self.num_train
        reg_max *= self.num_train

        num_folds = self.config['cross_validation_folds']

        regs = np.logspace(np.log10(reg_min), np.log10(reg_max), reg_samples)
        cv_errors = np.zeros_like(regs)
        fold_size = (self.num_train + num_folds - 1) // num_folds
        folds = [(k * num_folds, min((k + 1) * num_folds, self.num_train)) for k in range(num_folds)]

        for i, reg in enumerate(regs):
            with benchmark("Evaluating CV error for reg={}".format(reg)):
                cv_error = 0.0
                for k, fold in enumerate(folds):
                    fold_begin, fold_end = fold
                    train_indices = np.concatenate((np.arange(0, fold_begin), np.arange(fold_end, self.num_train)))
                    val_indices = np.arange(fold_begin, fold_end)

                    model.fit(self.train.subset(train_indices), l2_reg=reg)
                    fold_loss = model.get_total_loss(self.train.subset(val_indices), l2_reg=0)
                    cv_error += fold_loss

            cv_errors[i] = cv_error
            print('Cross-validation error is {} for reg={}.'.format(cv_error, reg))

        best_i = np.argmin(cv_errors)
        best_reg = regs[best_i]
        print('Cross-validation errors: {}'.format(cv_errors))
        print('Selecting weight_decay {}, with error {}.'.format(best_reg, cv_errors[best_i]))

        res['cv_regs'] = regs
        res['cv_errors'] = cv_errors
        res['cv_l2_reg'] = best_reg
        return res

    @phase(1)
    def initial_training(self):
        model = self.get_model()
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        with benchmark("Training original model"):
            model.fit(self.train, l2_reg=l2_reg)
            model.print_model_eval(self.datasets, l2_reg=l2_reg)
            model.save('initial')

        res['initial_train_losses'] = model.get_indiv_loss(self.train)
        res['initial_train_accuracy'] = model.get_accuracy(self.train)
        res['initial_test_losses'] = model.get_indiv_loss(self.test)
        res['initial_test_accuracy'] = model.get_accuracy(self.test)
        if self.num_classes == 2:
            res['initial_train_margins'] = model.get_indiv_margin(self.train)
            res['initial_test_margins'] = model.get_indiv_margin(self.test)

        with benchmark("Computing gradients"):
            res['train_grad_loss'] = model.get_indiv_grad_loss(self.train)

        return res

    @phase(2)
    def pick_test_points(self):
        dataset_id = self.config['dataset_config']['dataset_id']

        # Freeze each set after the first run
        if dataset_id == "hospital":
            fixed_test = [2267, 54826, 66678, 41567, 485, 25286]
        elif dataset_id == "spam":
            fixed_test = [14, 7, 10, 6, 15, 3]
        elif dataset_id == "mnist_small":
            fixed_test = [6172, 2044, 2293, 5305, 324, 3761]
        elif dataset_id == "processed_imageNet":
            fixed_test = [684, 850, 1492, 2357, 480, 2288]
        else:
            test_losses = self.R['initial_test_losses']
            argsort = np.argsort(test_losses)
            high_loss = argsort[-3:] # Pick 3 high loss points
            random_loss = np.random.choice(argsort[:-3], 3, replace=False) # Pick 3 random points

            fixed_test = list(high_loss) + list(random_loss)

        print("Fixed test points: {}".format(fixed_test))
        return { 'fixed_test': fixed_test }

    @phase(3)
    def hessian(self):
        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        if self.config['inverse_hvp_method'] == 'explicit':
            with benchmark("Computing hessian"):
                res['hessian'] = hessian = model.get_hessian(self.train, l2_reg=l2_reg)
        elif self.config['inverse_hvp_method'] == 'cg':
            print("Not computing explicit hessian.")
            res['hessian'] = None

        return res

    @phase(4)
    def fixed_test_influence(self):
        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        hessian = self.R['hessian']
        inverse_hvp_args = {
            'hessian_reg': hessian,
            'dataset': self.train,
            'l2_reg': l2_reg,
            'verbose': False,
            'verbose_cg': True,
        }

        fixed_test = self.R['fixed_test']
        fixed_test_grad_loss = []
        fixed_test_pred_infl = []
        fixed_test_pred_margin_infl = []
        for test_idx in fixed_test:
            single_test_point = self.test.subset([test_idx])

            with benchmark('Scalar infl for all training points on test_idx {}.'.format(test_idx)):
                test_grad_loss = model.get_indiv_grad_loss(single_test_point).reshape(-1, 1)
                test_grad_loss_H_inv = model.get_inverse_hvp(test_grad_loss, **inverse_hvp_args).reshape(-1)
                pred_infl = np.dot(self.R['train_grad_loss'], test_grad_loss_H_inv)
                fixed_test_grad_loss.append(test_grad_loss)
                fixed_test_pred_infl.append(pred_infl)

            if self.num_classes == 2:
                with benchmark('Scalar margin infl for all training points on test_idx {}.'.format(test_idx)):
                    test_grad_margin = model.get_indiv_grad_margin(single_test_point).reshape(-1, 1)
                    test_grad_margin_H_inv = model.get_inverse_hvp(test_grad_margin, **inverse_hvp_args).reshape(-1)
                    pred_margin_infl = np.dot(self.R['train_grad_loss'], test_grad_margin_H_inv)
                    fixed_test_pred_margin_infl.append(pred_margin_infl)

        res['fixed_test_pred_infl'] = np.array(fixed_test_pred_infl)
        if self.num_classes == 2:
            res['fixed_test_pred_margin_infl'] = np.array(fixed_test_pred_margin_infl)

        return res

    def get_random_subsets(self, rng, subset_sizes=None):
        subsets = []
        for i in range(self.num_subsets):
            subset_size = self.subset_size if subset_sizes is None else subset_sizes[i]
            subsets.append(rng.choice(self.num_train, subset_size, replace=False))
        return np.array(subsets)

    def get_scalar_infl_tails(self, rng, pred_infl, subset_sizes=None, fixed_window=None):
        scalar_infl_indices = np.argsort(pred_infl).reshape(-1)
        pos_subsets, neg_subsets = [], []
        for i in range(self.num_subsets):
            subset_size = self.subset_size if subset_sizes is None else subset_sizes[i]
            window = fixed_window if fixed_window is not None else 2 * subset_size
            assert window < self.num_train
            neg_subsets.append(rng.choice(scalar_infl_indices[:window], subset_size, replace=False))
            pos_subsets.append(rng.choice(scalar_infl_indices[-window:], subset_size, replace=False))
        return np.array(neg_subsets), np.array(pos_subsets)

    def get_same_grad_dir(self, rng, train_grad_loss, subset_sizes=None):
        # Using Pigeonhole to guarantee we get a sufficiently large cluster
        if self.subset_choice_type == "range":
            max_rel_size = self.config['subset_max_rel_size']
        else:
            max_rel_size = self.config['subset_rel_size']
        n_clusters = int(math.floor(1 / max_rel_size))

        km = KMeans(n_clusters=n_clusters)
        km.fit(train_grad_loss)
        cluster_labels, centroids = km.labels_, km.cluster_centers_
        _, counts = np.unique(cluster_labels, return_counts=True)

        cluster_indices = [ np.nonzero(cluster_labels == i)[0] for i in range(len(centroids)) ]
        subsets = []
        for i in range(self.num_subsets):
            subset_size = self.subset_size if subset_sizes is None else subset_sizes[i]
            valid_clusters = [ i for i, count in enumerate(counts) if count >= subset_size ]
            if len(valid_clusters) == 0: continue

            cluster_idx = rng.choice(valid_clusters)
            subset = rng.choice(cluster_indices[cluster_idx], subset_size, replace=False)
            subsets.append(subset)
        return np.array(subsets)

    def get_same_class_subsets(self, rng, labels, subset_sizes=None):
        label_vals, label_counts = np.unique(labels, return_counts=True)
        label_indices = [ np.nonzero(labels == label_val)[0] for label_val in label_vals ]

        subsets = []
        for i in range(self.num_subsets):
            subset_size = self.subset_size if subset_sizes is None else subset_sizes[i]
            valid_label_indices = np.nonzero(label_counts >= subset_size)[0]

            if len(valid_label_indices) == 0: continue
            valid_label_idx = rng.choice(valid_label_indices)
            subset = rng.choice(label_indices[valid_label_idx], subset_size, replace=False)
            subsets.append(subset)
        return np.array(subsets)

    def get_same_features_subsets(self, rng, features, labels):
        dataset_id = self.config['dataset_config']['dataset_id']
        center_data = self.config['dataset_config']['center_data']
        if dataset_id == 'hospital' and not center_data:
            indices = np.where(features[:,9]==1)[0]
            indices = np.where(features[indices,1]>5)[0]
            indices = np.where(features[indices,26]==1)[0]
            indices = np.where(features[indices,14]==1)[0]
            indices = np.where(features[indices,3]>5)[0]
            subsets = []
            print('Features subset has {} examples total'.format(len(indices)))
            for i in range(self.num_subsets):
                subsets.append(rng.choice(indices, 4*len(indices)//5, replace=False))
            return subsets
        else:
            print("Warning: unimplemented method to get subsets with the same features")
            return []

    @phase(5)
    def pick_subsets(self):
        rng = np.random.RandomState(self.config['subset_seed'])

        if self.subset_choice_type == "types":
            tagged_subsets = self.pick_subsets_many_types(rng)
        elif self.subset_choice_type == "range":
            tagged_subsets = self.pick_subsets_size_range(rng)

        subset_tags = [tag for tag, subset in tagged_subsets]
        subset_indices = [subset for tag, subset in tagged_subsets]

        subset_sizes = np.unique([len(subset) for tag, subset in tagged_subsets])

        return { 'subset_tags': subset_tags, 'subset_indices': subset_indices }

    def pick_subsets_many_types(self, rng):
        tagged_subsets = []

        with benchmark("Random subsets"):
            random_subsets = self.get_random_subsets(rng)
            tagged_subsets += [('random', s) for s in random_subsets]

        with benchmark("Same class subsets"):
            same_class_subsets = self.get_same_class_subsets(rng, self.train.labels)
            same_class_subset_labels = [self.train.labels[s[0]] for s in same_class_subsets]
            tagged_subsets += [('random_same_class-{}'.format(label), s) for s, label in zip(same_class_subsets, same_class_subset_labels)]

        with benchmark("Scalar infl tail subsets"):
            for pred_infl, test_idx in zip(self.R['fixed_test_pred_infl'], self.R['fixed_test']):
                neg_tail_subsets, pos_tail_subsets = self.get_scalar_infl_tails(rng, pred_infl)
                tagged_subsets += [('neg_tail_test-{}'.format(test_idx), s) for s in neg_tail_subsets]
                tagged_subsets += [('pos_tail_test-{}'.format(test_idx), s) for s in pos_tail_subsets]
                print('Found scalar infl tail subsets for test idx {}.'.format(test_idx))

        with benchmark("Same features subsets"):
            same_features_subsets = self.get_same_features_subsets(rng, self.train.x, self.train.labels)
            tagged_subsets += [('same_features', s) for s in same_features_subsets]

        with benchmark("Same gradient subsets"):
            same_grad_subsets = self.get_same_grad_dir(rng, self.R['train_grad_loss'])
            tagged_subsets += [('same_grad', s) for s in same_grad_subsets]

        return tagged_subsets

    def pick_subsets_size_range(self, rng):
        tagged_subsets = []
        subset_sizes = np.linspace(self.subset_min_size,
                                   self.subset_max_size,
                                   self.num_subsets).astype(np.int)

        with benchmark("Random subsets"):
            random_subsets = self.get_random_subsets(rng, subset_sizes=subset_sizes)
            tagged_subsets += [('random', s) for s in random_subsets]

        with benchmark("Same class subsets"):
            same_class_subsets = self.get_same_class_subsets(rng, self.train.labels, subset_sizes=subset_sizes)
            same_class_subset_labels = [self.train.labels[s[0]] for s in same_class_subsets]
            tagged_subsets += [('random_same_class-{}'.format(label), s) for s, label in zip(same_class_subsets, same_class_subset_labels)]

        with benchmark("Scalar infl tail growing window subsets"):
            for pred_infl, test_idx in zip(self.R['fixed_test_pred_infl'], self.R['fixed_test']):
                neg_tail_subsets, pos_tail_subsets = self.get_scalar_infl_tails(rng, pred_infl,
                                                                                subset_sizes=subset_sizes)
                tagged_subsets += [('neg_tail_test_grow-{}'.format(test_idx), s) for s in neg_tail_subsets]
                tagged_subsets += [('pos_tail_test_grow-{}'.format(test_idx), s) for s in pos_tail_subsets]
                print('Found scalar infl tail (growing window) subsets for test idx {}.'.format(test_idx))

        with benchmark("Scalar infl tail fixed window subsets"):
            fixed_window = 2 * self.subset_max_size
            for pred_infl, test_idx in zip(self.R['fixed_test_pred_infl'], self.R['fixed_test']):
                neg_tail_subsets, pos_tail_subsets = self.get_scalar_infl_tails(rng, pred_infl,
                                                                                subset_sizes=subset_sizes,
                                                                                fixed_window=fixed_window)
                tagged_subsets += [('neg_tail_test_fixed-{}'.format(test_idx), s) for s in neg_tail_subsets]
                tagged_subsets += [('pos_tail_test_fixed-{}'.format(test_idx), s) for s in pos_tail_subsets]
                print('Found scalar infl tail (fixed window) subsets for test idx {}.'.format(test_idx))

        return tagged_subsets

    @phase(6)
    def retrain(self):
        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']
        n, n_report = len(subset_indices), max(len(subset_indices) // 100, 1)

        start_time = time.time()
        train_losses, test_losses = [], []
        train_margins, test_margins = [], []
        for i, remove_indices in enumerate(subset_indices):
            if (i % n_report == 0):
                print('Retraining model {} out of {} (tag={}, size={})'.format(
                    i, n, subset_tags[i], len(remove_indices)))

            s = np.ones(self.num_train)
            s[remove_indices] = 0

            model.warm_fit(self.train, s, l2_reg=l2_reg)
            model.save('subset_{}'.format(i))
            train_losses.append(model.get_indiv_loss(self.train))
            test_losses.append(model.get_indiv_loss(self.test))
            if model.num_classes == 2:
                train_margins.append(model.get_indiv_margin(self.train))
                test_margins.append(model.get_indiv_margin(self.test))

            if (i % n_report == 0):
                cur_time = time.time()
                time_per_retrain = (cur_time - start_time) / (i + 1)
                remaining_time = time_per_retrain * (n - i - 1)
                print('Each retraining takes {} s, {} s remaining'.format(time_per_retrain, remaining_time))

        res['subset_train_losses'] = np.array(train_losses)
        res['subset_test_losses'] = np.array(test_losses)

        if self.num_classes == 2:
            res['subset_train_margins'] = np.array(train_margins)
            res['subset_test_margins'] = np.array(test_margins)

        return res

    @phase(7)
    def compute_self_pred_infl(self):
        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']
        n, n_report = len(subset_indices), max(len(subset_indices) // 100, 1)

        hessian = self.R['hessian']
        inverse_hvp_args = {
            'hessian_reg': hessian,
            'dataset': self.train,
            'l2_reg': l2_reg,
            'verbose': False,
            'verbose_cg': True,
        }
        train_grad_loss = self.R['train_grad_loss']

        # It is important that the influence gets calculated before the model is retrained,
        # so that the parameters are the original parameters
        start_time = time.time()
        subset_pred_dparam = []
        self_pred_infls = []
        self_pred_margin_infls = []
        for i, remove_indices in enumerate(subset_indices):
            if (i % n_report == 0):
                print('Computing self-influences for subset {} out of {} (tag={})'.format(i, n, subset_tags[i]))

            grad_loss = np.sum(train_grad_loss[remove_indices, :], axis=0)
            H_inv_grad_loss = model.get_inverse_hvp(grad_loss.reshape(-1, 1), **inverse_hvp_args).reshape(-1)
            pred_infl = np.dot(grad_loss, H_inv_grad_loss)
            subset_pred_dparam.append(H_inv_grad_loss)
            self_pred_infls.append(pred_infl)

            if model.num_classes == 2:
                s = np.zeros(self.num_train)
                s[remove_indices] = 1
                grad_margin = model.get_total_grad_margin(self.train, s)
                pred_margin_infl = np.dot(grad_margin, H_inv_grad_loss)
                self_pred_margin_infls.append(pred_margin_infl)

            if (i % n_report == 0):
                cur_time = time.time()
                time_per_vp = (cur_time - start_time) / (i + 1)
                remaining_time = time_per_vp * (n - i - 1)
                print('Each self-influence calculation takes {} s, {} s remaining'.format(time_per_vp, remaining_time))

        res['subset_pred_dparam'] = np.array(subset_pred_dparam)
        res['subset_self_pred_infl'] = np.array(self_pred_infls)
        if self.num_classes == 2:
            res['subset_self_pred_margin_infl'] = np.array(self_pred_margin_infls)

        return res

    @phase(8)
    def compute_actl_infl(self):
        res = dict()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']

        # Helper to collate fixed test infl and subset self infl on a quantity q
        def compute_collate_infl(fixed_test, fixed_test_pred_infl_q,
                                 initial_train_q, initial_test_q,
                                 subset_train_q, subset_test_q):
            subset_fixed_test_actl_infl = subset_test_q[:, fixed_test] - initial_test_q[fixed_test]
            subset_fixed_test_pred_infl = np.array([
                np.sum(fixed_test_pred_infl_q[:, remove_indices], axis=1).reshape(-1)
                for remove_indices in subset_indices])
            subset_self_actl_infl = np.array([
                np.sum(subset_train_q[i][remove_indices]) - np.sum(initial_train_q[remove_indices])
                for i, remove_indices in enumerate(subset_indices)])
            return subset_fixed_test_actl_infl, subset_fixed_test_pred_infl, subset_self_actl_infl

        # Compute influences on loss
        res['subset_fixed_test_actl_infl'], \
        res['subset_fixed_test_pred_infl'], \
        res['subset_self_actl_infl'] = compute_collate_infl(
            *[self.R[key] for key in ["fixed_test", "fixed_test_pred_infl",
                                      "initial_train_losses", "initial_test_losses",
                                      "subset_train_losses", "subset_test_losses"]])

        if self.num_classes == 2:
            # Compute influences on margin
            res['subset_fixed_test_actl_margin_infl'], \
            res['subset_fixed_test_pred_margin_infl'], \
            res['subset_self_actl_margin_infl'] = compute_collate_infl(
                *[self.R[key] for key in ["fixed_test", "fixed_test_pred_margin_infl",
                                          "initial_train_margins", "initial_test_margins",
                                          "subset_train_margins", "subset_test_margins"]])

        return res

    @phase(9)
    def newton(self):
        if self.config['skip_newton']:
            return dict()

        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        # The Newton approximation is obtained by evaluating
        # -g(|w|, theta_0)^T H(s+w, theta_0)^{-1} g(w, theta_0)
        # where w is the difference in weights. Since we already have the full
        # hessian H_reg(s), we can compute H(w) (with no regularization) and
        # use it to update H_reg(s+w) = H_reg(s) + H(w) instead.

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']
        n, n_report = len(subset_indices), max(len(subset_indices) // 100, 1)

        hessian = self.R['hessian']
        train_grad_loss = self.R['train_grad_loss']

        # It is important that the gradients get calculated on the original model
        # so that the parameters are the original parameters
        start_time = time.time()
        subset_newton_dparam = []
        self_newton_infls = []
        self_newton_margin_infls = []
        for i, remove_indices in enumerate(subset_indices):
            if (i % n_report == 0):
                print('Computing Newton self-influences for subset {} out of {} (tag={})'.format(i, n, subset_tags[i]))

            grad_loss = np.sum(train_grad_loss[remove_indices, :], axis=0).reshape(-1, 1)
            if self.config['inverse_hvp_method'] == 'explicit':
                hessian_sw = hessian - model.get_hessian(self.train.subset(remove_indices),
                                                         np.ones(len(remove_indices)), l2_reg=0, verbose=False)
                try:
                    H_inv_grad_loss = model.get_inverse_vp(hessian_sw, grad_loss).reshape(-1)
                except:
                    # floating-point error accumulation can cause the updated matrix to not be positive definite
                    hessian_sw += np.ones(hessian_sw.shape[0]) * 1e-9
                    H_inv_grad_loss = model.get_inverse_vp(hessian_sw, grad_loss).reshape(-1)
            elif self.config['inverse_hvp_method'] == 'cg':
                sample_weights = np.ones(self.num_train)
                sample_weights[remove_indices] = 0
                inverse_hvp_args = {
                    'dataset': self.train,
                    'sample_weights': sample_weights,
                    'l2_reg': l2_reg,
                    'verbose': False,
                    'verbose_cg': True,
                }
                H_inv_grad_loss = model.get_inverse_hvp(grad_loss, **inverse_hvp_args).reshape(-1)

            newton_infl = np.dot(grad_loss.reshape(-1), H_inv_grad_loss)
            subset_newton_dparam.append(H_inv_grad_loss)
            self_newton_infls.append(newton_infl)

            if model.num_classes == 2:
                s = np.zeros(self.num_train)
                s[remove_indices] = 1
                grad_margin = model.get_total_grad_margin(self.train, s)
                newton_margin_infl = np.dot(grad_margin, H_inv_grad_loss)
                self_newton_margin_infls.append(newton_margin_infl)

            if (i % n_report == 0):
                cur_time = time.time()
                time_per_vp = (cur_time - start_time) / (i + 1)
                remaining_time = time_per_vp * (n - i - 1)
                print('Each newton self-influence calculation takes {} s, {} s remaining'.format(time_per_vp, remaining_time))

        res['subset_newton_dparam'] = np.array(subset_newton_dparam)
        res['subset_self_newton_infl'] = np.array(self_newton_infls)
        if self.num_classes == 2:
            res['subset_self_newton_margin_infl'] = np.array(self_newton_margin_infls)

        return res

    @phase(10)
    def fixed_test_newton(self):
        if self.config['skip_newton']:
            return dict()

        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        hessian = self.R['hessian']
        fixed_test = self.R['fixed_test']
        test_grad_loss = model.get_indiv_grad_loss(self.test.subset(fixed_test))
        train_grad_loss = self.R['train_grad_loss']

        if self.num_classes == 2:
            test_grad_margin = model.get_indiv_grad_margin(self.test.subset(fixed_test))

        n, n_report = self.num_train, max(self.num_train // 100, 1)

        start_time = time.time()
        fixed_test_newton_infl = []
        fixed_test_newton_margin_infl = []
        for i in range(self.num_train):
            if (i % n_report == 0):
                print('Computing fixed test Newton influences for train idx {} out of {}'.format(i, n))

            grad_loss = train_grad_loss[i, :].reshape(-1, 1)
            if self.config['inverse_hvp_method'] == 'explicit':
                hessian_sw = hessian - model.get_hessian(self.train.subset([i]),
                                                         np.ones(1), l2_reg=0, verbose=False)
                try:
                    H_inv_grad_loss = model.get_inverse_vp(hessian_sw, grad_loss).reshape(-1)
                except:
                    # floating-point error accumulation can cause the updated matrix to not be positive definite
                    hessian_sw += np.ones(hessian_sw.shape[0]) * 1e-9
                    H_inv_grad_loss = model.get_inverse_vp(hessian_sw, grad_loss).reshape(-1)
            elif self.config['inverse_hvp_method'] == 'cg':
                sample_weights = np.ones(self.num_train)
                sample_weights[i] = 0
                inverse_hvp_args = {
                    'dataset': self.train,
                    'sample_weights': sample_weights,
                    'l2_reg': l2_reg,
                    'verbose': False,
                    'verbose_cg': True,
                }
                H_inv_grad_loss = model.get_inverse_hvp(grad_loss, **inverse_hvp_args).reshape(-1)

            newton_infl = np.dot(test_grad_loss, H_inv_grad_loss)
            fixed_test_newton_infl.append(newton_infl)

            if self.num_classes == 2:
                newton_margin_infl = np.dot(test_grad_margin, H_inv_grad_loss)
                fixed_test_newton_margin_infl.append(newton_margin_infl)

            if (i % n_report == 0):
                cur_time = time.time()
                time_per_vp = (cur_time - start_time) / (i + 1)
                remaining_time = time_per_vp * (n - i - 1)
                print('Each fixed test Newton influence calculation takes {} s, {} s remaining'.format(time_per_vp, remaining_time))

        subset_indices = self.R['subset_indices']
        res['fixed_test_newton_infl'] = np.array(fixed_test_newton_infl).T
        res['subset_fixed_test_newton_infl'] = np.array([
            np.sum(res['fixed_test_newton_infl'][:, remove_indices], axis=1).reshape(-1)
            for remove_indices in subset_indices])

        if self.num_classes == 2:
            res['fixed_test_newton_margin_infl'] = np.array(fixed_test_newton_margin_infl).T
            res['subset_fixed_test_newton_margin_infl'] = np.array([
                np.sum(res['fixed_test_newton_margin_infl'][:, remove_indices], axis=1).reshape(-1)
                for remove_indices in subset_indices])

        return res

    @phase(11)
    def param_changes(self):
        model = self.get_model()
        res = dict()

        model.load('initial')
        initial_param = model.get_params_flat()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']
        n, n_report = len(subset_indices), max(len(subset_indices) // 100, 1)

        # Calculate actual changes in parameters
        subset_dparam = []
        subset_train_acc, subset_test_acc = [], []
        for i, remove_indices in enumerate(subset_indices):
            model.load('subset_{}'.format(i))
            param = model.get_params_flat()
            subset_dparam.append(param - initial_param)
            subset_train_acc.append(model.get_accuracy(self.train))
            subset_test_acc.append(model.get_accuracy(self.test))
        res['subset_dparam'] = np.array(subset_dparam)
        res['subset_train_accuracy'] = np.array(subset_train_acc)
        res['subset_test_accuracy'] = np.array(subset_test_acc)

        return res

    @phase(12)
    def param_change_norms(self):
        if self.config['skip_param_change_norms']:
            return dict()

        res = dict()
        model = self.get_model()
        l2_reg = self.R['cv_l2_reg']
        model.load('initial')

        # Compute l2 norm of gradient
        train_grad_loss = self.R['train_grad_loss']
        res['subset_grad_loss_l2_norm'] = np.array([
            np.linalg.norm(np.sum(train_grad_loss[remove_indices, :], axis=0))
            for remove_indices in self.R['subset_indices']])

        # Compute l2 norms and norms under the Hessian metric of parameter changes
        l2_reg = self.R['cv_l2_reg']
        hessian = self.R['hessian']
        for dparam_type in ('subset_dparam', 'subset_pred_dparam', 'subset_newton_dparam'):
            dparam = self.R[dparam_type]
            res[dparam_type + '_l2_norm'] = np.linalg.norm(dparam, axis=1)
            if self.config['inverse_hvp_method'] == 'explicit':
                hvp = np.dot(dparam, hessian)
            else:
                hvp = model.get_hvp(dparam.T, self.train, l2_reg=l2_reg)
            res[dparam_type + '_hessian_norm'] = np.sqrt(np.sum(dparam * hvp, axis=1))

        return res

    @phase(13)
    def z_norms(self):
        if self.config['skip_z_norms'] or self.num_classes != 2:
            return dict()

        res = dict()
        model = self.get_model()
        l2_reg = self.R['cv_l2_reg']
        model.load('initial')

        inverse_hvp_args = {
            'hessian_reg': self.R['hessian'],
            'dataset': self.train,
            'l2_reg': l2_reg,
            'verbose': False,
            'verbose_cg': True,
        }

        # z_i = sqrt(sigma''_i) x_i so that H = ZZ^T
        res['zs'] = zs = model.get_zs(self.train)
        ihvp_zs = model.get_inverse_hvp(zs.T, **inverse_hvp_args).T
        res['z_norms'] = np.linalg.norm(zs, axis=1)
        res['z_hessian_norms'] = np.sqrt(np.sum(zs * ihvp_zs, axis=1))

        return res

    def plot_z_norms(self):
        if 'z_norms' not in self.R: return

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
        plot_z_norms(ax[0][0], self.R['z_norms'], self.dataset_id)
        fig.savefig(os.path.join(self.plot_dir, 'z_norms.png'),
                    bbox_inches='tight')

    def get_simple_subset_tags(self):
        def simplify_tag(tag):
            if '-' in tag: return tag.split('-')[0]
            return tag
        return map(simplify_tag, self.R['subset_tags'])

    def get_subtitle(self):
        if self.subset_choice_type == "types":
            subtitle='{}, {} subsets per type, proportion {}'.format(
                self.dataset_id, self.num_subsets, self.config['subset_rel_size'])
        elif self.subset_choice_type == "range":
            subtitle='{}, {} subsets per type, proportion {}-{}'.format(
                self.dataset_id, self.num_subsets,
                self.config['subset_min_rel_size'],
                self.config['subset_max_rel_size'])
        return subtitle

    def plot_self_influence(self):
        if 'subset_self_actl_infl' not in self.R: return
        if 'subset_self_pred_infl' not in self.R: return

        subset_tags = self.get_simple_subset_tags()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
        plot_influence_correlation(ax[0][0],
                                   self.R['subset_self_actl_infl'],
                                   self.R['subset_self_pred_infl'],
                                   label=subset_tags,
                                   title='Group self-influence',
                                   subtitle=self.get_subtitle())
        fig.savefig(os.path.join(self.plot_dir, 'self-influence_loss.png'),
                    bbox_inches='tight')

        if self.num_classes == 2:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
            plot_influence_correlation(ax[0][0],
                                       self.R['subset_self_actl_margin_infl'],
                                       self.R['subset_self_pred_margin_infl'],
                                       label=subset_tags,
                                       title='Group margin self-influence',
                                       subtitle=self.get_subtitle())
            fig.savefig(os.path.join(self.plot_dir, 'self-influence_margin.png'),
                        bbox_inches='tight')

    def plot_fixed_test_influence(self):
        if 'subset_fixed_test_actl_infl' not in self.R: return
        if 'subset_fixed_test_pred_infl' not in self.R: return

        subset_tags = self.get_simple_subset_tags()

        for i, test_idx in enumerate(self.R['fixed_test']):
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
            plot_influence_correlation(ax[0][0],
                                       self.R['subset_fixed_test_actl_infl'][:, i],
                                       self.R['subset_fixed_test_pred_infl'][:, i],
                                       label=subset_tags,
                                       title='Group influence on test pt {}'.format(test_idx),
                                       subtitle=self.get_subtitle())
            fig.savefig(os.path.join(self.plot_dir, 'fixed-test-influence-{}_loss.png'.format(test_idx)),
                        bbox_inches='tight')

            if self.num_classes == 2:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
                plot_influence_correlation(ax[0][0],
                                           self.R['subset_fixed_test_actl_margin_infl'][:, i],
                                           self.R['subset_fixed_test_pred_margin_infl'][:, i],
                                           label=subset_tags,
                                           title='Group margin influence on test pt {}'.format(test_idx),
                                           subtitle=self.get_subtitle())
                fig.savefig(os.path.join(self.plot_dir, 'fixed-test-influence-{}_margin.png'.format(test_idx)),
                            bbox_inches='tight')

    def plot_subset_sizes(self):
        if self.subset_choice_type != "range": return

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
        plot_against_subset_size(ax[0][0],
                                 self.R['subset_tags'],
                                 self.R['subset_indices'],
                                 self.R['subset_self_pred_infl'],
                                 title='Group self-influence',
                                 ylabel='Self-influence',
                                 subtitle=self.get_subtitle())
        fig.savefig(os.path.join(self.plot_dir, 'sizes_self-influence_loss.png'),
                    bbox_inches='tight')

        for i, test_idx in enumerate(self.R['fixed_test']):
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
            plot_against_subset_size(ax[0][0],
                                     self.R['subset_tags'],
                                     self.R['subset_indices'],
                                     self.R['subset_fixed_test_pred_infl'][:, i],
                                     title='Group influence on test pt {}'.format(test_idx),
                                     subtitle=self.get_subtitle())
            fig.savefig(os.path.join(self.plot_dir, 'sizes_fixed-test-influence-{}_loss.png'.format(test_idx)),
                        bbox_inches='tight')


    def plot_all(self):
        self.plot_self_influence()
        self.plot_fixed_test_influence()
        self.plot_z_norms()
        self.plot_subset_sizes()
