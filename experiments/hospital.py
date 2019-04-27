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
from influence.logistic_regression import LogisticRegression

import os
import time
import math
import numpy as np
import tensorflow as tf
import itertools

from sklearn.cluster import KMeans
from tensorflow.contrib.learn.python.learn.datasets import base

@collect_phases
class HospitalGroups(Experiment):
    """
    Explore group influence in the hospital dataset
    """
    def __init__(self, config, out_dir=None):
        super(HospitalGroups, self).__init__(config, out_dir)
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
        self.num_train = self.datasets.train.num_examples
        self.num_classes = self.model_config['arch']['num_classes']
        self.subset_seed = self.config['subset_seed']

    experiment_id = "hospital_groups"

    @property
    def run_id(self):
        return "{}_seed-{}".format(
            self.config['dataset_config']['dataset_id'],
            self.config['subset_seed'])

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
        res['fixed_test'] = [2267, 54826, 66678, 41567, 485, 25286]

        with benchmark("Computing gradients"):
            res['train_grad_loss'] = model.get_indiv_grad_loss(self.train)

        return res

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
    def pick_subsets(self):
        model = self.get_model()
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        D = self.train.x.shape[1]
        categorical = [
            i for i in range(D)
            if len(np.unique(self.train.x[:, i])) == 2
        ]

        demographic = set([
            i for i in categorical
            if any(kw in self.train.feature_names[i] for kw in ('race', 'gender', 'age'))
        ])

        # Find intersections
        tagged_subsets = []

        for i in categorical:
            i_name = self.train.feature_names[i]
            indices = np.where(self.train.x[:, i] == 1)[0]
            tag = "demographic={},{}={}".format(i in demographic, i_name, 1)
            if len(indices) <= 1: continue
            tagged_subsets.append((tag, indices))

        for i, j in itertools.combinations(categorical, 2):
            a, b = 1, 1
            i_name = self.train.feature_names[i]
            j_name = self.train.feature_names[j]
            indices = np.where(np.logical_and(self.train.x[:, i] == a, self.train.x[:, j] == b))[0]
            tag = "demographic={},{}={}_{}={}".format(i in demographic and j in demographic,
                                                      i_name, a, j_name, b)
            if len(indices) <= 1: continue
            tagged_subsets.append((tag, indices))

        subset_tags = [tag for tag, subset in tagged_subsets]
        subset_indices = [subset for tag, subset in tagged_subsets]

        print("Found {} subsets".format(len(tagged_subsets)))

        return { 'subset_tags': subset_tags, 'subset_indices': subset_indices }

    @phase(5)
    def compute_self_infl(self):
        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        res['self_infl'] = self.compute_indiv_self_influence(model, l2_reg, self.R['train_grad_loss'], self.train)
        res['subset_self_infl'] = self.compute_subset_self_influences(model, l2_reg, self.R['train_grad_loss'], self.train)

        return res

    def compute_indiv_self_influence(self, model, l2_reg, train_grad_loss, dataset):
        inverse_hvp_args = {
            'hessian_reg': self.R['hessian'],
            'dataset': dataset,
            'l2_reg': l2_reg,
            'verbose': False,
            'verbose_cg': True,
        }

        H_inv_train_grad_loss = model.get_inverse_hvp(train_grad_loss.T, **inverse_hvp_args).T
        self_infl = np.sum(np.multiply(train_grad_loss, H_inv_train_grad_loss), axis=1)
        return self_infl

    def compute_subset_self_influence(self, model, l2_reg, remove_indices, train_grad_loss, dataset):
        inverse_hvp_args = {
            'hessian_reg': self.R['hessian'],
            'dataset': dataset,
            'l2_reg': l2_reg,
            'verbose': False,
            'verbose_cg': True,
        }

        grad_loss = np.sum(train_grad_loss[remove_indices, :], axis=0)
        H_inv_grad_loss = model.get_inverse_hvp(grad_loss.reshape(-1, 1), **inverse_hvp_args).reshape(-1)
        infl = np.dot(grad_loss, H_inv_grad_loss)
        return infl

    def compute_subset_self_influences(self, model, l2_reg, train_grad_loss, dataset, verbose=True):
        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']
        n, n_report = len(subset_indices), max(len(subset_indices) // 100, 1)

        # It is important that the influence gets calculated before the model is retrained,
        # so that the parameters are the original parameters
        start_time = time.time()
        self_infls = []
        for i, remove_indices in enumerate(subset_indices):
            if (i % n_report == 0) and verbose:
                print('Computing self-influences for subset {} out of {} (tag={})'.format(i, n, subset_tags[i]))

            infl = self.compute_subset_self_influence(model, l2_reg, remove_indices,
                                                      train_grad_loss, dataset)
            self_infls.append(infl)

            if (i % n_report == 0) and verbose:
                cur_time = time.time()
                time_per_vp = (cur_time - start_time) / (i + 1)
                remaining_time = time_per_vp * (n - i - 1)
                print('Each self-influence calculation takes {} s, {} s remaining'.format(time_per_vp, remaining_time))

        return np.array(self_infls)

    @phase(6)
    def test_infl(self):
        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        res['fixed_test_infl'], res['avg_test_infl'] = \
            self.compute_test_influences(model, l2_reg, self.R['train_grad_loss'], self.train)

        return res

    def compute_test_influences(self, model, l2_reg, train_grad_loss, dataset):
        inverse_hvp_args = {
            'hessian_reg': self.R['hessian'],
            'dataset': dataset,
            'l2_reg': l2_reg,
            'verbose': False,
            'verbose_cg': True,
        }

        fixed_test_infl = []
        for test_idx in self.R['fixed_test']:
            single_test_point = self.test.subset([test_idx])

            with benchmark('Scalar infl for all training points on test_idx {}.'.format(test_idx)):
                test_grad_loss = model.get_indiv_grad_loss(single_test_point, verbose=False).reshape(-1, 1)
                test_grad_loss_H_inv = model.get_inverse_hvp(test_grad_loss, **inverse_hvp_args).reshape(-1)
                infl = np.dot(train_grad_loss, test_grad_loss_H_inv)
                fixed_test_infl.append(infl)

        avg_test_grad_loss = model.get_total_grad_loss(self.test, l2_reg=l2_reg).reshape(-1, 1) / self.test.num_examples
        H_inv_avg_test_grad_loss = model.get_inverse_hvp(avg_test_grad_loss, **inverse_hvp_args).reshape(-1)
        avg_test_infl = np.dot(train_grad_loss, H_inv_avg_test_grad_loss)

        return np.array(fixed_test_infl), np.array(avg_test_infl)

    @phase(7)
    def compute_subset_test_infl(self):
        res = dict()

        res['subset_fixed_test_infl'], res['subset_avg_test_infl'] = \
            self.collate_subset_test_influences(self.R['fixed_test_infl'],
                                                self.R['avg_test_infl'])

        return res

    def collate_subset_test_influences(self, fixed_test_infl, avg_test_infl):
        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']

        # Compute influences on loss
        subset_fixed_test_infl = np.array([
            np.sum(fixed_test_infl[:, remove_indices], axis=1).reshape(-1)
            for remove_indices in subset_indices])
        avg_test_infl = np.array([
            np.sum(avg_test_infl[remove_indices])
            for remove_indices in subset_indices])

        return subset_fixed_test_infl, avg_test_infl

    def perturb_train(self, indices, new_label_fn):
        x = self.train.x.copy()
        labels = self.train.labels.copy()
        for i in indices:
            labels[i] = new_label_fn(x[i, :], labels[i])
        return DataSet(x, labels, feature_names=self.train.feature_names)

    @phase(8)
    def perturb_subsets(self):
        model = self.get_model()
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']
        size_threshold = self.num_train / 4
        perturb_subsets = [ i for i, subset in enumerate(subset_indices)
                            if len(subset) <= size_threshold ]

        # Perturn a random sample of viable subsets
        num_perturb = self.config['num_perturb']
        rng = np.random.RandomState(self.subset_seed)
        rng.shuffle(perturb_subsets)
        perturb_subsets = perturb_subsets[:num_perturb]

        n, n_report = len(perturb_subsets), max(len(perturb_subsets) // 100, 1)

        # For each small subset, try perturbing its labels, and see if its group/indiv influence rank changes
        start_time = time.time()
        perturb_self_infl = []
        perturb_subset_self_infl = []
        perturb_fixed_test_infl = []
        perturb_avg_test_infl = []
        perturb_subset_fixed_test_infl = []
        perturb_subset_avg_test_infl = []
        for i, subset_idx in enumerate(perturb_subsets):
            if (i % n_report == 0):
                print('Perturbing subset {} out of {} (tag={})'.format(i, n, subset_tags[subset_idx]))

            indices = subset_indices[subset_idx]
            flip_label = lambda x, label: 1 - label
            set_to_0 = lambda x, label: 0
            set_to_1 = lambda x, label: 1
            perturbation = flip_label
            perturbed_dataset = self.perturb_train(indices, perturbation)

            # retrain
            model.load('initial')
            model.warm_fit(perturbed_dataset, l2_reg=l2_reg)
            model.print_model_eval(base.Datasets(train=perturbed_dataset, test=self.test, validation=None), l2_reg=l2_reg)

            train_grad_loss = model.get_indiv_grad_loss(perturbed_dataset)
            self_infl = self.compute_indiv_self_influence(model, l2_reg, train_grad_loss, perturbed_dataset)
            subset_self_infl = self.compute_subset_self_influences(model, l2_reg, train_grad_loss, perturbed_dataset,
                                                                   verbose=False)

            fixed_test_infl, avg_test_infl = \
                self.compute_test_influences(model, l2_reg, train_grad_loss, perturbed_dataset)
            subset_fixed_test_infl, subset_avg_test_infl = \
                self.collate_subset_test_influences(fixed_test_infl, avg_test_infl)

            perturb_self_infl.append(self_infl)
            perturb_subset_self_infl.append(subset_self_infl)
            perturb_fixed_test_infl.append(fixed_test_infl)
            perturb_avg_test_infl.append(avg_test_infl)
            perturb_subset_fixed_test_infl.append(subset_fixed_test_infl)
            perturb_subset_avg_test_infl.append(subset_avg_test_infl)

            if (i % n_report == 0):
                cur_time = time.time()
                time_per_vp = (cur_time - start_time) / (i + 1)
                remaining_time = time_per_vp * (n - i - 1)
                print('Each perturbed subset takes {} s, {} s remaining'.format(time_per_vp, remaining_time))

        res['perturb_subsets'] = perturb_subsets
        res['perturb_self_infl'] = perturb_self_infl
        res['perturb_subset_self_infl'] = perturb_subset_self_infl
        res['perturb_fixed_test_infl'] = perturb_fixed_test_infl
        res['perturb_avg_test_infl'] = perturb_avg_test_infl
        res['perturb_subset_fixed_test_infl'] = perturb_subset_fixed_test_infl
        res['perturb_subset_avg_test_infl'] = perturb_subset_avg_test_infl

        return res
