from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import datasets as ds
import datasets.loader
import datasets.mnist
from datasets.common import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base

from experiments.common import Experiment, collect_phases, phase
from experiments.benchmark import benchmark
from experiments.plot import *
from influence.logistic_regression import LogisticRegression

import os
import time
import math
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg
import itertools

@collect_phases
class Counterexamples(Experiment):
    """
    Synthesize toy datasets and find counterexamples to possible
    properties of influence approximations
    """
    def __init__(self, config, out_dir=None):
        super(Counterexamples, self).__init__(config, out_dir)
        self.dataset_id = config['dataset_id']

    experiment_id = "counterexamples"

    @property
    def run_id(self):
        return "{}".format(self.dataset_id)

    @phase(0)
    def generate_datasets(self):
        res = dict()

        rng = np.random.RandomState(self.config['seed'])

        # Separated Gaussian mixture
        N_per_class, D, separation = 20, 5, 0.5
        separator = rng.normal(0, 1, size=(D,))
        separator = separator * separation / np.linalg.norm(separator) / 2
        def generate_gaussian_mixture():
            X_pos = rng.normal(0, 1, size=(N_per_class, D)) + separator
            X_neg = rng.normal(0, 1, size=(N_per_class, D)) - separator
            X = np.vstack([X_pos, X_neg])
            Y = np.hstack([np.zeros(N_per_class), np.ones(N_per_class)])
            indices = np.arange(Y.shape[0])
            rng.shuffle(indices)
            return X[indices, :], Y[indices]

        res['gauss_train_X'], res['gauss_train_Y'] = generate_gaussian_mixture()
        res['gauss_test_X'], res['gauss_test_Y'] = generate_gaussian_mixture()

        # Fixed dataset
        X_fixed = np.array([[1, 0], [1, 0], [1, 0],
                            [0, 1], [0, 1], [0, 1]])
        Y_fixed = np.array([0, 0, 0, 1, 1, 1])
        X_confuse = np.array([[0.75, 0.25], [0.25, 0.75]])
        Y_confuse = np.array([1, 0])
        X = np.vstack([X_fixed, X_confuse])
        Y = np.hstack([Y_fixed, Y_confuse])
        indices = np.arange(Y.shape[0])
        rng.shuffle(indices)
        X, Y = X[indices, :], Y[indices]

        res['fixed_train_X'], res['fixed_train_Y'] = X, Y
        res['fixed_test_X'], res['fixed_test_Y'] = X, Y

        # Repeats dataset
        N_random, N_unique, D = 60, 30, 30
        X_unique = rng.normal(0, 1, (1, D))
        X_unique /= np.linalg.norm(X_unique)
        while X_unique.shape[0] < N_unique:
            X_new = rng.normal(0, 1, (1, D))
            new_rank = np.linalg.matrix_rank(np.vstack([X_unique, X_new]))
            if new_rank == X_unique.shape[0]: continue
            X_new -= np.dot(np.dot(X_unique.T, X_unique), X_new.T).T
            if np.linalg.norm(X_new) < 1e-3: continue
            X_new /= np.linalg.norm(X_new)
            X_unique = np.vstack([X_unique, X_new])
        axis = rng.normal(0, 1, (D,))
        Y_unique = rng.randint(0, 2, (N_unique,))

        X, Y = np.zeros((0, D)), np.zeros(0)
        for i in range(N_unique):
            X = np.vstack([X, np.repeat(X_unique[np.newaxis, i, :], i + 1, axis=0)])
            Y = np.hstack([Y, np.repeat(Y_unique[i], i + 1)])

        X_random = rng.normal(0, 0.1, (N_random, D))
        Y_random = rng.randint(0, 2, (N_random,))
        X = np.vstack([X, X_random])
        Y = np.hstack([Y, Y_random])

        res['repeats_train_X'], res['repeats_train_Y'] = X, Y
        res['repeats_test_X'], res['repeats_test_Y'] = X, Y
        res['repeats_N_unique'] = N_unique

        # Separated Gaussian mixture, high dimension
        N_per_class, D, separation = 20, 10, 1
        separator = rng.normal(0, 1, size=(D,))
        separator = separator * separation / np.linalg.norm(separator) / 2
        def generate_gaussian_mixture():
            X_pos = rng.normal(0, 1, size=(N_per_class, D)) + separator
            X_neg = rng.normal(0, 1, size=(N_per_class, D)) - separator
            X = np.vstack([X_pos, X_neg])
            Y = np.hstack([np.zeros(N_per_class), np.ones(N_per_class)])
            indices = np.arange(Y.shape[0])
            rng.shuffle(indices)
            return X[indices, :], Y[indices]

        res['gauss2_train_X'], res['gauss2_train_Y'] = generate_gaussian_mixture()
        res['gauss2_test_X'], res['gauss2_test_Y'] = generate_gaussian_mixture()

        return res

    def get_dataset(self, dataset_id=None):
        dataset_id = dataset_id if dataset_id is not None else self.dataset_id
        if not hasattr(self, 'datasets'):
            self.datasets = dict()
        if not dataset_id in self.datasets:
            ds_keys = ['{}_{}'.format(dataset_id, key) for key in
                         ('train_X', 'train_Y', 'test_X', 'test_Y')]
            if any(ds_key not in self.R for ds_key in ds_keys):
                raise ValueError('Dataset gauss has not been generated')
            train_X, train_Y, test_X, test_Y = [self.R[ds_key] for ds_key in ds_keys]
            train = DataSet(train_X, train_Y)
            test = DataSet(test_X, test_Y)
            self.datasets[dataset_id] = base.Datasets(train=train, test=test, validation=None)
        return self.datasets[dataset_id]

    def get_model(self, dataset_id=None):
        if not hasattr(self, 'model'):
            dataset = self.get_dataset(dataset_id)
            model_config = LogisticRegression.default_config()
            model_config['arch'] = LogisticRegression.infer_arch(dataset.train)
            model_dir = os.path.join(self.base_dir, 'models')
            self.model = LogisticRegression(model_config, model_dir)
        return self.model

    @phase(1)
    def training(self):
        res = dict()

        ds = self.get_dataset()
        model = self.get_model()

        res['l2_reg'] = l2_reg = ds.train.num_examples * 1e-3

        with benchmark("Training original model"):
            model.fit(ds.train, l2_reg=l2_reg)
            model.print_model_eval(ds, l2_reg=l2_reg)
            model.save('initial')

        res['train_losses'] = model.get_indiv_loss(ds.train)
        res['train_margins'] = model.get_indiv_margin(ds.train)
        res['train_accuracy'] = model.get_accuracy(ds.train)
        res['test_losses'] = model.get_indiv_loss(ds.test)
        res['test_margins'] = model.get_indiv_margin(ds.test)
        res['test_accuracy'] = model.get_accuracy(ds.test)

        with benchmark("Computing gradients"):
            res['train_grad_losses'] = model.get_indiv_grad_loss(ds.train)
            res['train_grad_margins'] = model.get_indiv_grad_margin(ds.train)
            res['test_grad_losses'] = model.get_indiv_grad_loss(ds.test)
            res['test_grad_margins'] = model.get_indiv_grad_margin(ds.test)

        res['hessian'] = model.get_hessian(ds.train, l2_reg=l2_reg)

        return res

    @phase(2)
    def find_newton_lt_pred(self):
        res = dict()

        ds = self.get_dataset()
        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['l2_reg']

        hessian = self.R['hessian']
        train_grad_losses = self.R['train_grad_losses']
        test_grad_losses = self.R['test_grad_losses']
        test_grad_margins = self.R['test_grad_margins']

        if self.dataset_id == "repeats":
            N_unique = self.R['repeats_N_unique']
            subsets = [
                list(range(i * (i + 1) // 2, i * (i + 1) // 2 + size))
                for i in range(N_unique)
                for size in range(1, (i + 1) + 1)
            ]
        else:
            size_min, size_max = 2, 3
            if self.dataset_id == "gauss2":
                size_max = 2
            subsets = list(list(subset)
                           for r in range(size_min, size_max + 1)
                           for subset in itertools.combinations(range(ds.train.num_examples), r))
        num_subsets = len(subsets)
        subset_grad_losses = np.array([np.sum(train_grad_losses[subset, :], axis=0) for subset in subsets])

        with benchmark('Computing first-order predicted parameters'):
            inverse_hvp_args = {
                'hessian_reg': hessian,
                'verbose': False,
                'inverse_hvp_method': 'explicit',
                'inverse_vp_method': 'cholesky',
            }
            pred_dparams = model.get_inverse_hvp(subset_grad_losses.T, **inverse_hvp_args).T

        with benchmark('Computing Newton predicted parameters'):
            newton_pred_dparams = np.zeros((num_subsets, model.params_dim))
            for i, subset in enumerate(subsets):
                hessian_w = model.get_hessian(ds.train.subset(subset), l2_reg=0, verbose=False)
                inverse_hvp_args = {
                    'hessian_reg': hessian - hessian_w,
                    'verbose': False,
                    'inverse_hvp_method': 'explicit',
                    'inverse_vp_method': 'cholesky',
                }
                subset_grad_loss = subset_grad_losses[i, :].reshape(-1, 1)
                pred_param = model.get_inverse_hvp(subset_grad_loss, **inverse_hvp_args).reshape(-1)
                newton_pred_dparams[i, :] = pred_param

        with benchmark('Computing actual parameters and influence'):
            model.load('initial')
            initial_params = model.get_params_flat()

            actl_dparams = np.zeros((num_subsets, model.params_dim))
            subset_train_losses = np.zeros((num_subsets, ds.train.num_examples))
            subset_train_margins = np.zeros((num_subsets, ds.train.num_examples))
            subset_test_losses = np.zeros((num_subsets, ds.test.num_examples))
            subset_test_margins = np.zeros((num_subsets, ds.test.num_examples))
            subset_actl_infl = np.zeros(num_subsets)
            subset_actl_margin_infl = np.zeros(num_subsets)
            for i, subset in enumerate(subsets):
                s = np.ones(ds.train.num_examples)
                s[subset] = 0

                model.warm_fit(ds.train, s, l2_reg=l2_reg)
                model.save('subset_{}'.format(i))
                subset_train_losses[i, :] = model.get_indiv_loss(ds.train, verbose=False)
                subset_train_margins[i, :] = model.get_indiv_margin(ds.train, verbose=False)
                subset_test_losses[i, :] = model.get_indiv_loss(ds.test, verbose=False)
                subset_test_margins[i, :] = model.get_indiv_margin(ds.test, verbose=False)
                actl_dparams[i, :] = model.get_params_flat() - initial_params

                subset_actl_infl = np.sum(subset_train_losses[i, subset]) - np.sum(self.R['train_losses'][subset])
                subset_actl_margin_infl = np.sum(subset_train_margins[i, subset]) - np.sum(self.R['train_margins'][subset])

        with benchmark('Find counterexamples'):
            overestimates = np.zeros((num_subsets, ds.test.num_examples)).astype(np.bool)
            subset_pred_infl = np.zeros((num_subsets, ds.test.num_examples))
            subset_pred_margin_infl = np.zeros((num_subsets, ds.test.num_examples))
            subset_newton_pred_infl = np.zeros((num_subsets, ds.test.num_examples))
            subset_newton_pred_margin_infl = np.zeros((num_subsets, ds.test.num_examples))
            cos = np.zeros(num_subsets)
            num_counterex, max_counterex = 0, 5
            for i, subset in enumerate(subsets):
                pred_dparam, newton_pred_dparam = pred_dparams[i, :], newton_pred_dparams[i, :]

                pred_infl = np.dot(test_grad_losses, pred_dparam)
                pred_margin_infl = np.dot(test_grad_margins, pred_dparam)
                newton_pred_infl = np.dot(test_grad_losses, newton_pred_dparam)
                newton_pred_margin_infl = np.dot(test_grad_margins, newton_pred_dparam)

                overestimates[i] = np.abs(pred_margin_infl) > np.sign(pred_margin_infl) * newton_pred_margin_infl + 1e-8

                # Print some counterexamples
                for j, overestimate in enumerate(overestimates[i]):
                    if not overestimate: continue
                    if num_counterex >= max_counterex:
                        break
                    print('subset: {}, test point: {}, first-order: {:.3}, newton: {:.3}'.format(
                          ' '.join(repr(x) for x in ds.train.x[subset, :]),
                          repr(ds.test.x[j, :]),
                          pred_infl[j],
                          newton_pred_infl[j]))
                    num_counterex += 1

                subset_pred_infl[i, :] = pred_infl
                subset_pred_margin_infl[i, :] = pred_margin_infl
                subset_newton_pred_infl[i, :] = newton_pred_infl
                subset_newton_pred_margin_infl[i, :] = newton_pred_margin_infl

            norm = np.linalg.norm(pred_dparams, axis=1) * np.linalg.norm(newton_pred_dparams)
            cos = np.sum(pred_dparams * newton_pred_dparams, axis=1) / (norm + 1e-3)

            subset_pred_over_newton = np.mean(overestimates, axis=1)
            print("Fraction of test point counterexamples per subset: min: {}, max: {}".format(
                np.min(subset_pred_over_newton), np.max(subset_pred_over_newton)))
            print("Cosine similarity between first-order and Newton predicted parameters: min: {}, max: {}".format(
                np.min(cos), np.max(cos)))
            print("{} of subsets have cos < 0.9".format(np.mean(cos < 0.99)))

        res['actl_dparams'] = actl_dparams
        res['pred_dparams'] = pred_dparams
        res['newton_pred_dparams'] = newton_pred_dparams
        res['overestimates'] = overestimates
        res['subset_cos_dparams'] = cos
        res['subset_indices'] = subsets
        res['subset_pred_infl'] = subset_pred_infl
        res['subset_pred_margin_infl'] = subset_pred_margin_infl
        res['subset_newton_pred_infl'] = subset_newton_pred_infl
        res['subset_newton_pred_margin_infl'] = subset_newton_pred_margin_infl
        res['subset_actl_infl'] = subset_actl_infl
        res['subset_actl_margin_infl'] = subset_actl_margin_infl

        return res

    @phase(3)
    def find_test_distribution(self):
        res = dict()
        ds = self.get_dataset()
        model = self.get_model()
        model.load('initial')

        # For K subsets, find a distribution of test points such that
        # (pred, newton_pred) ~ gaussian
        D = self.R['pred_dparams'].shape[1]
        K = 10 # not too-overconstrained system
        S = 10

        rng = np.random.RandomState(self.config['seed'])
        N_test = 1000
        cos_indices = np.argsort(self.R['subset_cos_dparams'])
        easiest = cos_indices[:S]
        X = []
        A = np.vstack([self.R['pred_dparams'][easiest, :],
                       self.R['newton_pred_dparams'][easiest, :]])
        for u, v in rng.normal(0, 1, (N_test, 2)):
            B = np.hstack([np.full(S, u), np.full(S, v)])
            x = np.linalg.lstsq(A, B, rcond=None)[0]
            X.append(x)
        X = np.array(X)
        Y = np.ones(X.shape[0])
        dist_ds = DataSet(X, Y)

        test_grad_loss = model.get_indiv_grad_loss(dist_ds)
        test_grad_margin = model.get_indiv_grad_margin(dist_ds)

        with benchmark('Computing first-order influence'):
            subset_pred_infl = np.dot(self.R['pred_dparams'][easiest, :], test_grad_loss.T)
            subset_pred_margin_infl = np.dot(self.R['pred_dparams'][easiest, :], test_grad_margin.T)

        with benchmark('Computing Newton influence'):
            subset_newton_pred_infl = np.dot(self.R['newton_pred_dparams'][easiest, :], test_grad_loss.T)
            subset_newton_pred_margin_infl = np.dot(self.R['newton_pred_dparams'][easiest, :], test_grad_margin.T)

        with benchmark('Computing actual influence'):
            num_subsets = len(self.R['subset_indices'])
            subset_actl_infl = []
            subset_actl_margin_infl = []

            model.load('initial')
            initial_loss = model.get_indiv_loss(dist_ds, verbose=False)
            initial_margin = model.get_indiv_margin(dist_ds, verbose=False)

            for i, subset in enumerate(self.R['subset_indices']):
                if i not in easiest: continue
                model.load('subset_{}'.format(i))
                subset_loss = model.get_indiv_loss(dist_ds, verbose=False)
                subset_margin = model.get_indiv_margin(dist_ds, verbose=False)

                subset_actl_infl.append(subset_loss - initial_loss)
                subset_actl_margin_infl.append(subset_margin - initial_margin)

        res['dist_subset_pred_infl'] = subset_pred_infl
        res['dist_subset_pred_margin_infl'] = subset_pred_margin_infl
        res['dist_subset_newton_pred_infl'] = subset_newton_pred_infl
        res['dist_subset_newton_pred_margin_infl'] = subset_newton_pred_margin_infl
        res['dist_subset_actl_infl'] = np.array(subset_actl_infl)
        res['dist_subset_actl_margin_infl'] = np.array(subset_actl_margin_infl)
        return res

    def plot_overestimates(self, save_and_close=False):
        ds = self.get_dataset()
        pred = self.R['subset_pred_margin_infl'].reshape(-1)
        newton = self.R['subset_newton_pred_margin_infl'].reshape(-1)
        actl = self.R['subset_actl_margin_infl'].reshape(-1)
        overestimates = self.R['overestimates'].reshape(-1)
        tags = np.array(['first-order < sign(first-order) * newton'] * len(pred))
        tags[overestimates] = 'first-order > sign(first-order) * newton'
        if self.dataset_id != "repeats":
            subset_sizes = np.repeat([len(subset) for subset in self.R['subset_indices']], ds.test.num_examples).reshape(-1)
            tags = ['{} (size {})'.format(tag, size) for tag, size in zip(tags, subset_sizes)]

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        plot_influence_correlation(ax, pred, newton,
                                   label=tags,
                                   xlabel='First-order influence',
                                   ylabel='Newton influence',
                                   title='Influence on margin, for all combinations of test points and subsets',
                                   subtitle=self.dataset_id,
                                   size=1)
        if save_and_close:
            fig.savefig(os.path.join(self.plot_dir, 'pred_over_newton.png'), bbox_inches='tight')
            plt.close(fig)

    def plot_repeats(self, save_and_close=False):
        if self.dataset_id != "repeats": return
        ds = self.get_dataset()
        pred = self.R['subset_pred_margin_infl'].reshape(-1)
        newton = self.R['subset_newton_pred_margin_infl'].reshape(-1)

        sizes = np.array([len(subset) for subset in self.R['subset_indices']])
        norm = mpl.colors.Normalize(vmin=1, vmax=np.max(sizes))
        cmap = plt.get_cmap('plasma')
        color_by_size = np.repeat(cmap(norm(sizes)), ds.test.num_examples, axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        plot_influence_correlation(ax, pred, newton,
                                   colors=color_by_size,
                                   xlabel='First-order influence',
                                   ylabel='Newton influence',
                                   title='Influence on margin, for all combinations of test points and subsets',
                                   subtitle=self.dataset_id,
                                   size=1,
                                   balanced=True,
                                   equal=False)
        ax.set_xlim([x * 0.5 for x in ax.get_xlim()])
        ax.set_ylim([x * 0.5 for x in ax.get_ylim()])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.ax.set_ylabel('number of repeats removed', rotation=90)

        if save_and_close:
            fig.savefig(os.path.join(self.plot_dir, 'pred_over_newton_size.png'), bbox_inches='tight')
            plt.close(fig)

        N_unique = self.R['repeats_N_unique']
        repeat_ids = np.repeat(np.array([i for i in range(N_unique) for _ in range(i + 1)]), ds.test.num_examples)
        norm = mpl.colors.Normalize(vmin=0, vmax=np.max(repeat_ids))
        cmap = plt.get_cmap('rainbow', N_unique)
        color_by_id = cmap(repeat_ids)
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        plot_influence_correlation(ax, pred, newton,
                                   colors=color_by_id,
                                   xlabel='First-order influence',
                                   ylabel='Newton influence',
                                   title='Influence on margin, for all combinations of test points and subsets',
                                   subtitle=self.dataset_id,
                                   size=3,
                                   balanced=True,
                                   equal=False)
        ax.set_xlim([x * 0.5 for x in ax.get_xlim()])
        ax.set_ylim([x * 0.5 for x in ax.get_ylim()])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.ax.set_ylabel('repeated point id', rotation=90)

        if save_and_close:
            fig.savefig(os.path.join(self.plot_dir, 'pred_over_newton_id.png'), bbox_inches='tight')
            plt.close(fig)

    def plot_counterex_distribution(self, save_and_close=False):
        overestimates = np.mean(self.R['overestimates'], axis=1)
        std = np.std(overestimates)
        if std < 1e-8: return
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_distribution(ax, overestimates,
                          title='Distribution of counterexamples',
                          xlabel='Fraction of test points with first-order > sign(first-order) * newton',
                          ylabel='Number of subsets',
                          subtitle=self.dataset_id)
        if save_and_close:
            fig.savefig(os.path.join(self.plot_dir, 'pred_over_newton_dist.png'), bbox_inches='tight')
            plt.close(fig)

    def plot_dist_infl(self, save_and_close=False):
        K = self.R['dist_subset_pred_margin_infl'].shape[0]

        pred_margin = self.R['dist_subset_pred_margin_infl']
        newton_margin = self.R['dist_subset_newton_pred_margin_infl']
        actl_margin = self.R['dist_subset_actl_margin_infl']

        pred = self.R['dist_subset_pred_infl']
        newton = self.R['dist_subset_newton_pred_infl']
        actl = self.R['dist_subset_actl_infl']

        def compare_influences(x, y, x_approx_type, y_approx_type, infl_type):
            approx_type_to_label = { 'pred': 'First-order influence',
                                     'newton': 'Newton influence',
                                     'actl': 'Actual influence' }
            xlabel = approx_type_to_label[x_approx_type]
            ylabel = approx_type_to_label[y_approx_type]
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            plot_influence_correlation(ax, x.reshape(-1), y.reshape(-1),
                                       xlabel=xlabel,
                                       ylabel=ylabel,
                                       title='Influence on {}, for {} subsets and a constructed test set'.format(infl_type, K),
                                       subtitle=self.dataset_id,
                                       size=3,
                                       equal=False)
            if save_and_close:
                fig.savefig(os.path.join(self.plot_dir, 'dist_{}-{}_{}.png'.format(
                    x_approx_type, y_approx_type, "infl" if infl_type == "loss" else "margin_infl")), bbox_inches='tight')
                plt.close(fig)

        compare_influences(pred_margin, newton_margin, 'pred', 'newton', 'margin')
        compare_influences(actl_margin, pred_margin, 'actl', 'pred', 'margin')
        compare_influences(pred, newton, 'pred', 'newton', 'loss')
        compare_influences(actl, pred, 'actl', 'pred', 'loss')


    def plot_all(self, save_and_close=False):
        self.plot_overestimates(save_and_close)
        self.plot_repeats(save_and_close)
        self.plot_counterex_distribution(save_and_close)
        self.plot_dist_infl(save_and_close)
