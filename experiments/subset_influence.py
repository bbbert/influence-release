from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from influence.logistic_regression import LogisticRegression
import datasets as ds
import datasets.loader
import datasets.mnist
from datasets.common import DataSet
from experiments.common import Experiment, collect_phases, phase
from experiments.benchmark import benchmark

import os
import time
import math
import numpy as np

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
        self.model_dir = model_dir
        self.model_config = model_config

        self.num_train = self.datasets.train.num_examples
        self.num_classes = self.model_config['arch']['num_classes']
        self.num_subsets = self.config['num_subsets']
        self.subset_size = int(self.num_train * self.config['subset_rel_size'])

        MAX_MEMORY = int(1e7)
        D = model_config['arch']['input_dim'] * self.num_classes
        self.eval_args = {
            'grad_batch_size': max(1, MAX_MEMORY // D),
            'hess_batch_size': max(1, MAX_MEMORY // (D * D)),
        }

    experiment_id = "ss_logreg"

    @property
    def run_id(self):
        return "{}_seed-{}_size-{}_num-{}".format(
            self.config['dataset_config']['dataset_id'],
            self.config['subset_seed'],
            self.config['subset_rel_size'],
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
                    fold_loss = model.get_total_loss(self.train.subset(val_indices), reg=True, l2_reg=reg)
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
            model.print_model_eval(self.datasets)
            model.save('initial')

        res['initial_train_losses'] = model.get_indiv_loss(self.train)
        res['initial_test_losses'] = model.get_indiv_loss(self.test)
        if self.num_classes == 2:
            res['initial_train_margins'] = model.get_indiv_margin(self.train)
            res['initial_test_margins'] = model.get_indiv_margin(self.test)

        with benchmark("Computing gradients"):
            res['train_grad_loss'] = model.get_indiv_grad_loss(self.train, **self.eval_args)

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
            test_losses = self.R['test_losses']
            argsort = np.argsort(test_losses)
            high_loss = argsort[-3:] # Pick 3 high loss points
            random_loss = np.random.choice(argsort[:-3], 3, replace=False) # Pick 3 random points

            fixed_test = list(high_loss) + list(random_loss)

        print("Fixed test points: {}".format(fixed_test))
        return { 'fixed_test': fixed_test }

    @phase(3)
    def fixed_test_influence(self):
        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        with benchmark("Computing hessian"):
            res['hessian'] = hessian = model.get_hessian_reg(self.train, l2_reg=l2_reg)

        fixed_test = self.R['fixed_test']
        fixed_test_pred_infl = []
        fixed_test_pred_margin_infl = []
        for test_idx in fixed_test:
            single_test_point = self.test.subset([test_idx])

            with benchmark('Scalar infl for all training points on test_idx {}.'.format(test_idx)):
                test_grad_loss = model.get_indiv_grad_loss(single_test_point).reshape(-1, 1)
                test_grad_loss_H_inv = model.get_inverse_vp(hessian, test_grad_loss)
                pred_infl = np.dot(self.R['train_grad_loss'], test_grad_loss_H_inv)
                fixed_test_pred_infl.append(pred_infl)

            if self.num_classes == 2:
                with benchmark('Scalar margin infl for all training points on test_idx {}.'.format(test_idx)):
                    test_grad_margin = model.get_total_grad_margin(single_test_point).reshape(-1, 1)
                    test_grad_margin_H_inv = model.get_inverse_vp(hessian, test_grad_margin)
                    pred_margin_infl = np.dot(self.R['train_grad_loss'], test_grad_margin_H_inv)
                    fixed_test_pred_margin_infl.append(pred_margin_infl)

        res['fixed_test_pred_infl'] = np.array(fixed_test_pred_infl)
        if self.num_classes == 2:
            res['fixed_test_pred_margin_infl'] = np.array(fixed_test_pred_margin_infl)

        return res

    def get_random_subsets(self, rng):
        subsets = []
        for i in range(self.num_subsets):
            subsets.append(rng.choice(self.num_train, self.subset_size, replace=False))
        return np.array(subsets)

    def get_scalar_infl_tails(self, rng, pred_infl):
        window = 2 * self.subset_size
        assert window < self.num_train
        scalar_infl_indices = np.argsort(pred_infl).reshape(-1)
        pos_subsets, neg_subsets = [], []
        for i in range(self.num_subsets):
            neg_subsets.append(rng.choice(scalar_infl_indices[:window], self.subset_size, replace=False))
            pos_subsets.append(rng.choice(scalar_infl_indices[-window:], self.subset_size, replace=False))
        return np.array(neg_subsets), np.array(pos_subsets)

    def get_same_grad_dir(self, rng, train_grad_loss):
        # Using Pigeonhole to guarantee we get a sufficiently large cluster
        n_clusters = int(math.floor(1 / self.config['subset_rel_size']))
        km = KMeans(n_clusters=n_clusters)
        km.fit(train_grad_loss)
        labels, centroids = km.labels_, km.cluster_centers_
        _, counts = np.unique(labels, return_counts=True)

        best = max([(count, i) for i, count in enumerate(counts) if count >= self.subset_size])[1]
        cluster_indices = np.where(labels == best)[0]
        subsets = []
        for i in range(self.num_subsets):
            subsets.append(rng.choice(cluster_indices, self.subset_size, replace=False))
        return np.array(subsets), best, labels

    def get_same_class_subsets(self, rng, labels, test_label=None):
        label_vals, counts = np.unique(labels, return_counts=True)
        valid_labels = []
        valid_indices = []
        for i in range(len(label_vals)):
            if counts[i] >= self.subset_size:
                valid_labels.append(label_vals[i])
                valid_indices.append(list(np.where(labels == label_vals[i])[0]))
        assert len(valid_indices) > 0
        flat = [i for sublist in valid_indices for i in sublist]
        label_to_ind = dict(zip(valid_labels, range(len(valid_indices))))

        subsets = []
        if test_label is not None and int(test_label) not in valid_labels:
            print('Couldn\'t use desired label.')
        for i in range(self.num_subsets):
            if (test_label is None) or (int(test_label) not in valid_labels):
                sample = rng.choice(flat)
                sample_ind = label_to_ind[labels[sample]]
            else:
                sample_ind = label_to_ind[int(test_label)]
            subsets.append(rng.choice(valid_indices[sample_ind], self.subset_size, replace=False))
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

    @phase(4)
    def pick_subsets(self):
        rng = np.random.RandomState(self.config['subset_seed'])
        tagged_subsets = []

        with benchmark("Random subsets"):
            random_subsets = self.get_random_subsets(rng)
            tagged_subsets += [('random', s) for s in random_subsets]

        with benchmark("Same class subsets"):
            same_class_subsets = self.get_same_class_subsets(rng, self.train.labels, test_label=None)
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
            same_grad_subsets, cluster_label, cluster_labels = self.get_same_grad_dir(rng, self.R['train_grad_loss'])
            tagged_subsets += [('same_grad', s) for s in same_grad_subsets]

        subset_tags = [tag for tag, subset in tagged_subsets]
        subset_indices = [subset for tag, subset in tagged_subsets]

        return { 'subset_tags': subset_tags, 'subset_indices': subset_indices }

    @phase(5)
    def retrain(self):
        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']
        n, n_report = len(subset_indices), max(len(subset_indices) // 100, 1)

        hessian = self.R['hessian']
        train_grad_loss = self.R['train_grad_loss']

        # It is important that the influence gets calculated before the model is retrained,
        # so that the parameters are the original parameters
        start_time = time.time()
        self_pred_infls = []
        self_pred_margin_infls = []
        for i, remove_indices in enumerate(subset_indices):
            if (i % n_report == 0):
                print('Computing self-influences for subset {} out of {} (tag={})'.format(i, n, subset_tags[i]))

            grad_loss = np.sum(train_grad_loss[remove_indices, :], axis=0)
            H_inv_grad_loss = model.get_inverse_vp(hessian, grad_loss.reshape(1, -1).T)
            pred_infl = np.dot(grad_loss, H_inv_grad_loss)
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

        res['subset_self_pred_infl'] = np.array(self_pred_infls)
        res['subset_train_losses'] = np.array(train_losses)
        res['subset_test_losses'] = np.array(test_losses)

        if self.num_classes == 2:
            res['subset_self_pred_margin_infl'] = np.array(self_pred_margin_infls)
            res['subset_train_margins'] = np.array(train_margins)
            res['subset_test_margins'] = np.array(test_margins)

        return res

    @phase(6)
    def compute_infl(self):
        res = dict()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']

        # Compute fixed test influences
        fixed_test = self.R['fixed_test']
        initial_fixed_test_loss = self.R['initial_test_losses'][fixed_test]
        subset_fixed_test_loss = self.R['subset_test_losses'][:, fixed_test]
        subset_fixed_test_actl_infl = subset_fixed_test_loss - initial_fixed_test_loss
        subset_fixed_test_pred_infl = np.zeros_like(subset_fixed_test_actl_infl)
        for i, remove_indices in enumerate(subset_indices):
            subset_fixed_test_pred_infl[i, :] = \
                np.sum(self.R['fixed_test_pred_infl'][:, remove_indices], axis=1).reshape(-1)

        res['subset_fixed_test_actl_infl'] = subset_fixed_test_actl_infl
        res['subset_fixed_test_pred_infl'] = subset_fixed_test_pred_infl

        # Compute subset self influences
        subset_self_actl_infl = np.zeros(len(subset_indices))
        for i, remove_indices in enumerate(subset_indices):
            subset_initial_loss = np.sum(self.R['initial_train_losses'][remove_indices])
            subset_retrained_loss = np.sum(self.R['subset_train_losses'][i][remove_indices])
            subset_self_actl_infl[i] = subset_retrained_loss - subset_initial_loss

        res['subset_self_actl_infl'] = subset_self_actl_infl

        if self.num_classes == 2:
            initial_fixed_test_margin = self.R['initial_test_margins'][fixed_test]
            subset_fixed_test_margin = self.R['subset_test_margins'][:, fixed_test]
            subset_fixed_test_actl_margin_infl = subset_fixed_test_margin - initial_fixed_test_margin
            subset_fixed_test_pred_margin_infl = np.zeros_like(subset_fixed_test_actl_margin_infl)
            for i, remove_indices in enumerate(subset_indices):
                subset_fixed_test_pred_margin_infl[i, :] = \
                    np.sum(self.R['fixed_test_pred_margin_infl'][:, remove_indices], axis=1).reshape(-1)

            res['subset_fixed_test_actl_margin_infl'] = subset_fixed_test_actl_margin_infl
            res['subset_fixed_test_pred_margin_infl'] = subset_fixed_test_pred_margin_infl

            subset_self_actl_margin_infl = np.zeros(len(subset_indices))
            for i, remove_indices in enumerate(subset_indices):
                subset_initial_margin = np.sum(self.R['initial_train_margins'][remove_indices])
                subset_retrained_margin = np.sum(self.R['subset_train_margins'][i][remove_indices])
                subset_self_actl_margin_infl[i] = subset_retrained_margin - subset_initial_margin

            res['subset_self_actl_margin_infl'] = subset_self_actl_margin_infl

        return res
