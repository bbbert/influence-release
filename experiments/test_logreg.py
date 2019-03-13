from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from influence.logistic_regression import LogisticRegression
import datasets as ds
import datasets.loader
import datasets.mnist
from experiments.common import Experiment, collect_phases, phase

import os
import time
import numpy as np

@collect_phases
class TestLogreg(Experiment):
    """
    Example experiment class demonstrating how to write experiment phases.
    """
    def __init__(self, config, out_dir=None):
        super(TestLogreg, self).__init__(config, out_dir)
        self.datasets = ds.loader.load_dataset(**self.config['dataset_config'])

        model_dir = os.path.join(self.base_dir, 'models')
        model_config = LogisticRegression.default_config()
        model_config['arch'] = LogisticRegression.infer_arch(self.datasets.train)
        model_config['arch']['fit_intercept'] = False
        self.model_dir = model_dir
        self.model_config = model_config

    experiment_id = "test_logreg"

    @property
    def run_id(self):
        return "run"

    def get_model(self):
        if not hasattr(self, 'model'):
            self.model = LogisticRegression(self.model_config, self.model_dir)
        return self.model

    @phase(0)
    def train_model(self):
        results = dict()

        model = self.get_model()
        model.fit(self.datasets.train)

        results['train_loss'] = model.get_total_loss(self.datasets.train, reg=True)
        results['indiv_train_loss'] = model.get_indiv_loss(self.datasets.train)
        results['test_loss'] = model.get_total_loss(self.datasets.test, reg=True)
        results['indiv_test_loss'] = model.get_indiv_loss(self.datasets.test)

        model.save('initial')
        model.print_model_eval(self.datasets)

        return results

    @phase(1)
    def retrain_model(self):
        model = self.get_model()
        model.load('initial')

        print("Sanity check: reloading the model gives same train and test losses")
        indiv_train_loss = model.get_indiv_loss(self.datasets.train)
        indiv_test_loss = model.get_indiv_loss(self.datasets.test)
        print("train loss l2 diff: {}, test loss l2 diff: {}".format(
            np.linalg.norm(indiv_train_loss - self.results['train_model']['indiv_train_loss']),
            np.linalg.norm(indiv_test_loss - self.results['train_model']['indiv_test_loss'])))

        print("Sanity check: warm fit is fast")
        st = time.time()
        model.warm_fit(self.datasets.train)
        en = time.time()
        print("That took {} seconds".format(en - st))

        print("Sanity check, we're going to force-recreate the model and load it.")
        del self.model
        model = self.get_model()
        model.load('initial')

        print("Sanity check: it should still be the same")
        indiv_train_loss = model.get_indiv_loss(self.datasets.train)
        indiv_test_loss = model.get_indiv_loss(self.datasets.test)
        print("train loss l2 diff: {}, test loss l2 diff: {}".format(
            np.linalg.norm(indiv_train_loss - self.results['train_model']['indiv_train_loss']),
            np.linalg.norm(indiv_test_loss - self.results['train_model']['indiv_test_loss'])))

        model.print_model_eval(self.datasets)

        return {}

    @phase(2)
    def compute_grad_loss(self):
        model = self.get_model()
        model.load('initial')

        # compute some gradients
        indiv_grad_loss = model.get_indiv_grad_loss(self.datasets.train)

        return { 'indiv_grad_loss': indiv_grad_loss }

    @phase(3)
    def hvp(self):
        model = self.get_model()
        model.load('initial')

        result = dict()
        result['hessian_reg'] = model.get_hessian_reg(self.datasets.train)
        result['eigs'] = eigs = np.linalg.eigvalsh(result['hessian_reg'])
        print("Hessian eigenvalue range:", np.min(eigs), np.max(eigs))

        indiv_grad_loss = self.results['compute_grad_loss']['indiv_grad_loss']

        some_indices = [1, 6, 2, 4, 3]
        vectors = indiv_grad_loss[some_indices, :].T
        result['inverse_hvp'] = model.get_inverse_hvp_reg(self.datasets.train, vectors)

        return result
