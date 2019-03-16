from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import math
import numpy as np
import tensorflow as tf
import sklearn
import sklearn.linear_model

from influence.model import Model, variable_with_l2_reg, flatten
from influence.model import get_assigners, get_accuracy

class LogisticRegression(Model):
    def __init__(self, config, model_dir=None):
        super(LogisticRegression, self).__init__(config, model_dir)

    def build_graph(self):
        # Setup architecture
        self.input_dim = self.config['arch']['input_dim']
        self.fit_intercept = self.config['arch']['fit_intercept']
        self.num_classes = self.config['arch']['num_classes']

        if self.num_classes > 2:
            self.multi_class = "multinomial"
            self.pseudo_num_classes = self.num_classes
        else:
            self.multi_class = "ovr"
            self.pseudo_num_classes = 1

        # Setup input
        self.input_placeholder = tf.placeholder(
            tf.float32,
            shape=(None, self.input_dim),
            name='input_placeholder')
        self.labels_placeholder = tf.placeholder(
            tf.int32,
            shape=(None,),
            name='labels_placeholder')
        self.sample_weights_placeholder = tf.placeholder(
            tf.float32,
            shape=(None,),
            name='sample_weights_placeholder')
        self.l2_reg = tf.Variable(self.config['default_l2_reg'],
                                  dtype=tf.float32,
                                  trainable=False,
                                  name='l2_reg')
        self.l2_reg_assigner = get_assigners([self.l2_reg])[0]

        # Setup inference and losses
        self.logits, self.params = self.infer(self.input_placeholder, self.labels_placeholder)
        self.params_assigners = get_assigners(self.params)
        self.params_flat = flatten(self.params)
        self.one_hot_labels = tf.one_hot(self.labels_placeholder, depth=self.num_classes)
        self.total_loss_reg, self.total_loss_no_reg, self.indiv_loss = self.loss(
            self.logits,
            self.one_hot_labels,
            self.sample_weights_placeholder)
        self.predictions = self.predict(self.logits)
        self.accuracy = get_accuracy(self.logits, self.labels_placeholder)

        # Setup margins, but only for binary logistic regression
        if self.num_classes == 2:
            y = tf.cast(self.labels_placeholder, tf.float32) * 2 - 1
            self.margins = tf.multiply(y, self.logits[:, 1])
            margin_input = self.input_placeholder
            if self.fit_intercept:
                margin_input = tf.pad(margin_input, [[0, 0], [0, 1]],
                                      mode="CONSTANT", constant_values=1.0)
            self.indiv_grad_margin = tf.multiply(
                margin_input,
                tf.expand_dims(y * self.sample_weights_placeholder, 1))
            self.total_grad_margin = tf.reduce_sum(self.indiv_grad_margin, axis=0)

        # Calculate gradients explicitly
        self.gradients(self.input_placeholder,
                       self.logits,
                       self.one_hot_labels,
                       self.sample_weights_placeholder)

        # Calculate gradients
        # self.total_grad_loss_reg = tf.gradients(self.total_loss_reg, self.params)
        # self.total_grad_loss_no_reg = tf.gradients(self.total_loss_no_reg, self.params)
        # self.total_grad_loss_reg_flat = flatten(self.total_grad_loss_reg)
        # self.total_grad_loss_no_reg_flat = flatten(self.total_grad_loss_no_reg)

        # Calculate gradients explicitly
        self.hessian(self.input_placeholder,
                     self.logits,
                     self.sample_weights_placeholder)

        # This only works for a single parameter. To fix, concatenate
        # all parameters into a flat tensor, then split them up again to obtain
        # phantom parameters and use those in the model.
        # Calculate Hessians
        # if not self.fit_intercept:
            # self.hessian_reg = tf.hessians(self.total_loss_reg, self.params)[0]

        self.matrix_placeholder = tf.placeholder(
            tf.float32,
            shape=(self.params_flat.shape[0], self.params_flat.shape[0]),
            name='matrix_placeholder')
        self.vectors_placeholder = tf.placeholder(
            tf.float32,
            shape=(None, self.params_flat.shape[0]),
            name='vectors_placeholder')
        self.inverse_vp = tf.cholesky_solve(tf.cholesky(self.matrix_placeholder),
                                            tf.transpose(self.vectors_placeholder))

    def infer(self, input, labels):
        params = []
        with tf.variable_scope('softmax_linear'):
            weights = variable_with_l2_reg(
                name='weights',
                shape=(self.input_dim * self.pseudo_num_classes,),
                stddev=1.0 / math.sqrt(float(self.input_dim)),
                l2_reg=self.l2_reg)
            self.weights = weights
            params.append(weights)

            if self.fit_intercept:
                biases = variable_with_l2_reg(
                    'biases',
                    (self.pseudo_num_classes,),
                    stddev=None,
                    l2_reg=None)
                self.biases = biases
                params.append(biases)

                logits = tf.matmul(input, tf.reshape(weights, (self.input_dim, self.pseudo_num_classes))) + biases
            else:
                logits = tf.matmul(input, tf.reshape(weights, (self.input_dim, self.pseudo_num_classes)))

        if self.num_classes == 2:
            zeros = tf.zeros_like(logits)
            logits = tf.concat([zeros, logits], 1)

        return logits, params

    def loss(self, logits, one_hot_labels, sample_weights):
        log_softmax = tf.nn.log_softmax(logits)
        cross_entropy = -tf.reduce_sum(tf.multiply(one_hot_labels, log_softmax), axis=1)

        indiv_loss = cross_entropy
        total_loss_no_reg = tf.reduce_sum(tf.multiply(cross_entropy, sample_weights),
                                          name='total_loss_no_reg')
        tf.add_to_collection('losses', total_loss_no_reg)

        total_loss_reg = tf.add_n(tf.get_collection('losses'), name='total_loss_reg')

        return total_loss_reg, total_loss_no_reg, indiv_loss

    def gradients(self, inputs, logits, one_hot_labels, sample_weights):
        """
        Explicitly computes the softmax gradients.

        grad_theta_i loss(x, y) = -([i == y] - softmax_i) * x
        grad_b_i loss(x, y) = -([i == y] - softmax_i)
        """
        K, Kp, D = self.num_classes, self.pseudo_num_classes, self.input_dim
        KpD = Kp * D

        # Gradient of loss
        softmax = tf.nn.softmax(logits)
        factor = -(one_hot_labels - softmax)           # (?, K)
        if self.num_classes == 2:
            # Pick only weights for the second class
            factor = factor[:, 1:2]                    # (?, Kp)
        expand_factor = tf.expand_dims(factor, axis=2) # (?, Kp, 1)
        expand_inputs = tf.expand_dims(inputs, 1)      # (?, 1, D)
        indiv_grad_loss = tf.reshape(tf.multiply(expand_factor, expand_inputs),
                                     (-1, KpD))

        # Gradient of l2 regularization
        grad_reg = self.l2_reg * tf.ones(KpD)

        # Handle bias term
        if self.fit_intercept:
            indiv_grad_loss = tf.concat([indiv_grad_loss, factor], axis=1) # (?, KpD + Kp)
            grad_reg = tf.pad(grad_reg, [[0, Kp]],
                              mode="CONSTANT", constant_values=0.0)

        # Compute grad losses
        self.indiv_grad_loss = indiv_grad_loss
        weighted_grad_loss = tf.multiply(indiv_grad_loss,
                                         tf.expand_dims(sample_weights, 1))
        self.total_grad_loss_no_reg_flat = tf.reduce_sum(weighted_grad_loss, axis=0)
        self.total_grad_loss_reg_flat = self.total_grad_loss_no_reg_flat + grad_reg

        # Separate weights and biases
        self.total_grad_loss_no_reg = [self.total_grad_loss_no_reg_flat[:KpD],
                                       self.total_grad_loss_no_reg_flat[KpD:]]
        self.total_grad_loss_reg = [self.total_grad_loss_reg_flat[:KpD],
                                    self.total_grad_loss_reg_flat[KpD:]]

    def hessian(self, inputs, logits, sample_weights):
        """
        Explicitly computes the softmax hessian.

        grad_theta_i grad_theta_j loss(x, y)
            = softmax_i ([i == j] - softmax_j) x x^T
        grad_theta_i grad_b_j loss(x, y)
            = softmax_i ([i == j] - softmax_j) x
        grad_b_i grad_b_j loss(x, y)
            = softmax_i ([i == j] - softmax_j)
        """
        K, Kp, D = self.num_classes, self.pseudo_num_classes, self.input_dim
        KpD = Kp * D

        softmax = tf.nn.softmax(logits)                             # (?, K)
        if self.num_classes == 2:
            softmax = softmax[:, 1:2]                               # (?, Kp)
        factor = tf.linalg.diag(softmax) - \
            tf.einsum('ai,aj->aij', softmax, softmax)               # (?, Kp, Kp)
        indiv_hessian = tf.reshape(
            tf.einsum('aij,ak,al->aikjl', factor, inputs, inputs),  # (?, Kp, D, Kp, D)
            (-1, KpD, KpD))                                         # (?, KpD, KpD)

        # Hessian of l2 regularization
        hess_reg = self.l2_reg * tf.eye(KpD, KpD)

        if self.fit_intercept:
            off_diag = tf.reshape(
                tf.einsum('aij,ak->aijk', factor, inputs),          # (?, Kp, Kp, D)
                (-1, Kp, KpD))                                      # (?, Kp, KpD)

            top_row = tf.concat([indiv_hessian,
                                 tf.transpose(off_diag, (0, 2, 1))], axis=2)
            bottom_row = tf.concat([off_diag, factor], axis=2)
            indiv_hessian = tf.concat([top_row, bottom_row], axis=1)

            hess_reg = tf.pad(hess_reg, [[0, Kp], [0, Kp]],
                              mode="CONSTANT", constant_values=0.0)

        self.hessian_no_reg = tf.einsum('aij,a->ij', indiv_hessian, sample_weights)
        self.hessian_reg = self.hessian_no_reg + hess_reg
        self.hessian_of_reg = hess_reg

    def predict(self, logits):
        predictions = tf.nn.softmax(logits, name='predictions')
        return predictions

    # Saving and restoring parameters

    def get_params_flat(self):
        return self.sess.run(self.params_flat)

    def get_params(self):
        return self.sess.run(self.params)

    def set_params_flat(self, params_flat):
        params = self.unflatten_params(params_flat.reshape(-1))
        self.set_params(params)

    def set_params(self, params):
        for param, assigner in zip(params, self.params_assigners):
            assign_op, placeholder = assigner
            self.sess.run(assign_op, feed_dict={placeholder: param})

    def unflatten_params(self, params_flat):
        assert params_flat.shape == self.params_flat.shape
        index, params = 0, []
        for orig_param in self.params:
            param = params_flat[index:index + tf.size(orig_param)].reshape(orig_param.shape)
            index += tf.size(orig_param)
            params.append(param)
        return params

    # Training

    def set_l2_reg(self, l2_reg):
        assign_op, placeholder = self.l2_reg_assigner
        self.sess.run(assign_op, feed_dict={placeholder: l2_reg})

    def fit(self, dataset, sample_weights=None, **kwargs):
        """
        Resets the model's parameters and trains the model to fit the dataset.
        Minimizes the objective:
            sum_i l(z_i, theta) + l2_reg / 2 * l2_norm(weights)^2
        Which is equivalent to the sklearn objective:
            C sum_i l(z_i, theta) + 1 / 2 * l2_norm(weights)^2
        with C = 1 / l2_reg.

        :param dataset: The dataset to fit the model to.
        :param sample_weights: The weight of each example in the dataset.
        """
        if sample_weights is None:
            sample_weights = np.ones(dataset.num_examples)

        l2_reg = kwargs.get('l2_reg', self.config['default_l2_reg'])
        C = 1.0 / l2_reg
        sklearn_model = sklearn.linear_model.LogisticRegression(
            C=C,
            tol=1e-8,
            fit_intercept=self.fit_intercept,
            solver='lbfgs',
            multi_class=self.multi_class,
            warm_start=False,
            max_iter=self.config['max_lbfgs_iter'])

        sklearn_model.fit(dataset.x,
                          dataset.labels,
                          sample_weight=sample_weights)
        self.copy_sklearn_model_to_params(sklearn_model)

    def warm_fit(self, dataset, sample_weights=None, **kwargs):
        """
        Trains the model to fit the dataset, using the previously stored
        parameters as a starting point.

        :param dataset: The dataset to fit the model to.
        :param sample_weights: The weight of each example in the dataset.
        """
        if sample_weights is None:
            sample_weights = np.ones(dataset.num_examples)

        l2_reg = kwargs.get('l2_reg', self.config['default_l2_reg'])
        C = 1.0 / l2_reg
        sklearn_model = sklearn.linear_model.LogisticRegression(
            C=C,
            tol=1e-8,
            fit_intercept=self.fit_intercept,
            solver='lbfgs',
            multi_class=self.multi_class,
            warm_start=True,
            max_iter=self.config['max_lbfgs_iter'])

        self.copy_params_to_sklearn_model(sklearn_model)
        sklearn_model.fit(dataset.x,
                          dataset.labels,
                          sample_weight=sample_weights)
        self.copy_sklearn_model_to_params(sklearn_model)

    def copy_params_to_sklearn_model(self, sklearn_model):
        params = self.get_params()
        W = params[0].reshape((self.input_dim, self.pseudo_num_classes)).T
        sklearn_model.coef_ = W

        if self.fit_intercept:
            b = params[1]
            sklearn_model.intercept_ = b

    def copy_sklearn_model_to_params(self, sklearn_model):
        W = sklearn_model.coef_.T.reshape(-1)
        params = [W]

        if self.fit_intercept:
            b = sklearn_model.intercept_
            params.append(b)

        self.set_params(params)

    # Extracting information

    def get_total_loss(self, dataset, sample_weights=None, reg=False, **kwargs):
        if sample_weights is None:
            sample_weights = np.ones(dataset.num_examples)

        if reg:
            l2_reg = kwargs.get('l2_reg', self.config['default_l2_reg'])
            self.set_l2_reg(l2_reg)
            loss_op = self.total_loss_reg
        else:
            loss_op = self.total_loss_no_reg

        loss = self.sess.run(loss_op, feed_dict={
            self.input_placeholder: dataset.x,
            self.labels_placeholder: dataset.labels,
            self.sample_weights_placeholder: sample_weights,
        })
        return loss

    def get_indiv_loss(self, dataset, **kwargs):
        indiv_loss = self.sess.run(self.indiv_loss, feed_dict={
            self.input_placeholder: dataset.x,
            self.labels_placeholder: dataset.labels,
        })
        return indiv_loss

    def get_total_grad_loss(self, dataset, sample_weights=None, reg=False, **kwargs):
        if sample_weights is None:
            sample_weights = np.ones(dataset.num_examples)

        if reg:
            l2_reg = kwargs.get('l2_reg', self.config['default_l2_reg'])
            self.set_l2_reg(l2_reg)
            grad_loss_op = self.total_grad_loss_reg_flat
        else:
            grad_loss_op = self.total_grad_loss_no_reg_flat

        grad_loss = self.sess.run(grad_loss_op, feed_dict={
            self.input_placeholder: dataset.x,
            self.labels_placeholder: dataset.labels,
            self.sample_weights_placeholder: sample_weights,
        })
        return grad_loss

    def get_indiv_grad_loss(self, dataset, **kwargs):
        method = kwargs.get('method', 'batched')

        if method == "batched":
            # This works only when we can explicitly compute the individual gradients
            return self.get_indiv_grad_loss_batched(dataset, **kwargs)
        elif method == "from_total_grad":
            # The default method
            return self.get_indiv_grad_loss_from_total_grad(dataset, **kwargs)
        else:
            raise ValueError('Unknown method {}'.format(method))

    def get_indiv_grad_loss_batched(self, dataset, **kwargs):
        if not hasattr(self, 'indiv_grad_loss'):
            raise Exception('Batched gradient evaluation not supported')

        batch_size = kwargs.get('grad_batch_size', 256)
        indiv_grad_losses = []
        for i in range(0, dataset.num_examples, batch_size):
            print("\rGradients computed: {}/{}".format(i, dataset.num_examples), end="")
            end = min(i + batch_size, dataset.num_examples)
            indiv_grad_loss = self.sess.run(self.indiv_grad_loss, feed_dict={
                self.input_placeholder: dataset.x[i:end, :],
                self.labels_placeholder: dataset.labels[i:end],
            })
            print("\rGradients computed: {}/{}".format(end, dataset.num_examples), end="")
            indiv_grad_losses.append(indiv_grad_loss)
        print()
        return np.vstack(indiv_grad_losses)

    def get_hessian_reg(self, dataset, sample_weights=None, **kwargs):
        l2_reg = kwargs.get('l2_reg', self.config['default_l2_reg'])
        self.set_l2_reg(l2_reg)

        if sample_weights is None:
            sample_weights = np.ones(dataset.num_examples)

        batch_size = kwargs.get('hess_batch_size', 256)
        hessian_reg = self.sess.run(self.hessian_of_reg)
        for i in range(0, dataset.num_examples, batch_size):
            print("\rHessians computed: {}/{}".format(i, dataset.num_examples), end="")
            end = min(i + batch_size, dataset.num_examples)
            hessian = self.sess.run(self.hessian_no_reg, feed_dict={
                self.input_placeholder: dataset.x[i:end, :],
                self.labels_placeholder: dataset.labels[i:end],
                self.sample_weights_placeholder: sample_weights[i:end],
            })
            print("\rHessians computed: {}/{}".format(end, dataset.num_examples), end="")
            hessian_reg += hessian
        print()
        return hessian_reg

    def get_inverse_hvp_reg(self, dataset, vectors, sample_weights=None, **kwargs):
        """
        Computes the inverse Hessian vector product with a list of vectors.
        The Hessian contains l2 regularization.

        :param dataset: The dataset to compute the hessian on.
        :param vectors: A (D, K) numpy array where D is the total number of parameters
        :param sample_weights: The sample weights for the dataset.
        :return: A (D, K) numpy array = hessian^{-1} vectors
        """
        assert vectors.shape[0] == self.params_flat.shape[0]

        hessian_reg = self.get_hessian_reg(dataset, sample_weights, **kwargs)
        return self.get_inverse_vp(self, hessian_reg, vectors, **kwargs)

    def get_inverse_vp(self, matrix, vectors, **kwargs):
        inverse_vp = self.sess.run(self.inverse_vp, feed_dict={
            self.matrix_placeholder: matrix,
            self.vectors_placeholder: vectors.T,
        })
        return inverse_vp

    # Margins (only for binary classification)

    def get_indiv_margin(self, dataset, **kwargs):
        assert self.num_classes == 2, "Margins only supported for binary classification"

        indiv_margin = self.sess.run(self.margins, feed_dict={
            self.input_placeholder: dataset.x,
            self.labels_placeholder: dataset.labels,
        })
        return indiv_margin

    def get_total_grad_margin(self, dataset, sample_weights=None, **kwargs):
        assert self.num_classes == 2, "Margins only supported for binary classification"

        if sample_weights is None:
            sample_weights = np.ones(dataset.num_examples)

        grad_margin = self.sess.run(self.total_grad_margin, feed_dict={
            self.input_placeholder: dataset.x,
            self.labels_placeholder: dataset.labels,
            self.sample_weights_placeholder: sample_weights,
        })
        return grad_margin

    # Evaluation

    def print_model_eval(self, datasets):
        params_flat = self.get_params_flat()

        train_loss_reg, train_loss_no_reg, train_acc = self.sess.run(
            [self.total_loss_reg, self.total_loss_no_reg, self.accuracy],
            feed_dict={
                self.input_placeholder: datasets.train.x,
                self.labels_placeholder: datasets.train.labels,
                self.sample_weights_placeholder: np.ones(datasets.train.x.shape[0]),
            })

        test_loss_reg, test_loss_no_reg, test_acc = self.sess.run(
            [self.total_loss_reg, self.total_loss_no_reg, self.accuracy],
            feed_dict={
                self.input_placeholder: datasets.test.x,
                self.labels_placeholder: datasets.test.labels,
                self.sample_weights_placeholder: np.ones(datasets.test.x.shape[0]),
            })

        train_total_grad_loss = self.sess.run(
            [self.total_grad_loss_no_reg_flat],
            feed_dict={
                self.input_placeholder: datasets.train.x,
                self.labels_placeholder: datasets.train.labels,
                self.sample_weights_placeholder: np.ones(datasets.train.x.shape[0]),
            })

        print('Train loss (w reg) on all data: %s' %
              (train_loss_reg / datasets.train.num_examples))
        print('Train loss (w/o reg) on all data: %s' %
              (train_loss_no_reg / datasets.train.num_examples))

        print('Test loss (w/o reg) on all data: %s' %
              (test_loss_no_reg / datasets.test.num_examples))
        print('Train acc on all data:  %s' % train_acc)
        print('Test acc on all data:   %s' % test_acc)

        print('Norm of the mean of gradients: %s' %
              (np.linalg.norm(train_total_grad_loss) / datasets.train.num_examples))
        print('Norm of the params: %s' % np.linalg.norm(params_flat))

    def get_accuracy(self, dataset):
        accuracy = self.sess.run(self.accuracy, feed_dict={
            self.input_placeholder: dataset.x,
            self.labels_placeholder: dataset.labels,
        })
        return accuracy

    def get_predictions(self, X):
        sample_weights = np.ones(X.shape[0])
        predictions = self.sess.run(self.predictions, feed_dict={
            self.input_placeholder: X
        })
        return predictions

    @staticmethod
    def infer_arch(dataset):
        arch = dict()
        arch['input_dim'] = dataset.x.shape[1]
        arch['fit_intercept'] = False
        arch['num_classes'] = np.unique(dataset.labels).shape[0]
        return arch

    @staticmethod
    def default_config():
        return {
            # The tensorflow initialization seed
            'tf_init_seed': 0,

            # The L2 regularization to use if not overriden by a keyword
            # argument to model.fit()
            'default_l2_reg': 1,

            'max_lbfgs_iter': 2048,
        }
