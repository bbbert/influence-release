from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import abc
import sys

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse 

import os.path
import time
import tensorflow as tf
import math

from tensorflow.python.ops import array_ops

from influence.hessians import hessians
from influence.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay


class LogisticRegressionWithLBFGS(GenericNeuralNet):

    def __init__(self, config_dict):

        spec_dict = config_dict['spec']

        self.weight_decay = spec_dict['weight_decay']
        self.input_dim = spec_dict['input_dim']
        self.max_lbfgs_iter = spec_dict['max_lbfgs_iter']

        self.num_classes = config_dict['gen']['num_classes']

        if spec_dict['has_biases'] is not None:
            self.has_biases = spec_dict['has_biases']
        else:
            self.has_biases = False
                
        if self.num_classes == 2:
            self.multi_class='ovr'
            self.weight_shape = [self.input_dim]
            self.pseudo_num_classes = 1
        else:
            self.multi_class='multinomial'
            self.weight_shape = [self.input_dim * self.num_classes]
            self.pseudo_num_classes = self.num_classes
        
        super(LogisticRegressionWithLBFGS, self).__init__(config_dict)

        if self.num_classes == 2:
            # margin formulation only for binary logistic regression
            y = tf.subtract(tf.multiply(tf.cast(self.labels_placeholder, tf.float32), 2), 1)
            self.margins = tf.multiply(y, self.logits[:, 1])
        else:
            self.margins = None
        
        self.set_params_op = self.set_params()
        # self.hessians_op = hessians(self.total_loss, self.params)        
        
        # Multinomial has weird behavior when it's binary
        C = 1.0 / (self.num_train_examples * self.weight_decay)        
        self.sklearn_model = linear_model.LogisticRegression(
            C=C,
            tol=1e-8,
            fit_intercept=False, 
            solver='lbfgs',
            multi_class=self.multi_class,
            warm_start=True,
            max_iter=self.max_lbfgs_iter)

        C_minus_one = 1.0 / ((self.num_train_examples - 1) * self.weight_decay)
        self.sklearn_model_minus_one = linear_model.LogisticRegression(
            C=C_minus_one,
            tol=1e-8,
            fit_intercept=False, 
            solver='lbfgs',
            multi_class=self.multi_class,
            warm_start=True,
            max_iter=self.max_lbfgs_iter)  


    def get_all_params(self):
        all_params = []
        for layer in ['softmax_linear']:
            if self.has_biases:
                names = ['weights', 'biases']
            else:
                names = ['weights']
            for var_name in names:                
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))            
                all_params.append(temp_tensor)      
        return all_params        
        

    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.float32, 
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,             
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder


    def inference(self, input):        
        with tf.variable_scope('softmax_linear'):
            weights = variable_with_weight_decay(
                'weights', 
                self.weight_shape,
                stddev=1.0 / math.sqrt(float(self.input_dim)),
                wd=self.weight_decay)
            if not self.has_biases:
                logits = tf.matmul(input, tf.reshape(weights, [self.input_dim, self.pseudo_num_classes]))
            else:
                biases = variable(
                    'biases',
                    [self.pseudo_num_classes],
                    tf.constant_initializer(0.0))
                logits = tf.matmul(input, tf.reshape(weights, [self.input_dim, self.pseudo_num_classes])) + biases

        self.weights = weights
        if self.has_biases:
            self.biases = biases

        if self.num_classes == 2:
            zeros = tf.zeros_like(logits)
            logits = tf.concat([zeros, logits], 1)

        return logits


    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds


    def set_params(self):
        # See if we can automatically infer weight shape
        self.W_placeholder = tf.placeholder(
            tf.float32,
            shape=self.weight_shape,
            name='W_placeholder')
        set_weights = tf.assign(self.weights, self.W_placeholder, validate_shape=True)
        if not self.has_biases:
            return [set_weights]
        else:
            self.b_placeholder = tf.placeholder(
                 tf.float32,
                shape=[self.pseudo_num_classes],
                name='b_placeholder')
            set_biases = tf.assign(self.biases, self.b_placeholder, validate_shape=True)
            return [set_weights, set_biases]



    def retrain(self, num_steps, feed_dict, verbose=False):
    
        self.train_with_LBFGS(
            feed_dict=feed_dict,
            save_checkpoints=False, 
            verbose=verbose)
        
        # super(LogisticRegressionWithLBFGS, self).train(
        #     num_steps, 
        #     iter_to_switch_to_batch=0,
        #     iter_to_switch_to_sgd=1000000,
        #     save_checkpoints=False, verbose=False)

    def warm_retrain(self, start_step, end_step, feed_dict, idx):

        self.train_with_LBFGS(
                feed_dict=feed_dict,
                save_checkpoints=False,
                verbose=False)


    def train(self, num_steps=None, 
              iter_to_switch_to_batch=None, 
              iter_to_switch_to_sgd=None,
              save_checkpoints=True, verbose=True, track_losses=True):

        self.train_with_LBFGS(
            feed_dict=self.all_train_feed_dict,
            save_checkpoints=save_checkpoints, 
            verbose=verbose)

        # super(LogisticRegressionWithLBFGS, self).train(
        #     num_steps=500, 
        #     iter_to_switch_to_batch=0,
        #     iter_to_switch_to_sgd=100000,
        #     save_checkpoints=True, verbose=True)


    def train_with_SGD(self, **kwargs):
        super(LogisticRegressionWithLBFGS, self).train(**kwargs)


    def train_with_LBFGS(self, feed_dict, save_checkpoints=True, verbose=True):
        # More sanity checks to see if predictions are the same?        

        X_train = feed_dict[self.input_placeholder]
        Y_train = feed_dict[self.labels_placeholder]
        num_train_examples = len(Y_train)
        assert len(Y_train.shape) == 1
        assert X_train.shape[0] == Y_train.shape[0]

        if num_train_examples == self.num_train_examples:
            if verbose: print('Using normal model')
            model = self.sklearn_model
        elif num_train_examples == self.num_train_examples - 1:
            if verbose: print('Using model minus one')
            model = self.sklearn_model_minus_one
        else:
            C = 1.0 / (num_train_examples * self.weight_decay)
            model = linear_model.LogisticRegression(
                C=C,
                tol=1e-8,
                fit_intercept=False, 
                solver='lbfgs',
                multi_class=self.multi_class,
                warm_start=True,
                max_iter=self.max_lbfgs_iter)  

        model.fit(X_train, Y_train) 
        # sklearn returns coefficients in shape num_classes x num_features
        # whereas our weights are defined as num_features x num_classes
        # so we have to tranpose them first.
        W = np.reshape(model.coef_.T, -1)
        
        params_feed_dict = {}
        params_feed_dict[self.W_placeholder] = W
        if self.has_biases:
            params_feed_dict[self.b_placeholder] = model.intercept_
        self.sess.run(self.set_params_op, feed_dict=params_feed_dict)
        if save_checkpoints: self.saver.save(self.sess, self.checkpoint_file, global_step=0)

        if verbose:
            print('LBFGS training took %s iter.' % model.n_iter_)
            print('After training with LBFGS: ')
            self.print_model_eval()





