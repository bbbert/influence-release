from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import abc
import sys

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse 

import os.path
import time
import IPython
import tensorflow as tf
import math
import copy

from influence.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay
from influence.dataset import DataSet

def conv2d(x, W, r):
    return tf.nn.conv2d(x, W, strides=[1, r, r, 1], padding='VALID')

def softplus(x):
    return tf.log(tf.exp(x) + 1)


class All_CNN_C(GenericNeuralNet):

    def __init__(self, config_dict):
        spec_dict = config_dict['spec']

        self.weight_decay = spec_dict['weight_decay']
        self.input_side = spec_dict['input_side']
        self.input_channels = spec_dict['input_channels']
        self.conv_patch_size = spec_dict['conv_patch_size']
        self.hidden_units = spec_dict['hidden_units']

        self.num_hidden = len(hidden_units)
        self.input_dim = self.input_side * self.input_side * self.input_channels

        super(All_CNN_C, self).__init__(config_dict)


    def conv2d_softplus(self, input_x, conv_patch_size, input_channels, output_channels, stride):
        weights = variable_with_weight_decay(
            'weights', 
            [conv_patch_size * conv_patch_size * input_channels * output_channels],
            stddev=2.0 / math.sqrt(float(conv_patch_size * conv_patch_size * input_channels)),
            wd=self.weight_decay)
        if num_hidden == 0:
            hidden = tf.nn.tanh(conv2d(input_x, weights_reshaped, stride))
            return hidden

        biases = variable(
            'biases',
            [output_channels],
            tf.constant_initializer(0.0))
        weights_reshaped = tf.reshape(weights, [conv_patch_size, conv_patch_size, input_channels, output_channels])
        hidden = tf.nn.tanh(conv2d(input_x, weights_reshaped, stride) + biases)

        return hidden



    def get_all_params(self):
        all_params = []
        for layer in ['h{}_a'.format(i+1) for i in range(num_hidden)] + ['h{}_c'.format(i+1) for i in range(num_hidden)] + ['softmax_linear']:
            if (num_hidden == 0): names = ['weights']
            else: names = ['weights', 'biases']
            for var_name in names:
               temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))
               all_params.append(temp_tensor)      
        return all_params        
        

    def warm_retrain(self, start_step, end_step, idx, feed_dict):
        omits = np.zeros(self.num_train_examples,dtype=bool)
        if idx is not None:
            omits[idx] = True
        self.data_sets.train.set_omits(omits)
        self.data_sets.train.reset_clone()

        start_time = time.time()

        retrain_losses = []
        for step in xrange(start_step,end_step):
            self.update_learning_rate(step)
            iter_feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train,which_rng="clone",batch_size=0)
            #iter_feed_dict = self.fill_feed_dict_with_all_ex(retrain_dataset)
            self.sess.run(self.train_op, feed_dict=iter_feed_dict)
            if step % 20000 == 0:
                print('Step {} took {} sec'.format(step,time.time() - start_time))
                start_time = time.time()
            if step % 1000 == 0:
                feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.test,self.test_point)
                retrain_losses.append(self.sess.run(self.loss_no_reg,feed_dict=feed_dict))
        
        np.savez('../output/{}_remove{}_retrain_losses'.format(self.model_name,idx),retrain_losses=retrain_losses)
        print(retrain_losses[((end_step-start_step)//1000)-1])

        self.data_sets.train.reset_omits()



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


    def inference(self, input_x):

        if num_hidden == 0:
            last_layer_units = self.input_dim

            with tf.variable_scope('softmax_linear'):
                weights = variable_with_weight_decay(
                        'weights',
                        [last_layer_units * self.num_classes],
                        stddev=1.0/math.sqrt(float(last_layer_units)),
                        wd=self.weight_decay)
                logits = tf.matmul(input_x, tf.reshape(weights, [last_layer_units, self.num_classes]))
            return logits

        input_reshaped = tf.reshape(input_x, [-1, self.input_side, self.input_side, self.input_channels])

        h_a = []
        h_c = []

        with tf.variable_scope('h1_a'):
            h_a.append(self.conv2d_softplus(input_reshaped, self.conv_patch_size, self.input_channels, self.hidden_units[0], stride=1))

        for i in range(num_hidden-1):
            with tf.variable_scope('h{}_c'.format(i+1)):
                h_c.append(self.conv2d_softplus(h_a[-1], self.conv_patch_size, self.hidden_units[i], self.hidden_units[i], stride=2))
            with tf.variable_scope('h{}_a'.format(i+2)):
                h_a.append(self.conv2d_softplus(h_c[-1], self.conv_patch_size, self.hidden_units[i], self.hidden_units[i+1], stride=1))

        last_layer_units = self.num_classes
        with tf.variable_scope('h{}_c'.format(self.num_hidden)):
            h_c.append(self.conv2d_softplus(h_a[-1], 1, self.hidden_units[-1], last_layer_units, stride=1))
        
        h_d = tf.reduce_mean(h_c[-1], axis=[1, 2])
        
        with tf.variable_scope('softmax_linear'):

            weights = variable_with_weight_decay(
                'weights', 
                [last_layer_units * self.num_classes],
                stddev=1.0 / math.sqrt(float(last_layer_units)),
                wd=self.weight_decay)            
            biases = variable(
                'biases',
                [self.num_classes],
                tf.constant_initializer(0.0))

            logits = tf.matmul(h_d, tf.reshape(weights, [last_layer_units, self.num_classes])) + biases
            
        return logits

    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds
