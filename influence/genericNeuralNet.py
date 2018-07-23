from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import abc
import sys

import warnings

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster

import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse
from scipy.optimize import fmin_ncg

import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import array_ops
from keras import backend as K
from tensorflow.contrib.learn.python.learn.datasets import base

from influence.hessians import hessian_vector_product
from influence.dataset import DataSet
from load_mnist import load_mnist, load_small_mnist
from load_cifar10 import load_cifar10, load_small_cifar10


def variable(name, shape, initializer):
    dtype = tf.float32
    var = tf.get_variable(
        name, 
        shape, 
        initializer=initializer, 
        dtype=dtype)
    return var

def variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    var = variable(
        name, 
        shape, 
        initializer=tf.truncated_normal_initializer(
            stddev=stddev, 
            dtype=dtype))
 
    if wd is not None:
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)

    return var

def normalize_vector(v):
    """
    Takes in a vector in list form, concatenates it to form a single vector,
    normalizes it to unit length, then returns it in list form together with its norm.
    """
    norm_val = np.linalg.norm(np.concatenate(v))
    norm_v = [a/norm_val for a in v]
    return norm_v, norm_val


class GenericNeuralNet(object):
    """
    Multi-class classification.
    """

    def __init__(self, config_dict):
        
        gen_dict = config_dict['gen']

        self._batching_seed = gen_dict['batching_seed']
        self._initialization_seed = gen_dict['initialization_seed']

        self.batch_size = gen_dict['batch_size']
        self.dataset_type = gen_dict['dataset_type']
        if self.dataset_type == 'mnist':
            self.data_sets = load_mnist('data')
            print('LOADED FULL MNIST')
        elif self.dataset_type == 'mnist_small':
            self.data_sets = load_small_mnist('data')
            print('LOADED SMALL MNIST')
        elif self.dataset_type == 'cifar10':
            self.data_sets = load_cifar10('data')
            print('LOADED FULL CIFAR10')
        elif self.dataset_type == 'cifar10_small':
            self.data_sets = load_small_cifar10('data')
            print('LOADED SMALL CIFAR10')
        else:
            warnings.warn('Invalid dataset')

        for dataset in self.data_sets:
            if dataset is not None:
                dataset.set_randomState_and_reset_rngs(self._batching_seed)
        self.data_sets.train.reset_omits()

        self.train_dir = gen_dict['train_dir']
        self.log_dir = gen_dict['log_dir'] #unused
        self.model_name = gen_dict['model_name']
        self.num_classes = gen_dict['num_classes']
        self.initial_learning_rate = gen_dict['initial_learning_rate']
        self.decay_epochs = gen_dict['decay_epochs']
        self.keep_probs = gen_dict['keep_probs']
        self.mini_batch = gen_dict['mini_batch']
        self.damping = gen_dict['damping']
        self.test_point = gen_dict['test_point']


        # Default params for certain functions
        self.lissa_params = gen_dict['lissa_params']
        self.fmin_ncg_params = gen_dict['fmin_ncg_params']
        self.test_grad_batch_size = gen_dict['test_grad_batch_size']

        #if 'keep_probs' in kwargs: self.keep_probs = kwargs.pop('keep_probs')
        
        #else: self.keep_probs = None
        
        #if 'mini_batch' in kwargs: self.mini_batch = kwargs.pop('mini_batch')        
        #else: self.mini_batch = True
        
        #if 'damping' in kwargs: self.damping = kwargs.pop('damping')
        #else: self.damping = 0.0
         
        np.random.seed(self._batching_seed)
        
        # This sets the global tf random seed. There are also operation-level random
        # seeds, which we can sort of ignore if we set the global seed. However,
        # it doesn't seem possible to get the intermediate internal state of this tf
        # seed, so that can cause an issue when trying to restore a model midway.
        # In current code, this doesn't matter because tf randomness is only used
        # for variable initialization--just remember to always initialize in the same
        # order! And if tf randomness plays a role in later features, we may want to
        # control the op-level seeds.
        tf.set_random_seed(self._initialization_seed)
       
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        # Initialize session
        config = tf.ConfigProto()        
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)
        
        # Setup input
        self.input_placeholder, self.labels_placeholder = self.placeholder_inputs()
        self.num_train_examples = self.data_sets.train.labels.shape[0]
        self.num_test_examples = self.data_sets.test.labels.shape[0]
        
        # Setup inference and training
        if self.keep_probs is not None:
            self.keep_probs_placeholder = tf.placeholder(tf.float32, shape=(2))
            warnings.warn("No longer using keep_probs", DeprecationWarning)
            self.logits = self.inference(self.input_placeholder, self.keep_probs_placeholder)
        elif hasattr(self, 'inference_needs_labels'):            
            self.logits = self.inference(self.input_placeholder, self.labels_placeholder)
        else:
            self.logits = self.inference(self.input_placeholder)

        self.total_loss, self.loss_no_reg, self.indiv_loss_no_reg = self.loss(
            self.logits, 
            self.labels_placeholder)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.Variable(self.initial_learning_rate, name='learning_rate', trainable=False)
        self.learning_rate_placeholder = tf.placeholder(tf.float32)
        self.update_learning_rate_op = tf.assign(self.learning_rate, self.learning_rate_placeholder)
        
        self.train_op = self.get_train_op(self.total_loss, self.global_step, self.learning_rate)
        self.train_sgd_op = self.get_train_sgd_op(self.total_loss, self.global_step, self.learning_rate)
        self.accuracy_op = self.get_accuracy_op(self.logits, self.labels_placeholder)        
        self.preds = self.predictions(self.logits)

        # Setup misc
        self.saver = tf.train.Saver()
        self.test_losses = []
        self.test_losses_fine = []


        # Setup gradients and Hessians
        self.params = self.get_all_params()
        self.grad_total_loss_op = tf.gradients(self.total_loss, self.params)
        self.grad_loss_no_reg_op = tf.gradients(self.loss_no_reg, self.params)
        self.v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in self.params]
        self.u_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in self.params]

        self.hessian_vector = hessian_vector_product(self.total_loss, self.params, self.v_placeholder)

        self.grad_loss_wrt_input_op = tf.gradients(self.total_loss, self.input_placeholder)        

        # Because tf.gradients auto accumulates, we probably don't need the add_n (or even reduce_sum)        
        self.influence_op = tf.add_n(
            [tf.reduce_sum(tf.multiply(a, array_ops.stop_gradient(b))) for a, b in zip(self.grad_total_loss_op, self.v_placeholder)])

        self.grad_influence_wrt_input_op = tf.gradients(self.influence_op, self.input_placeholder)
    
        self.checkpoint_file = os.path.join(self.train_dir, "%s-checkpoint" % self.model_name)
        self.rngs_file = os.path.join(self.train_dir, "%s-rngs-checkpoint" % self.model_name)
        self.test_losses_file = os.path.join(self.train_dir, "%s-tracked-losses-checkpoint" % self.model_name)

        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)
        self.all_test_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.test)

        init = tf.global_variables_initializer()        
        self.sess.run(init)

        self.vec_to_list = self.get_vec_to_list_fn()
        self.adversarial_loss, self.indiv_adversarial_loss = self.adversarial_loss(self.logits, self.labels_placeholder)
        if self.adversarial_loss is not None:
            self.grad_adversarial_loss_op = tf.gradients(self.adversarial_loss, self.params)


    def get_vec_to_list_fn(self):
        params_val = self.sess.run(self.params)
        self.num_params = len(np.concatenate(params_val))        
        print('Total number of parameters: %s' % self.num_params)


        def vec_to_list(v):
            return_list = []
            cur_pos = 0
            for p in params_val:
                return_list.append(v[cur_pos : cur_pos+len(p)])
                cur_pos += len(p)

            assert cur_pos == len(v)
            return return_list

        return vec_to_list


    def reset_datasets(self):
        for data_set in self.data_sets:
            if data_set is not None:
                data_set.reset_batch()

    def warn_about_fill_feed_dict(self):
        warnings.warn("Use a dataset object and its omits instead of fill_feed_dict", DeprecationWarning)

    def fill_feed_dict_with_all_ex(self, data_set):
        self.warn_about_fill_feed_dict()
        feed_dict = {
            self.input_placeholder: data_set.x,
            self.labels_placeholder: data_set.labels
        }
        return feed_dict


    def fill_feed_dict_with_all_but_one_ex(self, data_set, idx_to_remove):

        self.warn_about_fill_feed_dict()
        idx = np.array([True] * data_set.num_examples, dtype=bool)
        idx[idx_to_remove] = False
        feed_dict = {
            self.input_placeholder: data_set.x[idx, :],
            self.labels_placeholder: data_set.labels[idx]
        }
        return feed_dict


    def fill_feed_dict_with_batch(self, data_set, which_rng, batch_size, verbose=False):
        self.warn_about_fill_feed_dict()
        if batch_size is None:
            return self.fill_feed_dict_with_all_ex(data_set)
        elif batch_size == 0:
            batch_size = self.batch_size
    
        input_feed, labels_feed = data_set.next_batch(batch_size, which_rng, verbose=verbose)

        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,            
        }
        return feed_dict


    def fill_feed_dict_with_some_ex(self, data_set, target_indices):
        self.warn_about_fill_feed_dict()
        input_feed = data_set.x[target_indices, :].reshape(len(target_indices), -1)
        labels_feed = data_set.labels[target_indices].reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict


    def fill_feed_dict_with_one_ex(self, data_set, target_idx):
        self.warn_about_fill_feed_dict()
        input_feed = data_set.x[target_idx, :].reshape(1, -1)
        labels_feed = data_set.labels[target_idx].reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict


    def fill_feed_dict_manual(self, X, Y):
        self.warn_about_fill_feed_dict()
        X = np.array(X)
        Y = np.array(Y) 
        input_feed = X.reshape(len(Y), -1)
        labels_feed = Y.reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict        


    def minibatch_mean_eval(self, ops, data_set):

        num_examples = data_set.num_examples
        #assert num_examples % self.batch_size == 0
        num_iter = (num_examples + self.batch_size - 1) // self.batch_size
        #num_iter = int(num_examples / self.batch_size)

        data_set.reset_orig()

        ret = []
        for i in xrange(num_iter):
            this_batch_size = min(self.batch_size, num_examples - self.batch_size * i)
            feed_dict = self.fill_feed_dict_with_batch(data_set, which_rng="orig", batch_size=this_batch_size)
            #feed_dict = self.fill_feed_dict_with_batch(data_set)
            ret_temp = self.sess.run(ops, feed_dict=feed_dict)
            
            if len(ret)==0:
                for b in ret_temp:
                    if isinstance(b, list):
                        ret.append([c / float(num_iter) for c in b])
                    else:
                        ret.append([b / float(num_iter)])
            else:
                for counter, b in enumerate(ret_temp):
                    if isinstance(b, list):
                        ret[counter] = [a + (c / float(num_iter)) for (a, c) in zip(ret[counter], b)]
                    else:
                        ret[counter] += (b / float(num_iter))
            
        return ret


    def print_model_eval(self):
        params_val = self.sess.run(self.params)

        if self.mini_batch == True:
            grad_loss_val, loss_no_reg_val, loss_val, train_acc_val = self.minibatch_mean_eval(
                [self.grad_total_loss_op, self.loss_no_reg, self.total_loss, self.accuracy_op],
                self.data_sets.train)
            
            test_loss_val, test_acc_val = self.minibatch_mean_eval(
                [self.loss_no_reg, self.accuracy_op],
                self.data_sets.test)

        else:
            grad_loss_val, loss_no_reg_val, loss_val, train_acc_val = self.sess.run(
                [self.grad_total_loss_op, self.loss_no_reg, self.total_loss, self.accuracy_op], 
                feed_dict=self.all_train_feed_dict)

            test_loss_val, test_acc_val = self.sess.run(
                [self.loss_no_reg, self.accuracy_op], 
                feed_dict=self.all_test_feed_dict)

        print('Train loss (w reg) on all data: %s' % loss_val)
        print('Train loss (w/o reg) on all data: %s' % loss_no_reg_val)

        print('Test loss (w/o reg) on all data: %s' % test_loss_val)
        print('Train acc on all data:  %s' % train_acc_val)
        print('Test acc on all data:   %s' % test_acc_val)

        print('Norm of the mean of gradients: %s' % np.linalg.norm(np.concatenate(grad_loss_val)))
        print('Norm of the params: %s' % np.linalg.norm(np.concatenate(params_val)))



    def warm_retrain(self, start_step, end_step, feed_dict, idx):
        for step in xrange(start_step,end_step):
            self.update_learning_rate(step)
            self.sess.run(self.train_op, feed_dict=feed_dict)



    def update_learning_rate(self, step):
        #assert self.num_train_examples % self.batch_size == 0
        num_steps_in_epoch = (self.num_train_examples + self.batch_size - 1) // self.batch_size
        #num_steps_in_epoch = self.num_train_examples / self.batch_size
        epoch = step // num_steps_in_epoch
        if step % 1000 == 0:
            print('Epoch {}'.format(epoch))

        multiplier = 1
        for i in self.decay_epochs:
            if epoch >= i:
                multiplier = multiplier / 10
        """
        if epoch < self.decay_epochs[0]:
            multiplier = 1
        elif epoch < self.decay_epochs[1]:
            multiplier = 0.1
        else:
            multiplier = 0.01
        """
        self.sess.run(
            self.update_learning_rate_op, 
            feed_dict={self.learning_rate_placeholder: multiplier * self.initial_learning_rate})        


    def train(self, num_steps, iter_to_switch_to_batch, iter_to_switch_to_sgd,
              save_checkpoints=True, verbose=True, track_losses=True):
        """
        Trains a model for a specified number of steps.
        """
        if verbose: print('Training for %s steps' % num_steps)

        sess = self.sess

        self.data_sets.train.reset_rng()

        test_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.test,self.test_point)

        for step in xrange(num_steps):
            self.update_learning_rate(step)

            start_time = time.time()

            if step < iter_to_switch_to_batch:
                feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, which_rng="normal", batch_size=0, verbose = False)#(step % 500 == 0)) ###
                _, loss_val = sess.run([self.train_op, self.total_loss], feed_dict=feed_dict)
                
            elif step < iter_to_switch_to_sgd:
                feed_dict = self.all_train_feed_dict          
                _, loss_val = sess.run([self.train_op, self.total_loss], feed_dict=feed_dict)

            else: 
                feed_dict = self.all_train_feed_dict          
                _, loss_val = sess.run([self.train_sgd_op, self.total_loss], feed_dict=feed_dict)          

            duration = time.time() - start_time

            if verbose:
                if step % 1000 == 0:
                    # Print status to stdout.
                    print('Step %d: train loss = %.8f, test loss = %.8f (%.3f sec)' % (step, sess.run(self.total_loss, feed_dict=self.all_train_feed_dict), sess.run(self.loss_no_reg,feed_dict=test_feed_dict), duration)) ########

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 100000 == 0 or (step + 1) == num_steps:
                if save_checkpoints:
                    self.save(step)
                if verbose: self.print_model_eval()

            if track_losses:
                if step < 100 or step % 1000 == 0:
                    if step < 100:
                        self.test_losses_fine.append(sess.run(self.loss_no_reg, feed_dict=test_feed_dict))
                    if step % 1000 == 0:
                        self.test_losses.append(sess.run(self.loss_no_reg, feed_dict=test_feed_dict))
        
        #print(self.test_losses_fine)
        #print(self.test_losses)

    def save(self, step):
        self.saver.save(self.sess, self.checkpoint_file, global_step=step)
        # Note: we don't save orig or clone info
        states = [dataset._rng.get_state() for dataset in self.data_sets]
        arrs = [state[1] for state in states]
        poss = [state[2] for state in states]
        gausses = [state[3] for state in states]
        caches = [state[4] for state in states]
        batch_indices = [dataset._batch_indices for dataset in self.data_sets]
        epoch_indices = [dataset._indices_in_epoch for dataset in self.data_sets]
        np.savez(self.rngs_file, arrs=arrs,poss=poss,gausses=gausses,caches=caches,
                batch_indices=batch_indices,epoch_indices=epoch_indices)
        np.savez(self.test_losses_file, test_losses=self.test_losses, test_losses_fine=self.test_losses_fine)

    def get_all_losses(self):
        return self.test_losses, self.test_losses_fine


    def load_checkpoint(self, iter_to_load, do_checks=True):
        checkpoint_to_load = "%s-%s" % (self.checkpoint_file, iter_to_load) 
        self.saver.restore(self.sess, checkpoint_to_load)
        if os.path.exists(self.rngs_file+".npz"):
            f = np.load(self.rngs_file + ".npz")
            name = 'MT19937'
            arrs = f['arrs']
            poss = f['poss']
            gausses = f['gausses']
            caches = f['caches']
            batch_indices = f['batch_indices']
            epoch_indices = f['epoch_indices']
            for i, dataset in enumerate(self.data_sets):
                dataset._rng.set_state((name, arrs[i], poss[i], gausses[i], caches[i]))
                dataset._batch_indices = batch_indices[i]
                dataset._indices_in_epoch = epoch_indices[i]
        else:
            warnings.warn("NOT RELOADING DATASET RANDOM STATES")

        if (os.path.exists(self.test_losses_file+".npz")):
            f = np.load(self.test_losses_file+".npz")
            self.test_losses = f['test_losses']
            self.test_losses_fine = f['test_losses_fine']
        else:
            warnings.warn("NOT RELOADING TRACKED TEST LOSSES")

        if do_checks:
            print('Model %s loaded. Sanity checks ---' % checkpoint_to_load)
            self.print_model_eval()


    def get_train_op(self, total_loss, global_step, learning_rate):
        """
        Return train_op

        WARNING: does the momentum carrying over here cause problems?
        Probably is saved by the saver but still...
        """
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)
        return train_op


    def get_train_sgd_op(self, total_loss, global_step, learning_rate):
        """
        Return train_sgd_op
        """
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)
        return train_op


    def get_accuracy_op(self, logits, labels):
        """Evaluate the quality of the logits at predicting the label.
        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).
        Returns:
          A scalar int32 tensor with the number of examples (out of batch_size)
          that were predicted correctly.
        """        
        correct = tf.nn.in_top_k(logits, labels, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32)) / tf.shape(labels)[0]


    def loss(self, logits, labels):

        labels = tf.one_hot(labels, depth=self.num_classes)
        # correct_prob = tf.reduce_sum(tf.multiply(labels, tf.nn.softmax(logits)), reduction_indices=1)
        cross_entropy = - tf.reduce_sum(tf.multiply(labels, tf.nn.log_softmax(logits)), reduction_indices=1)

        indiv_loss_no_reg = cross_entropy
        loss_no_reg = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        tf.add_to_collection('losses', loss_no_reg)

        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return total_loss, loss_no_reg, indiv_loss_no_reg


    def adversarial_loss(self, logits, labels):
        # Computes sum of log(1 - p(y = true|x))
        # No regularization (because this is meant to be computed on the test data)

        labels = tf.one_hot(labels, depth=self.num_classes)        
        wrong_labels = (labels - 1) * -1 # Flips 0s and 1s
        wrong_labels_bool = tf.reshape(tf.cast(wrong_labels, tf.bool), [-1, self.num_classes])

        wrong_logits = tf.reshape(tf.boolean_mask(logits, wrong_labels_bool), [-1, self.num_classes - 1])
        
        indiv_adversarial_loss = tf.reduce_logsumexp(wrong_logits, reduction_indices=1) - tf.reduce_logsumexp(logits, reduction_indices=1)
        adversarial_loss = tf.reduce_mean(indiv_adversarial_loss)
        
        return adversarial_loss, indiv_adversarial_loss #, indiv_wrong_prob


    def update_feed_dict_with_v_placeholder(self, feed_dict, vec):
        for pl_block, vec_block in zip(self.v_placeholder, vec):
            feed_dict[pl_block] = vec_block        
        return feed_dict


    def get_inverse_hvp(self, v, approx_type='cg', approx_params=None, verbose=True):
        assert approx_type in ['cg', 'lissa']
        if approx_type == 'lissa':
            if approx_params is None:
                approx_params = self.lissa_params
            return self.get_inverse_hvp_lissa(v, **approx_params)
        elif approx_type == 'cg':
            return self.get_inverse_hvp_cg(v, verbose=verbose, **self.fmin_ncg_params)


    def get_inverse_hvp_lissa(self, v, batch_size, scale, damping, num_samples, recursion_depth):
        """
        This uses mini-batching; uncomment code for the single sample case.
        """

        inverse_hvp = None
        print_iter = recursion_depth / 10

        self.data_sets.train.reset_orig()

        for i in range(num_samples):
            # samples = np.random.choice(self.num_train_examples, size=recursion_depth)
           
            cur_estimate = v

            for j in range(recursion_depth):
             
                # feed_dict = fill_feed_dict_with_one_ex(
                #   data_set, 
                #   images_placeholder, 
                #   labels_placeholder, 
                #   samples[j])   
                feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, which_rng="orig", batch_size=batch_size)

                feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, cur_estimate)
                hessian_vector_val = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
                cur_estimate = [a + (1-damping) * b - c/scale for (a,b,c) in zip(v, cur_estimate, hessian_vector_val)]    

                # Update: v + (I - Hessian_at_x) * cur_estimate
                if (j % print_iter == 0) or (j == recursion_depth - 1):
                    print("Recursion at depth %s: norm is %.8lf" % (j, np.linalg.norm(np.concatenate(cur_estimate))))
                    feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, cur_estimate)

            if inverse_hvp is None:
                inverse_hvp = [b/scale for b in cur_estimate]
            else:
                inverse_hvp = [a + b/scale for (a, b) in zip(inverse_hvp, cur_estimate)]  

        inverse_hvp = [a/num_samples for a in inverse_hvp]
        return inverse_hvp
  
    
    def minibatch_hessian_vector_val(self, v):

        num_examples = self.num_train_examples
        if self.mini_batch == True:
            batch_size = self.batch_size #100
            #assert num_examples % batch_size == 0
        else:
            batch_size = self.num_train_examples
        
        num_iter = (num_examples + batch_size - 1) // batch_size
        #num_iter = int(num_examples / batch_size)

        self.data_sets.train.reset_orig()

        hessian_vector_val = None
        for i in xrange(num_iter):
            feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, which_rng="orig", batch_size=batch_size)
            # Can optimize this
            feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, v)
            hessian_vector_val_temp = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
            if hessian_vector_val is None:
                hessian_vector_val = [b / float(num_iter) for b in hessian_vector_val_temp]
            else:
                hessian_vector_val = [a + (b / float(num_iter)) for (a,b) in zip(hessian_vector_val, hessian_vector_val_temp)]
            
        hessian_vector_val = [a + self.damping * b for (a,b) in zip(hessian_vector_val, v)]

        return hessian_vector_val


    def get_fmin_loss_fn(self, v):

        def get_fmin_loss(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))

            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x) - np.dot(np.concatenate(v), x)
        return get_fmin_loss


    def get_fmin_grad_fn(self, v):
        def get_fmin_grad(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))
            
            return np.concatenate(hessian_vector_val) - np.concatenate(v)
        return get_fmin_grad


    def get_fmin_hvp(self, x, p):
        hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(p))

        return np.concatenate(hessian_vector_val)


    def get_cg_callback(self, v, verbose=True):
        fmin_loss_fn = self.get_fmin_loss_fn(v)
        
        def fmin_loss_split(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))

            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x), -np.dot(np.concatenate(v), x)

        def cg_callback(x):
            # x is current params
            v = self.vec_to_list(x)
            idx_to_remove = 5

            single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)      
            train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
            predicted_loss_diff = np.dot(np.concatenate(v), np.concatenate(train_grad_loss_val)) / self.num_train_examples

            if verbose:
                print('Function value: %s' % fmin_loss_fn(x))
                quad, lin = fmin_loss_split(x)
                print('Split function value: %s, %s' % (quad, lin))
                print('Predicted loss diff on train_idx %s: %s' % (idx_to_remove, predicted_loss_diff))

        return cg_callback


    def get_inverse_hvp_cg(self, v, avextol, maxiter, verbose=True):
        fmin_loss_fn = self.get_fmin_loss_fn(v)
        fmin_grad_fn = self.get_fmin_grad_fn(v)
        cg_callback = self.get_cg_callback(v, verbose)

        fmin_results = fmin_ncg(
            f=fmin_loss_fn,
            x0=np.concatenate(v),
            fprime=fmin_grad_fn,
            fhess_p=self.get_fmin_hvp,
            callback=cg_callback,
            avextol=avextol,
            maxiter=maxiter)

        return self.vec_to_list(fmin_results)


    def get_test_grad_loss_no_reg_val(self, test_indices, batch_size, loss_type='normal_loss'):

        if loss_type == 'normal_loss':
            op = self.grad_loss_no_reg_op
        elif loss_type == 'adversarial_loss':
            op = self.grad_adversarial_loss_op
        else:
            raise ValueError('Loss must be specified')

        if test_indices is not None:
            num_iter = int(np.ceil(len(test_indices) / batch_size))

            test_grad_loss_no_reg_val = None
            for i in range(num_iter):
                start = i * batch_size
                end = int(min((i+1) * batch_size, len(test_indices)))

                test_feed_dict = self.fill_feed_dict_with_some_ex(self.data_sets.test, test_indices[start:end])

                temp = self.sess.run(op, feed_dict=test_feed_dict)

                if test_grad_loss_no_reg_val is None:
                    test_grad_loss_no_reg_val = [a * (end-start) for a in temp]
                else:
                    test_grad_loss_no_reg_val = [a + b * (end-start) for (a, b) in zip(test_grad_loss_no_reg_val, temp)]

            test_grad_loss_no_reg_val = [a/len(test_indices) for a in test_grad_loss_no_reg_val]

        else:
            test_grad_loss_no_reg_val = self.minibatch_mean_eval([op], self.data_sets.test)[0]
        
        return test_grad_loss_no_reg_val


    def get_influence_on_test_loss(self, test_indices, train_idx, force_refresh, batch_size,
        approx_type='cg', approx_params=None, test_description=None,
        loss_type='normal_loss',
        X=None, Y=None):
        
        if batch_size == 'default':
            batch_size = self.test_grad_batch_size

        # If train_idx is None then use X and Y (phantom points)
        # Need to make sure test_idx stays consistent between models
        # because mini-batching permutes dataset order

        if train_idx is None: 
            if (X is None) or (Y is None): raise ValueError('X and Y must be specified if using phantom points.')
            if X.shape[0] != len(Y): raise ValueError('X and Y must have the same length.')
        else:
            if (X is not None) or (Y is not None): raise ValueError('X and Y cannot be specified if train_idx is specified.')

        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, batch_size=batch_size, loss_type=loss_type)

        print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (self.model_name, approx_type, loss_type, test_description))
        if os.path.exists(approx_filename) and force_refresh == False:
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            print('Loaded inverse HVP from %s' % approx_filename)
        else:
            inverse_hvp = self.get_inverse_hvp(
                test_grad_loss_no_reg_val,
                approx_type,
                approx_params,
                verbose=True)

            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            print('Saved inverse HVP to %s' % approx_filename)

        duration = time.time() - start_time
        print('Inverse HVP took %s sec' % duration)



        start_time = time.time()
        if train_idx is None:
            num_to_remove = len(Y)
            predicted_loss_diffs = np.zeros([num_to_remove])            
            for counter in np.arange(num_to_remove):
                single_train_feed_dict = self.fill_feed_dict_manual(X[counter, :], [Y[counter]])      
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / self.num_train_examples            

        else:            
            num_to_remove = len(train_idx)
            predicted_loss_diffs = np.zeros([num_to_remove])
            for counter, idx_to_remove in enumerate(train_idx):            
                single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / self.num_train_examples
                
        duration = time.time() - start_time
        print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))

        return predicted_loss_diffs



    def find_eigvals_of_hessian(self, num_iter, num_prints=10):

        # Setup        
        print_iterations = num_iter / num_prints
        feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, 0)

        # Initialize starting vector
        grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=feed_dict)
        initial_v = []

        for a in grad_loss_val:
            initial_v.append(np.random.random(a.shape))        
        initial_v, _ = normalize_vector(initial_v)

        # Do power iteration to find largest eigenvalue
        print('Starting power iteration to find largest eigenvalue...')

        largest_eig = norm_val
        print('Largest eigenvalue is %s' % largest_eig)

        # Do power iteration to find smallest eigenvalue
        print('Starting power iteration to find smallest eigenvalue...')
        cur_estimate = initial_v
        
        for i in range(num_iter):          
            cur_estimate, norm_val = normalize_vector(cur_estimate)
            hessian_vector_val = self.minibatch_hessian_vector_val(cur_estimate)
            new_cur_estimate = [a - largest_eig * b for (a,b) in zip(hessian_vector_val, cur_estimate)]

            if i % print_iterations == 0:
                print(-norm_val + largest_eig)
                dotp = np.dot(np.concatenate(new_cur_estimate), np.concatenate(cur_estimate))
                print("dot: %s" % dotp)
            cur_estimate = new_cur_estimate

        smallest_eig = -norm_val + largest_eig
        assert dotp < 0, "Eigenvalue calc failed to find largest eigenvalue"

        print('Largest eigenvalue is %s' % largest_eig)
        print('Smallest eigenvalue is %s' % smallest_eig)
        return largest_eig, smallest_eig


    def get_grad_of_influence_wrt_input(self, train_indices, test_indices, 
        approx_type='cg', approx_params=None, force_refresh=True, verbose=True, test_description=None,
        loss_type='normal_loss'):
        """
        If the loss goes up when you remove a point, then it was a helpful point.
        So positive influence = helpful.
        If we move in the direction of the gradient, we make the influence even more positive, 
        so even more helpful.
        Thus if we want to make the test point more wrong, we have to move in the opposite direction.
        """

        # Calculate v_placeholder (gradient of loss at test point)
        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)            

        if verbose: print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))
        
        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (self.model_name, approx_type, loss_type, test_description))
        
        if os.path.exists(approx_filename) and force_refresh == False:
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            if verbose: print('Loaded inverse HVP from %s' % approx_filename)
        else:            
            inverse_hvp = self.get_inverse_hvp(
                test_grad_loss_no_reg_val,
                approx_type,
                approx_params,
                verbose=verbose)
            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            if verbose: print('Saved inverse HVP to %s' % approx_filename)            
        
        duration = time.time() - start_time
        if verbose: print('Inverse HVP took %s sec' % duration)

        grad_influence_wrt_input_val = None

        for counter, train_idx in enumerate(train_indices):
            # Put in the train example in the feed dict
            grad_influence_feed_dict = self.fill_feed_dict_with_one_ex(
                self.data_sets.train,  
                train_idx)

            self.update_feed_dict_with_v_placeholder(grad_influence_feed_dict, inverse_hvp)

            # Run the grad op with the feed dict
            current_grad_influence_wrt_input_val = self.sess.run(self.grad_influence_wrt_input_op, feed_dict=grad_influence_feed_dict)[0][0, :]            
            
            if grad_influence_wrt_input_val is None:
                grad_influence_wrt_input_val = np.zeros([len(train_indices), len(current_grad_influence_wrt_input_val)])

            grad_influence_wrt_input_val[counter, :] = current_grad_influence_wrt_input_val

        return grad_influence_wrt_input_val


    def override_train_x(self, new_train_x):
        raise DeprecationWarning("Use dataset's self._omit instead of overriding entirely")
        assert np.all(new_train_x.shape == self.data_sets.train.x.shape)
        self.reset_datasets()
        new_train = DataSet(new_train_x, np.copy(self.data_sets.train.labels))
        self.data_sets = base.Datasets(train=new_train, validation=self.data_sets.validation, test=self.data_sets.test)
        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)                


    def override_train_x_y(self, new_train_x, new_train_y):
        raise DeprecationWarning("Use dataset's self._omit instead of overriding entirely")
        self.reset_datasets()
        new_train = DataSet(new_train_x, new_train_y)
        self.data_sets = base.Datasets(train=new_train, validation=self.data_sets.validation, test=self.data_sets.test)
        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)                
        self.num_train_examples = len(new_train_y)      


    def override_test_x_y(self, new_test_x, new_test_y):
        raise DeprecationWarning("Use dataset's self._omit instead of overriding entirely")
        self.reset_dataset()
        new_test = DataSet(new_test_x, new_test_y)
        self.data_sets = base.Datasets(train=self.data_sets.train, validation=self.data_sets.validation, test=new_test)
        self.all_test_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.test)                
        self.num_test_examples = len(new_test_y)

