from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython

import tensorflow as tf
import time
import os.path

import influence.experiments as experiments
from influence.all_CNN_c_hidden2 import All_CNN_C_Hidden2

from load_mnist import load_small_mnist, load_mnist

data_sets = load_small_mnist('data')    

num_classes = 10
input_side = 28
input_channels = 1
input_dim = input_side * input_side * input_channels 
weight_decay = 0.001
batch_size = 500

initial_learning_rate = 0.0001
decay_epochs = [5000,10000]
hidden1_units = 8
hidden2_units = 8
conv_patch_size = 3
keep_probs = [1.0, 1.0]

first_seed = 0
num_seeds = 41
tot_num_seeds = 41
num_points = 5500
seeds = range(first_seed,first_seed+num_seeds)
train_dir = '../scr/output'
test_idx = 6558

num_steps = 300000
damping = 2e-2

# 1173, 4644 good. 1891, 4936 bad. 1735, 3562 controversial
top_points = [1173, 4644, 1891, 4936, 1735, 3562]
num_top_points = len(top_points)

loss_full = np.zeros(tot_num_seeds, dtype=np.float64)
correct = np.ones(tot_num_seeds, dtype=bool)
loss_remove_one = [{} for _ in range(tot_num_seeds)]
actual_retrain_loss_diffs = [{} for _ in range(tot_num_seeds)]
predicted_loss_diffs = [None for _ in range(tot_num_seeds)]

def get_originals(loss_full, correct, predicted_loss_diffs, first_seed, num_seeds):
    start_time = time.time()
    path_name = '{}/{}-{}seeds_test-{}_loss_and_correctness_and_preds.npz'.format(train_dir,first_seed,first_seed+num_seeds-1,[test_idx])
    if not os.path.exists(path_name):
        for seed in range(first_seed,first_seed+num_seeds):
            print("Starting seed {}".format(seed))
            model_name = 'mnist_small_all_cnn_c_hidden2_seed{}_wd{}_damping{}_iter-{}'.format(seed, int(weight_decay*1000), int(damping*100), num_steps)
            tf.reset_default_graph()
            model = All_CNN_C_Hidden2(
                    input_side=input_side, 
                    input_channels=input_channels,
                    conv_patch_size=conv_patch_size,
                    hidden1_units=hidden1_units, 
                    hidden2_units=hidden2_units,
                    weight_decay=weight_decay,
                    num_classes=num_classes, 
                    batch_size=batch_size,
                    data_sets=data_sets,
                    initial_learning_rate=initial_learning_rate,
                    damping=damping,
                    decay_epochs=decay_epochs,
                    mini_batch=True,
                    train_dir=train_dir, 
                    log_dir='log',
                    model_name=model_name,
                    seed=seed)

            iter_to_load = num_steps - 1
            model.load_checkpoint(iter_to_load)

            sess = model.sess
            test_feed_dict = model.fill_feed_dict_with_one_ex(model.data_sets.test, test_idx)
            loss_full[seed] = sess.run(model.loss_no_reg, feed_dict=test_feed_dict)
            correct[seed] = sess.run(model.accuracy_op, feed_dict=test_feed_dict)

            f = np.load("{}/{}_predicted_loss_diffs-test-{}.npz".format(train_dir,model_name,[test_idx]))
            predicted_loss_diffs[seed] = f['predicted_loss_diffs']

            print("Ending seed {}, loss_full {}, correctness {}".format(seed, loss_full[seed], correct[seed]))

        predicted_loss_diffs = np.array(predicted_loss_diffs)
        np.savez(path_name, loss_full=loss_full, correct=correct, predicted_loss_diffs=predicted_loss_diffs)

    else:
        f = np.load(path_name)
        loss_full = f['loss_full']
        correct = f['correct']
        predicted_loss_diffs = f['predicted_loss_diffs']
    print("Getting original losses and correctnesses and preds took {} sec".format(time.time() - start_time))
    return loss_full, correct, predicted_loss_diffs

def complete_retrain(loss_remove_one, actual_retrain_loss_diffs, loss_full):
    for seed in seeds:
        print("Starting complete retraining on seed {}".format(seed))
        start_time_seed = time.time()
        for point in top_points:
            print("Starting complete retraining on seed {}, point {}.".format(seed, point))
            tf.reset_default_graph()
            start_time_point = time.time()
            model_name = 'mnist_small_all_cnn_c_hidden2_seed{}_wd{}_damping{}_iter-{}_remove_{}'.format(seed, int(weight_decay*1000), int(damping*100), num_steps, point)
            model = All_CNN_C_Hidden2(
                    input_side=input_side, 
                    input_channels=input_channels,
                    conv_patch_size=conv_patch_size,
                    hidden1_units=hidden1_units, 
                    hidden2_units=hidden2_units,
                    weight_decay=weight_decay,
                    num_classes=num_classes, 
                    batch_size=batch_size,
                    data_sets=data_sets,
                    initial_learning_rate=initial_learning_rate,
                    damping=damping,
                    decay_epochs=decay_epochs,
                    mini_batch=True,
                    train_dir=train_dir, 
                    log_dir='log',
                    model_name=model_name,
                    seed=seed)

            idx = np.array([True] * data_sets.train.x.shape[0], dtype=bool)
            idx[point] = False
            new_train_x = data_sets.train.x[idx, :]
            new_train_y = data_sets.train.labels[idx]
            model.update_train_x_y(new_train_x, new_train_y)

            iter_to_load = num_steps - 1
            if not os.path.exists("{}/{}-checkpoint-{}.meta".format(train_dir, model_name, iter_to_load)):
                model.train(num_steps=num_steps,iter_to_switch_to_batch=10000000,iter_to_switch_to_sgd=10000000)
            else:
                model.load_checkpoint(iter_to_load)

            sess = model.sess
            test_feed_dict = model.fill_feed_dict_with_one_ex(model.data_sets.test, test_idx)
            loss_remove_one[seed][point] = sess.run(model.loss_no_reg, feed_dict=test_feed_dict)
            actual_retrain_loss_diffs[seed][point] = loss_remove_one[seed][point] - loss_full[seed] 

            print("Loss removing {} on seed {}: {}, previously {}".format(point, seed,
                loss_remove_one[seed][point], loss_full[seed]))
            print("Actual loss diff w/ complete retrain: {}, predicted loss diff: {}".format(
                actual_retrain_loss_diffs[seed][point], predicted_loss_diffs[seed][point]))
            print("Actual - predicted loss diff: {}".format(actual_retrain_loss_diffs[seed][point]
                - predicted_loss_diffs[seed][point]))
            print("Seed {} took {} sec for point {}".format(seed, time.time() - start_time_point, point))

        print("Seed {} took {} sec for {} points".format(seed, time.time() - start_time_seed, num_top_points))
    
    loss_remove_one = np.array(loss_remove_one)
    actual_retrain_loss_diffs = np.array(actual_retrain_loss_diffs)
    np.savez("{}/{}_{}-{}seeds_complete_retrain_losses_and_actuals-test-{}-points-{}.npz".format(
            train_dir,model_name,first_seed,first_seed+num_seeds-1,[test_idx],top_points),
        loss_remove_one = loss_remove_one,
        actual_retrain_loss_diffs = actual_retrain_loss_diffs)

    return loss_remove_one, actual_retrain_loss_diffs

loss_full, correct, predicted_loss_diffs = get_originals(loss_full, correct, predicted_loss_diffs, 0, tot_num_seeds)
loss_remove_one, actual_retrain_loss_diffs = complete_retrain(loss_remove_one, actual_retrain_loss_diffs, loss_full)
print(loss_remove_one)
print(actual_retrain_loss_diffs)
