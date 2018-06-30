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
from influence.all_CNN_c_hidden0 import All_CNN_C_Hidden0

from load_mnist import load_small_mnist, load_mnist

import argparse

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

parser = argparse.ArgumentParser(description='Input seed and point.')
parser.add_argument('seed', type=int)
parser.add_argument('point', type=int)
args=parser.parse_args()

point = args.point
seed = args.seed
num_points = 5500
saving_pred = True
mode='logreg'

force_refresh=True

train_dir = '../scr/output'
oldpathname = '{}/seed{}_remove{}_losses'.format(train_dir,seed,point)
regpathname = '{}/seed{}_losses'.format(train_dir,seed)
finepathname = '{}/seed{}_remove{}_only_losses'.format(train_dir,seed,point)
test_idx = 6558

if mode == 'logreg':
    oldpathname += '_logreg'
    regpathname += '_logreg'
    finepathname += '_logreg'

num_steps = 300000 ########
damping = 2e-2

print("Starting seed{}".format(seed))
tf.reset_default_graph()
model_name = 'mnist_small_all_cnn_c_logreg_seed{}_iter-{}'.format(seed,num_steps)
for dataset in data_sets:
    if dataset is not None:
        dataset.set_randomState_and_reset_rngs(seed)
data_sets.train.set_omits(np.zeros(len(data_sets.train.labels),dtype=bool))
model = All_CNN_C_Hidden0(
                    input_side=input_side, 
                    input_channels=input_channels,
                    conv_patch_size=conv_patch_size,
                    #hidden1_units=hidden1_units, 
                    #hidden2_units=hidden2_units,
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
                    seed=seed,
                    test_point=test_idx)
if os.path.exists('{}.npz'.format(oldpathname)):# and not force_refresh:
    f = np.load('{}.npz'.format(oldpathname))
    losses = f['losses']
    losses_fine = f['losses_fine']
elif os.path.exists('{}.npz'.format(regpathname)):# and not force_refresh:
    f = np.load('{}.npz'.format(regpathname))
    losses = f['losses']
    losses_fine = f['losses_fine'] 
else:
    model.train(num_steps=num_steps,
        iter_to_switch_to_batch=10000000,
        iter_to_switch_to_sgd=10000000,
        track_losses=True)
    losses,losses_fine = model.get_all_losses()
model.load_checkpoint(num_steps-1)
np.savez('{}/seed{}_pred_infl_{}'.format(train_dir,seed,mode),
        pred_infl=model.get_influence_on_test_loss([test_idx],
            np.arange(len(model.data_sets.train.labels)),
            force_refresh=False)
        )
np.savez(regpathname, losses=losses, losses_fine=losses_fine)
print(losses)
print(losses_fine)

print("Starting seed{} removing {}".format(seed,point))
tf.reset_default_graph()
model_name = 'mnist_small_all_cnn_c_logreg_seed{}_iter-{}_remove_{}'.format(seed,num_steps,point)
for dataset in data_sets:
    if dataset is not None:
        dataset.set_randomState_and_reset_rngs(seed)
omits = np.zeros(len(data_sets.train.labels), dtype=bool)
omits[point] = True
data_sets.train.set_omits(omits)
model = All_CNN_C_Hidden0(
                    input_side=input_side, 
                    input_channels=input_channels,
                    conv_patch_size=conv_patch_size,
                    #hidden1_units=hidden1_units, 
                    #hidden2_units=hidden2_units,
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
                    seed=seed,
                    test_point=test_idx)
if os.path.exists('{}.npz'.format(oldpathname)) and not force_refresh:
    f = np.load('{}.npz'.format(oldpathname))
    losses_removed= f['losses_removed']
    losses_removed_fine = f['losses_removed_fine']
elif os.path.exists('{}.npz'.format(finepathname)) and not force_refresh:
    f = np.load('{}.npz'.format(finepathname))
    losses = f['losses_removed']
    losses_removed_fine = f['losses_removed_fine'] 
else:
    model.train(num_steps=num_steps,
        iter_to_switch_to_batch=10000000,
        iter_to_switch_to_sgd=10000000,
        track_losses=True)
    losses_removed,losses_removed_fine = model.get_all_losses()
np.savez(finepathname,losses_removed=losses_removed,losses_removed_fine=losses_removed_fine)
print(losses_removed)
print(losses_removed_fine)
