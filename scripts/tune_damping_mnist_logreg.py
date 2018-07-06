from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython
import cPickle as pickle

import tensorflow as tf
import time
import os.path

import influence.experiments as experiments
from influence.all_CNN_c import All_CNN_C

import argparse

def load_config(model_name):
    pickle_in = open('configs/config_dict_{}.pickle'.format(model_name), 'rb')
    return pickle.load(pickle_in)

#parser = argparse.ArgumentParser(description='Input seed and point.')
#parser.add_argument('seed', type=int)
#parser.add_argument('point', type=int)
#args=parser.parse_args()

#point = args.point
#seed = args.seed
indices_to_remove = [1173,4644,1891,4936,1735,3562]
seed = 0
force_refresh=True
train_dir = '../output'
regpathname = '{}/special_seed{}_losses_logreg'.format(train_dir,seed)
test_idx = 6558
num_steps = 300000 ########

print("Starting seed{}".format(seed))
tf.reset_default_graph()
model_name = 'special_mnist_small_all_cnn_c_hidden0_seed{}_iter-{}'.format(seed,num_steps)
model = All_CNN_C(load_config(model_name))

print('Initial learning rate {}, decay epochs {}'.format(model.initial_learning_rate,model.decay_epochs))

# Training
if os.path.exists('{}.npz'.format(regpathname)):# and not force_refresh:
    f = np.load('{}.npz'.format(regpathname))
    losses = f['losses']
    losses_fine = f['losses_fine'] 
else:
    model.train(num_steps=num_steps,
        iter_to_switch_to_batch=10000000,
        iter_to_switch_to_sgd=10000000)
    losses,losses_fine = model.get_all_losses()
np.savez(regpathname, losses=losses, losses_fine=losses_fine)

# Warm-start retraining -- actual is actually warm
warm_loss_diffs = experiments.test_only_retraining(
        model,
        num_to_remove=len(indices_to_remove),
        test_idx=test_idx,
        iter_to_load=num_steps-1,
        num_steps=600000,
        remove_type='manual',
        force_refresh=True,
        random_seed=None,
        indices_to_remove=indices_to_remove,
        do_sanity_checks=True
        )
model.load_checkpoint(num_steps-1,True)
np.savez('{}/special_seed{}_warm_pred_infl_logreg_remove_{}'.format(train_dir,seed,indices_to_remove),
        warm_infl=warm_loss_diffs,
        pred_infl=model.get_influence_on_test_loss([test_idx],
            np.arange(len(model.data_sets.train.labels)),
            force_refresh=False,
            batch_size='default')
        )
print(losses)
print(losses_fine)
"""
# Complete retraining
for point in indices_to_remove:
    rempathname = '{}/special_seed{}_remove{}_only_losses_logreg'.format(train_dir,seed,point)
    print("Starting seed{} removing {}".format(seed,point))
    tf.reset_default_graph()
    model_name = 'special_mnist_small_all_cnn_c_hidden0_seed{}_iter-{}_remove_{}'.format(seed,num_steps,point)
    model = All_CNN_C(load_config(model_name))
    
    print('Initial learning rate {}, decay epochs {}'.format(model.initial_learning_rate,model.decay_epochs))

    model.data_sets.train._omits[point] = True ### Forgot to put this in the configs; needs omits
    if os.path.exists('{}.npz'.format(rempathname)) and not force_refresh:
        f = np.load('{}.npz'.format(rempathname))
        losses = f['losses_removed']
        losses_removed_fine = f['losses_removed_fine'] 
    else:
        model.train(num_steps=num_steps,
            iter_to_switch_to_batch=10000000,
            iter_to_switch_to_sgd=10000000,
            track_losses=True)
        losses_removed,losses_removed_fine = model.get_all_losses()
    np.savez(rempathname,losses_removed=losses_removed,losses_removed_fine=losses_removed_fine)
    print(losses_removed)
    print(losses_removed_fine)
"""
