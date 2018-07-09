from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os, warnings
import cPickle as pickle

test_idx = 6558
seed = 0
#model = 'logreg_lbfgs'
model = 'all_cnn_c_hidden'
dataset_type = 'mnist_small'

assert dataset_type in ['mnist', 'mnist_small']
assert model in ['all_cnn_c_hidden', 'logreg_lbfgs']

if model == 'all_cnn_c_hidden':
    hidden_units = [8,8]#[]
    weight_decay = 0.001#0.01#0.001
    damping = 0#2e-2
    decay_epochs = [3000,18000,32000]#[1000,3000,5000]#[500,1000,3000]#[5000,10000]
    batch_size = 500
    initial_learning_rate = 0.001#0.0001
    num_steps = 550000 #300000
    model_name = 'relaxed_{}_{}{}_seed{}_iter-{}_remove_3562'.format(dataset_type, model, len(hidden_units), seed, num_steps)
elif model == 'logreg_lbfgs':
    weight_decay = 0.01
    damping = 0.0
    decay_epochs = [1000,10000]
    batch_size = 1400
    initial_learning_rate = 0.001
    model_name = 'special_{}_{}_seed{}'.format(dataset_type, model, seed)

#genericNN
gen_dict = {
        'model_name':           model_name,
        'test_point':           test_idx,
        'batching_seed':        seed,
        'initialization_seed':  seed,
        'dataset_type':         dataset_type,
        'batch_size':           batch_size,
        'initial_learning_rate':initial_learning_rate,
        'decay_epochs':         decay_epochs,
        'damping':              damping,
        'keep_probs':           None,
        'mini_batch':           True,
        'train_dir':            '../unitTest-output',
        'log_dir':              'log',
        'num_classes':          10,
        'lissa_params':         {'batch_size':None,'scale':10,'damping':0.0,'num_samples':1,'recursion_depth':5000},
        'fmin_ncg_params':      {'avextol':1e-8,'maxiter':100},
        'test_grad_batch_size': 100
        }

#model-specific
if model == 'all_cnn_c_hidden':
    spec_dict = {
            'input_side':       28,
            'input_channels':   1,
            'conv_patch_size':  3,
            'hidden_units':     hidden_units,
            'weight_decay':     weight_decay
            }
elif model == 'logreg_lbfgs':
    spec_dict = {
            'input_dim':        28*28,
            'weight_decay':     weight_decay,
            'max_lbfgs_iter':   100
            }
else:
    warnings.warn("Invalid model type")

config_dict = {'gen': gen_dict, 'spec': spec_dict}

pickle_out = open('configs/config_dict_{}.pickle'.format(model_name), 'wb')
pickle.dump(config_dict, pickle_out)
pickle_out.close()
