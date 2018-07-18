from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os, warnings
import cPickle as pickle

valid_datasets = ['mnist', 'mnist_small', 'cifar10', 'cifar10_small']
valid_models = ['all_cnn_c_hidden', 'logreg_lbfgs']

def load_config(model_name):
    return pickle.load(open('configs/config_dict_{}.pickle'.format(model_name), 'rb'))

def get_model_name(nametag, dataset_type, model_type, seed, num_units=None, num_steps=None):
    assert dataset_type in valid_datasets
    assert model_type in valid_models

    if model_type == 'all_cnn_c_hidden':
        assert num_units is not None and num_steps is not None
        return '{}_{}_{}{}_seed{}_iter-{}'.format(nametag, dataset_type, model_type, num_units, seed, num_steps)
    elif model_type == 'logreg_lbfgs':
        return '{}_{}_{}_seed{}'.format(nametag, dataset_type, model_type, seed)

def make_config(seed, dataset_type, model_type, out, num_steps=300000, nametag='default', save=True, test_idx=6558):

    assert dataset_type in valid_datasets
    assert model_type in valid_models

    if model_type == 'all_cnn_c_hidden':
        hidden_units = [8,8]#[8,8,8]#[8,8]
        weight_decay = 0.001#0.01#0.001
        damping = 2e-2#2e-3#2e-2
        decay_epochs = [5000,10000]#[500,1000,2500,5000,7500]#[5000,10000]
        batch_size = 500
        initial_learning_rate = 0.0001#0.01#0.0001
        num_steps = num_steps
        model_name = get_model_name(nametag=nametag, dataset_type=dataset_type, model_type=model_type, num_units=len(hidden_units), seed=seed, num_steps=num_steps)
    elif model_type == 'logreg_lbfgs':
        weight_decay = 0.01
        damping = 0.0
        decay_epochs = [1000,10000]
        batch_size = 1400
        initial_learning_rate = 0.001
        model_name = get_model_name(nametag=nametag, dataset_type=dataset_type, model_type=model_type, seed=seed)

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
            'train_dir':            out,
            'log_dir':              'log',
            'num_classes':          10,
            'lissa_params':         {'batch_size':None,'scale':10,'damping':0.0,'num_samples':1,'recursion_depth':5000},
            'fmin_ncg_params':      {'avextol':1e-8,'maxiter':100},
            'test_grad_batch_size': 100
            }

    #model-specific
    if model_type == 'all_cnn_c_hidden':
        spec_dict = {
                'input_side':       28,
                'input_channels':   1,
                'conv_patch_size':  3,
                'hidden_units':     hidden_units,
                'weight_decay':     weight_decay
                }
    elif model_type == 'logreg_lbfgs':
        spec_dict = {
                'input_dim':        28*28,
                'weight_decay':     weight_decay,
                'max_lbfgs_iter':   100
                }
    else:
        warnings.warn("Invalid model type")

    config_dict = {'gen': gen_dict, 'spec': spec_dict}

    if save:
        pickle_out = open('configs/config_dict_{}.pickle'.format(model_name), 'wb')
        pickle.dump(config_dict, pickle_out)
        pickle_out.close()

    return config_dict
