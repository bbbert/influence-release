import numpy as np
import os, sys
import yaml

from load_mnist import load_mnist, load_small_mnist
from influence.logger import Logger
from influence.all_CNN_c import All_CNN_C

BASE_OUTPUT_DIR = os.environ("INFLUENCE_OUTPUT_PATH")

class Experiment(object):
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.load(f.read())
        self.logger = None

    # Each run should create its own logger
    def run(self):
        raise NotImplementedError()

    def log(self, message):
        assert self.logger is not None
        self.logger.log(message)

class CNNExperiment(Experiment):
    def __init__(self, config_path):
        super(self).__init__(config_path)

    @property
    def get_path(self):
        return os.path.join(BASE_OUTPUT_DIR, 'cnn')

    @override
    def run(self):
        self.logger = Logger(self.get_path, self.config['name'])
        self.load_base_dataset()
        self.initialize_model()
        self.do_initial_training()
        self.predict_influence()
        self.logger = None

    def load_base_dataset(self):
        if self.config['dataset']['name'] == "mnist_small":
            self.base_dataset = load_small_mnist('data')
            self.log("Loaded dataset {}".format(self.config['dataset']['name']))
        else:
            message = "Failed to load unknown dataset {}".format(self.config['dataset']['name'])
            self.log(message)
            raise ValueError(message)

    def initialize_model(self):
        model_cfg = self.config['model']
        if model_cfg['type'] == 'all_cnn_c_hidden':
            model_cfg['dataset'] = self.config['dataset']
            self.model = All_CNN_C(model_cfg)
            self.log("Initialized model of type {}".format(model_cfg['type']))
        else:
            message = "Failed to initialize model due to unknown type {}".format(model_cfg['type'])
            self.log(message)
            raise ValueError(message)

    def do_initial_training(self):
        seeds = range(self.config['seeds']['count'])

        checkpt_interval = self.config['training']['checkpoint_interval']
        min_epochs = self.config['training']['min_epochs']
        max_epochs = self.config['training']['max_epochs']
        convergence_tol = self.config['training']['convergence_tol']

        self.log("Beginning initial training with seeds {}, ".format(seeds)
                + "checkpt_interval {}, min_epochs {}, ".format(checkpt_interval, min_epochs)
                + "max_epochs {}, convergence_tol {}.".format(max_epochs, convergence_tol))

        test = self.base_dataset.test

        for seed in seeds:
            self.log("Beginning seed {}".format(seed))

            new_train = self.base_dataset.train.clone()
            new_train.init_state(seed)

            seed_path = os.path.join(self.get_path, "seed_{}".format(seed))
            initial_training_path = os.path.join(seed_path, "initial_training")

            output = ModelOutput(initial_training_path)
            if output.converged:
                self.log("Skipped seed {}".format(seed))
                continue

            train_loss_history = []
            test_loss_history = []
            self.model.reset_state(seed)

            checkpt_epoch = output.last_checkpt_epoch
            if checkpt_epoch is not None:
                output.load_checkpt(checkpt_epoch, self.model, new_train)
                data = output.load_history()
                train_loss_history = data['train_loss_history']
                test_loss_history = data['test_loss_history']
                self.log("Loaded checkpoint before epoch {}".format(checkpt_epoch))
            else:
                # We want to save model before epoch 0
                output.save_checkpt(self.model.epoch, self.model, new_train)

            converged = False
            while self.model.epoch < max_epochs or (converged and self.model.epoch < min_epochs):
                train_losses, time_for_epoch, converged = self.model.train_one_epoch(new_train)
                test_losses = self.model.get_losses(test)
                train_loss_history.extend(train_losses)
                test_loss_history.extend(test_losses)
                
                output.log("Epoch {} ({} ms): avg train loss {}, avg test loss {}".format(
                    self.model.epoch-1, time_for_epoch, np.mean(train_losses), np.mean(test_losses))

                if self.model.epoch % checkpt_interval == 0:
                    output.save_checkpt(self.model.epoch, self.model, new_train)
                    output.save_history(train_loss_history, test_loss_history)
                    self.log("Completed {} epochs".format(self.model.epoch))

            output.save_checkpt(self.model.epoch, self.model, new_train)
            output.save_history(train_loss_history, test_loss_history, converged=converged)
            self.log("Completed training for seed {} after {} epochs, convergence={}".format(
                seed, self.model.epoch, converged)
