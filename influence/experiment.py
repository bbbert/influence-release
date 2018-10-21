import numpy as np
import os, sys
import yaml

from load_mnist import load_mnist, load_small_mnist

BASE_OUTPUT_DIR = os.environ("INFLUENCE_OUTPUT_PATH")

class Experiment(object):
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.load(f.read())

    # Running the experiment should be an idempotent operation on the output path
    def run(self):
        raise NotImplementedError()

class CNNExperiment(Experiment):
    def __init__(self, config_path):
        super(self).__init__(config_path)

    def get_path(self):
        return os.path.join(BASE_OUTPUT_DIR, 'cnn')

    @override
    def run(self):
        self.load_base_dataset()
        self.initialize_model()
        self.do_initial_training()

    def load_base_dataset(self):
        if self.config['dataset'] == "mnist_small":
            self.base_dataset = load_small_mnist('data')
        else:
            raise ValueError("Unknown dataset {}".format(self.config['dataset']))

    def initialize_model(self):
        model_cfg = self.config['model']
        if model_cfg['type'] == 'all_cnn_c_hidden':
            # TODO: initialize self.model with its training schedule too
            pass
        else:
            raise ValueError("Unknown model type".format(model_cfg['type']))

    def do_initial_training(self):
        seeds = range(self.config['seeds']['count'])

        checkpt_interval = self.config['training']['checkpoint_interval']
        min_epochs = self.config['training']['min_epochs']
        max_epochs = self.config['training']['max_epochs']
        convergence_tol = self.config['training']['convergence_tol']

        for seed in seeds:
            new_train = self.base_dataset.train.clone()
            new_train.init_state(seed)

            seed_path = os.path.join(self.get_path(), "seed_{}".format(seed))
            initial_training_path = os.path.join(seed_path, "initial_training")

            output = ModelOutput(initial_training_path)
            if output.converged:
                continue

            train_loss_history = []
            self.model.reset_state(seed)

            checkpt_epoch = output.last_checkpt_epoch
            if checkpt_epoch is not None:
                output.load_checkpt(checkpt_epoch, self.model, new_train)
                train_loss_history = output.load_history()

            while self.model.epoch < max_epochs:
                # TODO: figure out what to save
                train_losses = self.model.train_one_epoch(new_train)
                train_loss_history.extend(train_losses)

                if self.model.epoch % checkpt_interval == 0:
                    output.save_checkpt(self.model.epoch, self.model, new_train)
                    output.save_history(train_loss_history)

            output.save_checkpt(self.model.epoch, self.model, new_train)
            output.save_history(train_loss_history, converged=True)

