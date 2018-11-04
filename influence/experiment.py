import numpy as np
import os, sys
import yaml
import logging
import datetime

from load_mnist import load_mnist, load_small_mnist
from influence.all_CNN_c import All_CNN_C
from influence.output import ModelOutput

DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output"))
BASE_OUTPUT_DIR = os.environ.get("INFLUENCE_OUTPUT_PATH", DEFAULT_OUTPUT_DIR)

class Experiment(object):
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.load(f.read())
        self.logger = None

    # Each run should create its own logger
    def run(self):
        raise NotImplementedError()

    def log(self, *args, **kwargs):
        assert self.logger is not None
        self.logger.info(*args, **kwargs)

class CNNExperiment(Experiment):
    @property
    def base_path(self):
        return os.path.join(BASE_OUTPUT_DIR, 'cnn')

    def __init__(self, config_path):
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        super(CNNExperiment, self).__init__(config_path)
        self.setup_logging()

    def setup_logging(self):
        now = datetime.datetime.now()
        log_name = "{}_{:%y%m%d-%H%M%S}.log".format(self.config['name'], now)
        self.log_path = os.path.join(self.base_path, log_name)
        logging.basicConfig(filename=self.log_path, level=logging.DEBUG)
        self.logger = logging.getLogger()
        self.logger.addHandler(logging.StreamHandler())

    def run(self):
        self.log("Starting CNN experiment with configuration:")
        self.log("{}".format(self.config))
        self.load_base_dataset()
        self.initialize_model()
        self.do_initial_training()
        # self.predict_influence()
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
        convergence_tol = self.config['training']['convergence_tolerance']

        self.log("Beginning initial training with seeds {}, ".format(seeds)
                + "checkpt_interval {}, min_epochs {}, ".format(checkpt_interval, min_epochs)
                + "max_epochs {}, convergence_tol {}.".format(max_epochs, convergence_tol))

        test = self.base_dataset.test

        for seed in seeds:
            self.log("Beginning seed {}".format(seed))

            new_train = self.base_dataset.train.clone()
            new_train.init_state(seed)

            seed_path = os.path.join(self.base_path, "seed_{}".format(seed))
            initial_training_path = os.path.join(seed_path, "initial_training")

            output = ModelOutput(initial_training_path)
            if output.converged:
                self.log("Skipped initial training for seed {}".format(seed))
                continue

            train_loss_history = []
            test_loss_history = []
            self.model.reset_state(seed)

            checkpt_epoch = output.last_checkpt_epoch
            if checkpt_epoch is not None:
                output.load_checkpt(checkpt_epoch, self.model, new_train)
                data = output.load_history()
                train_loss_history = list(data['train_loss_history'])
                test_loss_history = list(data['test_loss_history'])
                self.log("Loaded checkpoint before epoch {}".format(checkpt_epoch))
            else:
                # We want to save model before epoch 0
                output.save_checkpt(self.model.epoch, self.model, new_train)
                output.save_history([], [])

            converged = False
            while self.model.epoch < min_epochs or (not converged and self.model.epoch < max_epochs):
                train_losses, time_for_epoch, converged = self.model.train_one_epoch(new_train)
                test_losses = self.model.get_indiv_losses(test)
                train_loss_history.extend(train_losses)
                test_loss_history.append(test_losses)

                output.log("Epoch {} ({} ms): train loss {}, avg test loss {}".format(
                    self.model.epoch-1, time_for_epoch, train_losses[-1], np.mean(test_losses)))

                if self.model.epoch % checkpt_interval == 0:
                    output.save_checkpt(self.model.epoch, self.model, new_train)
                    output.save_history(train_loss_history, test_loss_history)
                    self.log("Completed {} epochs".format(self.model.epoch))

            output.save_checkpt(self.model.epoch, self.model, new_train)
            output.save_history(train_loss_history, test_loss_history, converged=converged)
            self.log("Completed training for seed {} after {} epochs, convergence={}".format(
                seed, self.model.epoch, converged))
