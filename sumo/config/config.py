import os
import pickle
from datetime import datetime
from importlib import import_module
from pathlib import Path

import numpy as np
import torch.nn as nn
import torch.optim as optim
import yaml


class Config:
    """
    Class storing configuration of one experiment run.

    Parameters
    ----------
    experiment : str
        Name of yaml configuration file for the experiment to run.
    create_dirs: bool, optional
        When True, create the output directories used for this run.
    """

    def __init__(self, experiment: str, create_dirs=True):
        self.experiment = experiment

        base_dir = Path(__file__).absolute().parents[2]
        with open(base_dir / 'config' / 'default.yaml', 'r') as conf:
            config = yaml.safe_load(conf)

        with open(base_dir / 'config' / f'{experiment}.yaml', 'r') as conf:
            custom_config = yaml.safe_load(conf)
        self.merge_dicts(config, custom_config)

        # configuration of the used input data
        config_data = config['data']
        # the data might be None, in case predictions should be run on data in a custom format
        if config_data:
            self.data_dir = Path(config_data['directory'])
            self.data_file = self.data_dir / config_data['file_name']

            with open(self.data_file, 'rb') as data_file:
                subjects = pickle.load(data_file)

            self.train_split_name = config_data['split']['train']
            self.val_split_name = config_data['split']['validation']
            self.cross_validation = config_data['split']['cross_validation']
            assert (self.train_split_name is None) == (self.val_split_name is None), \
                'either train and validation splits are given or none of them when using cross validation'
            assert sum([self.val_split_name is not None, self.cross_validation is not None]) == 1, \
                'either train and validation splits are given or none of them when using cross validation'

            if self.train_split_name is not None:
                assert self.train_split_name in subjects, 'data file does not contain given train split'
                assert self.val_split_name in subjects, 'data file does not contain given validation split'
            else:
                assert type(self.cross_validation) in [int, list]
                if type(self.cross_validation) is int:
                    assert self.cross_validation >= 2, 'there have to be at least two cv folds'  # type: ignore
                    # use fold_0, fold_1, ... as default names for cross validation folds if none are explicitly given
                    self.cross_validation = [f'fold_{i}' for i in range(self.cross_validation)]  # type: ignore
                for fold in self.cross_validation:
                    assert fold in subjects, f'data file does not contain given cross validation split {fold}'

            self.test_split_name = config_data['split']['test']
            if self.test_split_name is not None:
                assert self.test_split_name in subjects, 'data file does not contain given test split'

            self.batch_size = config_data['batch_size']
            assert type(self.batch_size) is int and self.batch_size >= 1, 'the batch size has to be at least one'
            self.preprocessing = config_data['preprocessing']
            assert type(self.preprocessing) in [bool, type(None)]

        # configuration of the experiment itself
        config_experiment = config['experiment']

        # configuration of the used model
        config_model = config_experiment['model']
        self.n_classes = config_model['n_classes']
        assert type(self.n_classes) is int and self.n_classes >= 1, \
            'at least one class has to be predicted by the model'
        self.activation = getattr(nn, config_model['activation'])
        self.depth = config_model['depth']
        assert type(self.depth) is int and self.depth >= 1, 'the depth of the model has to be at least one'
        self.channel_size = config_model['channel_size']
        assert type(self.channel_size) is int and self.channel_size >= 1, \
            'the number of channels in the first layer has to be at least one'
        self.pools = config_model['pools']
        assert type(self.pools) is list
        assert len(self.pools) == self.depth, 'the number of given pooling sizes does not match the specified depth'
        for p in self.pools:
            assert type(p) is int and p >= 1, f'the given pooling size {p} is invalid'
        self.convolution_params = config_model['convolution_params']
        assert type(self.convolution_params) in [dict, type(None)]
        if self.convolution_params is None:
            self.convolution_params = {}
        self.moving_avg_size = config_model['moving_avg_size']
        assert type(self.moving_avg_size) is int and self.moving_avg_size >= 1, \
            'the size of the average pooling kernel has to be at least one'

        # configuration of the training process
        config_train = config_experiment['train']
        self.n_epochs = config_train['n_epochs']
        assert type(self.n_epochs) is int and self.n_epochs >= 1, 'the number of training epochs has to be at least one'
        self.early_stopping = config_train['early_stopping']
        assert type(self.early_stopping) in [int, type(None)]
        if type(self.early_stopping) is int:
            assert self.early_stopping >= 1, 'the number of epochs used for early stopping has to be at least one'

        # configuration of the used optimizer
        config_optimizer = config_train['optimizer']
        self.optimizer = getattr(optim, config_optimizer['class_name'])
        assert issubclass(self.optimizer, optim.Optimizer), 'optimizer must be a subtype of torch.optim.Optimizer'
        self.optimizer_params = config_optimizer['params']
        assert type(self.optimizer_params) in [dict, type(None)]
        if self.optimizer_params is None:
            self.optimizer_params = {}

        # (optional) configuration of the used learning rate scheduler
        config_lr_scheduler = config_train['lr_scheduler']
        assert type(config_lr_scheduler) in [dict, type(None)]
        if config_lr_scheduler is None:
            self.lr_scheduler = None
            self.lr_scheduler_params = {}
        else:
            self.lr_scheduler = getattr(optim.lr_scheduler, config_lr_scheduler['class_name'])
            self.lr_scheduler_params = config_lr_scheduler['params']
            assert type(self.lr_scheduler_params) in [dict, type(None)]
            if self.lr_scheduler_params is None:
                self.lr_scheduler_params = {}

        # configuration of the used loss function
        config_loss = config_train['loss']
        try:
            losses = import_module('sumo.loss')
            self.loss = getattr(losses, config_loss['class_name'])  # first try to import a custom loss function
        except AttributeError:
            self.loss = getattr(nn, config_loss['class_name'])  # if none (matching) exists import a pytorch function
        self.loss_params = config_loss['params']
        assert type(self.loss_params) in [dict, type(None)]
        if self.loss_params is None:
            self.loss_params = {}

        # configuration of the validation process
        config_val = config_experiment['validation']
        overlap_threshold_step = config_val['overlap_threshold_step']
        assert type(overlap_threshold_step) is float and 0.0 < overlap_threshold_step <= 1.0, \
            'the step size of the used overlap thresholds has to be greater than zero and not more than one'
        self.overlap_thresholds, step = np.linspace(0., 1., int(1 / overlap_threshold_step) + 1, retstep=True)
        assert step == overlap_threshold_step, \
            'the interval of overlap thresholds from zero to one cannot be evenly spaced by the given step size'

        # unique identifier for this run used for directory and log file names
        self.run_name = f'{self.experiment}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        self.output_dir = base_dir / 'output' / self.run_name

        if create_dirs:
            self.create_dirs()

    def create_dirs(self):
        """
        Create output directories of this experiment run.
        """
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def merge_dicts(default_dict: dict, custom_dict: dict):
        """
        Method to merge two dictionaries representing the content of a configuration file.

        `default_dict` contains the parameters of the default configuration, which should be overwritten by values
        specified in `custom_dict`, which represents the content of an experiment configuration.

        Parameters
        ----------
        default_dict: dict
            The default configuration, which at the end of the method contains the merged values.
        custom_dict: dict
            The custom experiment configuration to be merged into `default_dict`.

        Returns
        -------
        default_dict: dict
            The (in-place) merged dictionary.
        """

        assert default_dict is not None, 'default configuration can not be empty'
        if custom_dict is None:
            return default_dict

        for k, v in custom_dict.items():
            if isinstance(v, dict):
                if default_dict.get(k) is None:
                    default_dict[k] = {}
                default_dict[k] = Config.merge_dicts(default_dict[k], v)
            else:
                default_dict[k] = v
        return default_dict
