import argparse
import logging
import os
import sys
from argparse import Namespace
from typing import Union, Dict, Optional

import torch
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

from sumo.config import Config


class ExperimentLogger(logging.Logger):
    """
    Logger writing logs to a file.

    The log file is written to `LOG_DIR` and the logger is registered under the name `experiments`.

    Parameters
    ----------
    config : Config
        Config containing the parameters from the configuration yaml file
    """

    def __init__(self, config: Config) -> None:
        super(ExperimentLogger, self).__init__('experiments', logging.INFO)

        self.config = config

        # create the logging file handler
        log_file = os.path.join(config.output_dir, f'{config.run_name}.log')
        fh = logging.FileHandler(log_file)

        # format
        formatter = logging.Formatter('[%(asctime)s] {%(module)s:%(lineno)d} %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        # add handlers to logger object
        self.addHandler(fh)

    def initial_message(self, args: Namespace) -> None:
        """
        Log necessary information about the run.

        Info logged are e.g. the script name, the arguments given to the script and the config specified by the chosen
        experiment.

        Parameters
        ----------
        args: Namespace
            The arguments given to the script using this logger.
        """

        arguments = sys.argv.copy()
        self.info(arguments.pop(0))
        self.info('=' * 80)
        self.info(f'Argument List: {arguments}')
        self.info('Arguments:')
        [self.info(f'{x:30s}: {y}') for x, y in args.__dict__.items()]
        self.info('=' * 80)
        self.info('Configuration:')
        [self.info(f'{x:30s}: {y}') for x, y in self.config.__dict__.items()]
        self.info('=' * 80)

    def log_metrics(self, metrics_dict: Dict[str, float]) -> None:
        """
        Logs the given losses and metrics, but only if they are computed on the validation or test data.

        Parameters
        ----------
        metrics_dict
            a dict containing the computed losses and metrics
        """

        if ('epoch' not in metrics_dict) or ('loss/train' in metrics_dict):
            return

        def _handle_value(value):
            if isinstance(value, torch.Tensor):
                return value.item()
            return value

        self.info(f'validation metrics at epoch {metrics_dict["epoch"]}:')
        [self.info(f'{x:30s}: {_handle_value(y)}') for x, y in metrics_dict.items() if x != 'epoch']


class FileLogger(LightningLoggerBase):
    """
    Logger for Pytorch Lightning using the ExperimentLogger to log losses and metrics to a log file.

    Parameters
    ----------
    logger : ExperimentLogger
        ExperimentLogger to be used by this Pytorch Lightning logger
    """

    def __init__(self, logger: ExperimentLogger) -> None:
        super(FileLogger, self).__init__()

        self._experiment = logger

    @property
    def name(self) -> str:
        return 'experiment_logger'

    @property
    @rank_zero_experiment
    def experiment(self) -> ExperimentLogger:
        return self._experiment

    @property
    def version(self) -> Union[int, str]:
        return 0

    @rank_zero_only
    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs) -> None:
        pass  # hyperparameters are already logged in ExperimentLogger.initial_message

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        self.experiment.log_metrics(metrics)
