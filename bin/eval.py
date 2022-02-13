import argparse
import re
from pathlib import Path
from sys import path
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch

# append root dir to python path so that we find `sumo`
path.insert(0, str(Path(__file__).absolute().parents[1]))
from sumo.config import Config
from sumo.data import MODADataModule, MODADataModuleCV
from sumo.evaluation import calculate_metrics, calculate_test_metrics, plot_metrics
from sumo.model import SUMO


def custom_evaluation(datamodule, model, plot=False):
    datamodule.prepare_data()
    overlap_thresholds = config.overlap_thresholds

    if args.test:
        precisions, recalls, f1s = calculate_test_metrics(model, datamodule.test_dataloader(), overlap_thresholds)

        if plot:
            for precision, recall, f1 in zip(precisions, recalls, f1s):
                plot_metrics(precision, recall, f1, overlap_thresholds)

        return precisions, recalls, f1s
    else:
        precision, recall, f1 = calculate_metrics(model, datamodule.val_dataloader(), overlap_thresholds)

        if plot:
            plot_metrics(precision, recall, f1, overlap_thresholds)

        return precision, recall, f1


def get_model(path: Union[str, Path]):
    path = Path(path)

    model_file = path if path.is_file() else get_best_model(path)
    model_checkpoint = torch.load(model_file)

    model = SUMO(config)
    model.load_state_dict(model_checkpoint['state_dict'])

    return model


def get_best_model(experiment_path: Path, sort_by_loss: bool = False):
    models_path = experiment_path / 'models'
    models = list(models_path.glob('epoch=*.ckpt'))

    regex = r'.*val_loss=(0\.[0-9]+).*\.ckpt' if sort_by_loss else r'.*val_f1_mean=(0\.[0-9]+).*\.ckpt'
    regex_results = [re.search(regex, str(m)) for m in models]

    models_score = np.array([float(r.group(1)) for r in regex_results])
    model_idx = np.argmin(models_score) if sort_by_loss else np.argmax(models_score)

    return models[model_idx]


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the best UTime model of a training experiment')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to completed experiment, run with train.py')
    parser.add_argument('-e', '--experiment', type=str, default='default',
                        help='Name of configuration yaml file to use for this run')
    parser.add_argument('-t', '--test', action=argparse.BooleanOptionalAction, default=False,
                        help='Use test data instead of validation data')

    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1, num_sanity_val_steps=0)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    config = Config(args.experiment, create_dirs=False)

    if config.cross_validation is None:
        datamodule = MODADataModule(config)
        model = get_model(args.input)

        precisions, recalls, f1s = custom_evaluation(datamodule, model)
    else:
        results = []
        fold_directories = sorted(Path(args.input).glob('fold_*'))
        for fold, directory in enumerate(fold_directories):
            datamodule = MODADataModuleCV(config, fold)
            model = get_model(directory)

            results.append(custom_evaluation(datamodule, model))

        if args.test:
            # results in format (dataset, metric, fold, overlap_threshold)
            results = np.array(results).transpose((2, 1, 0, 3))

            # calculate results/metrics
            # F1 at an overlap threshold of 20 percent per dataset and per fold
            metric_per_fold = results[:, 2, :, 4]
            print(np.round(metric_per_fold.mean(axis=1), 3), np.round(metric_per_fold.std(axis=1), 3))

            # F1 averaged over the overlap thresholds per dataset and per fold
            metric_per_fold = results[:, 2].mean(axis=-1)
            print(np.round(metric_per_fold.mean(axis=1), 3), np.round(metric_per_fold.std(axis=1), 3))
        else:
            # results in format (metric, fold, overlap_threshold)
            results = np.array(results).transpose((1, 0, 2))

            # calculate results/metrics
            # F1 at an overlap threshold of 20 percent per fold
            metric_per_fold = results[2, :, 4]
            print(np.round(metric_per_fold.mean(), 3), np.round(metric_per_fold.std(), 3))

            # F1 averaged over the overlap thresholds per fold
            metric_per_fold = results[2].mean(axis=1)
            print(np.round(metric_per_fold.mean(), 3), np.round(metric_per_fold.std(), 3))
