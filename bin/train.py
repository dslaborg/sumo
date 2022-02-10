import argparse
from pathlib import Path
from sys import path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# append root dir to python path so that we find `sumo`
path.insert(0, str(Path(__file__).absolute().parents[1]))
from sumo.config import Config
from sumo.data import MODADataModule, MODADataModuleCV
from sumo.log import FileLogger, ExperimentLogger
from sumo.model import SUMO


def get_args():
    parser = argparse.ArgumentParser(description='Train UTime on EEG data and gold standard spindles')
    parser.add_argument('-e', '--experiment', type=str, required=True,
                        help='Name of configuration yaml file to use for this run')

    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1, num_sanity_val_steps=0)

    return parser.parse_args()


def get_callbacks(output_path):
    checkpoint_dir = Path(output_path) / 'models'
    checkpoint_fn = 'epoch={epoch:04d}-val_f1_mean={metrics/val_f1_mean:.3f}-val_loss={loss/val:.3f}'
    checkpoint = ModelCheckpoint(checkpoint_dir, filename=checkpoint_fn, auto_insert_metric_name=False,
                                 monitor='metrics/val_f1_mean', save_weights_only=True, save_last=True, save_top_k=3,
                                 mode='max')
    return [checkpoint]


def main(datamodule, logger):
    model = SUMO(config)
    file_logger = FileLogger(logger)
    tb_logger = TensorBoardLogger(config.output_dir, name='tensorboard', version='.')
    trainer = pl.Trainer.from_argparse_args(args, callbacks=get_callbacks(config.output_dir),
                                            logger=[file_logger, tb_logger], max_epochs=config.n_epochs,
                                            log_every_n_steps=45)
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    args = get_args()

    config = Config(args.experiment)

    if config.cross_validation is None:
        logger = ExperimentLogger(config)
        logger.initial_message(args)

        datamodule = MODADataModule(config)
        main(datamodule, logger)
    else:
        output_dir = config.output_dir

        for fold in range(len(config.cross_validation)):
            config.output_dir = Path(output_dir) / f'fold_{fold:02d}'
            config.create_dirs()

            logger = ExperimentLogger(config)
            logger.initial_message(args)

            datamodule = MODADataModuleCV(config, fold)
            main(datamodule, logger)
