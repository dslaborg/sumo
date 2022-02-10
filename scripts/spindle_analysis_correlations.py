from pathlib import Path
from sys import path

import numpy as np
import pytorch_lightning as pl
import torch

from a7.detect_spindles import detect_spindles

# append root dir to python path so that we find `sumo`
path.insert(0, str(Path(__file__).absolute().parents[1]))
from spindle_analysis import get_density, get_duration
from sumo.config import Config
from sumo.data import MODADataModule, spindle_vect_to_indices
from sumo.model import SUMO


def print_correlations(densities_gs, densities_a7, densities, durations_gs, durations_a7, durations):
    corr_density_a7 = np.corrcoef(densities_gs, densities_a7)
    corr_density = np.corrcoef(densities_gs, densities)

    corr_duration_a7 = np.corrcoef(durations_gs, durations_a7)
    corr_duration = np.corrcoef(durations_gs, durations)

    print(f'method\tyoung\tolder\tyoung\tolder')

    def f(m, x1, x2, y1, y2):
        return f'{m:6s}\t{round(x1**2, 2)}\t{round(x2**2, 2)}\t{round(y1**2, 2)}\t{round(y2**2, 2)}\t'
    print(f('A7', corr_density_a7[0, 2], corr_density_a7[1, 3], corr_duration_a7[0, 2], corr_duration_a7[1, 3]))
    print(f('SUMO', corr_density[0, 2], corr_density[1, 3], corr_duration[0, 2], corr_duration[1, 3]))


if __name__ == '__main__':
    experiment = 'final'
    base_dir = Path(__file__).absolute().parents[1]
    checkpoint = base_dir / 'output' / 'final.ckpt'

    config = Config(experiment, create_dirs=False)
    config.batch_size = 3

    datamodule = MODADataModule(config)
    datamodule.prepare_data()

    model_state = torch.load(checkpoint)
    model = SUMO(config)
    model.load_state_dict(model_state['state_dict'])

    sampling_rate = 100
    win_length_sec = 0.3
    win_step_sec = 0.1
    thresholds = np.array([1.25, 1.6, 1.3, 0.69])

    trainer = pl.Trainer(gpus=1, num_sanity_val_steps=0, logger=False)
    predictions_all = trainer.predict(model, datamodule.test_dataloader())

    densities, durations = [], []
    for subjects, predictions in zip(datamodule.subjects_test, predictions_all):
        density, duration = [], []

        for subject, prediction in zip(subjects, predictions):
            spindle_indices_gs = [spindle_vect_to_indices(v) for v in subject.spindles]

            spindle_indices_a7 = [detect_spindles(d, thresholds, win_length_sec, win_step_sec, sampling_rate)[1] for d in subject.data]

            prediction = prediction.detach().cpu().numpy()
            spindle_indices = [spindle_vect_to_indices(v) for v in prediction]

            density.append(get_density(spindle_indices_gs, spindle_indices_a7, spindle_indices))
            duration.append(get_duration(spindle_indices_gs, spindle_indices_a7, spindle_indices))

        density = np.array(density).T
        duration = np.array(duration).T
        densities.append(density)
        durations.append(duration)

    densities_gs, densities_a7, densities = np.transpose(np.array(densities), axes=[1, 0, 2])
    durations_gs, durations_a7, durations = np.transpose(np.array(durations), axes=[1, 0, 2])

    print_correlations(densities_gs, densities_a7, densities, durations_gs, durations_a7, durations)
