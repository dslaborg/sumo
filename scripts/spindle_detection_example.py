from pathlib import Path
from sys import path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

# append root dir to python path so that we find `sumo`
path.insert(0, str(Path(__file__).absolute().parents[1]))
from sumo.config import Config
from sumo.data import MODADataModule, spindle_vect_to_indices
from sumo.evaluation import f1_scores
from sumo.model import SUMO


def plot_data(data, spindle_gs_idx, spindle_idx, prediction):
    start_time, stop_time = 63, 83
    data_start, data_stop = round(start_time * sample_rate), round(stop_time * sample_rate)
    times = np.arange(0, stop_time-start_time, 1 / sample_rate)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    major_ticks = np.arange(0, stop_time - start_time + 0.1, 5)
    minor_ticks = np.arange(0, stop_time - start_time + 0.1, 0.5)
    ax2.set_xlim([0, stop_time - start_time])
    ax2.set_xticks(major_ticks)
    ax2.set_xticks(minor_ticks, minor=True)
    ax2.set_xlabel(r'time ($s$)')

    for ax in [ax1, ax2]:
        ax.grid(which='both', axis='x', alpha=0.5)

    ax1.set_ylabel(r'EEG ($\mu V$)')
    ax1.set_yticks([-50, 0, 50])
    ax1.plot(times, data[data_start:data_stop], linewidth=1, color='black')

    y_min = ax1.get_ylim()[0]
    lines_gs = (spindle_gs_idx[(spindle_gs_idx[:, 0] >= data_start) & (spindle_gs_idx[:, 1] <= data_stop)] - data_start) / sample_rate
    ax1.hlines(np.repeat(y_min + 15, lines_gs.shape[0]), lines_gs[:, 0], lines_gs[:, 1], linewidth=2, color='green', label='expert consensus')
    lines = (spindle_idx[(spindle_idx[:, 0] >= data_start) & (spindle_idx[:, 1] <= data_stop)] - data_start) / sample_rate
    ax1.hlines(np.repeat(y_min + 5, lines.shape[0]), lines[:, 0], lines[:, 1], linewidth=2, color='#ff7f0e', label='SUMO')
    ax1.legend(loc='upper right')

    ax2.set_ylabel(r'probability')
    ax2.set_yticks([0, 1])
    ax2.set_yticks([0.5], minor=True)
    ax2.plot(times, prediction[1, data_start:data_stop], linewidth=1, color='#ff7f0e')
    ax2.axhline(0.5, linestyle=':', color='black', alpha=0.7)

    fig.tight_layout()

    output_dir = base_dir / 'output' / 'figures'
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / 'spindle_detection_example.svg')


def plot_good_interval(subjects_all, predictions_all):
    for i, (subjects, predictions) in enumerate(zip(subjects_all, predictions_all)):
        for j, (subject, prediction) in enumerate(zip(subjects, predictions)):
            spindle_vect = prediction.argmax(dim=1).long().detach().cpu().numpy()
            prediction = prediction.detach().cpu().numpy()

            for k, (d, sv_gs, sv, pred) in enumerate(zip(subject.data, subject.spindles, spindle_vect, prediction)):
                sp_gs = spindle_vect_to_indices(sv_gs)
                sp = spindle_vect_to_indices(sv)

                precision, recall, f1 = f1_scores(sp, sp_gs, overlap_thresholds)
                if len(sp_gs) >= 10 and len(sp) >= 10 and f1[4] >= 0.95 and f1.mean() >= 0.8:
                    plot_data(d, sp_gs, sp, pred)
                    return


def predict_step(self, batch, batch_idx, dataloader_idx=None):
    data, mask = batch
    # return the predictions after postprocessing and transformation by softmax
    return F.softmax(self(data), dim=1)


if __name__ == '__main__':
    plt.rcParams['svg.fonttype'] = 'none'

    # dirty hack to be able to easily plot the prediction (without argmax)
    SUMO.predict_step = predict_step

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

    sample_rate = 100
    overlap_thresholds = config.overlap_thresholds

    trainer = pl.Trainer(gpus=1, num_sanity_val_steps=0, logger=False)
    predictions_all = trainer.predict(model, datamodule.test_dataloader())

    plot_good_interval(datamodule.subjects_test, predictions_all)
