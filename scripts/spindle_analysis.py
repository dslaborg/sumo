from pathlib import Path
from sys import path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import linregress

from a7.detect_spindles import detect_spindles

# append root dir to python path so that we find `sumo`
path.insert(0, str(Path(__file__).absolute().parents[1]))
from sumo.config import Config
from sumo.data import MODADataModule, spindle_vect_to_indices
from sumo.model import SUMO


def get_overlap(spindle_indices_gs, spindle_indices):
    assert len(spindle_indices_gs) == len(spindle_indices)

    def overlap_(spindles_gs, spindles_detected):
        n_detected_spindles = spindles_detected.shape[0]
        n_gs_spindles = spindles_gs.shape[0]

        # If either there is no spindle detected or the gold standard doesn't contain any spindles, there can't be any overlap
        if (n_detected_spindles == 0) | (n_gs_spindles == 0):
            return np.empty((n_detected_spindles, n_gs_spindles))

        # The (relative) overlap between each pair of detected spindle and gs spindle
        overlap = np.empty((n_detected_spindles, n_gs_spindles))

        for index in np.ndindex(n_detected_spindles, n_gs_spindles):
            idx_detected, idx_gs = spindles_detected[index[0]], spindles_gs[index[1]]
            # [start, stop) indices of the detected spindle and of the gs spindle
            idx_range_detected, idx_range_gs = np.arange(idx_detected[0], idx_detected[1]), np.arange(idx_gs[0], idx_gs[1])

            # Calculate intersect and union of the spindle indices
            intersect = np.intersect1d(idx_range_detected, idx_range_gs, assume_unique=True)
            union = np.union1d(idx_range_detected, idx_range_gs)

            # Overlap of a detected spindle and a gs spindle is defined as the intersect over the union
            overlap[index] = intersect.shape[0] / union.shape[0]  # type: ignore

        # Make sure there is at max one detection per gs event
        overlap_valid = np.where(overlap == np.max(overlap, axis=0), overlap, 0)
        # Make sure there is at max one gs event per detection
        overlap_valid = np.where(overlap_valid == np.max(overlap_valid, axis=1).reshape(-1, 1), overlap_valid, 0)

        return overlap_valid

    overlaps = []
    for spindles_gs, spindles in zip(spindle_indices_gs, spindle_indices):
        overlap = overlap_(spindles_gs, spindles)
        overlaps.append(overlap[overlap > 0])

    return np.concatenate(overlaps)


def get_density(spindle_indices_gs, spindle_indices_a7, spindle_indices):
    assert len(spindle_indices_gs) == len(spindle_indices_a7) == len(spindle_indices)
    n = len(spindle_indices)

    def density_(spindles):
        spindles = np.concatenate(spindles)
        return len(spindles) / (n * 115 / 60)

    density_gs = density_(spindle_indices_gs)
    density_a7 = density_(spindle_indices_a7)
    density = density_(spindle_indices)

    return density_gs, density_a7, density


def get_duration(spindle_indices_gs, spindle_indices_a7, spindle_indices):
    assert len(spindle_indices_gs) == len(spindle_indices_a7) == len(spindle_indices)

    def duration_(spindles):
        spindles = np.concatenate(spindles)
        return 0 if len(spindles) == 0 else np.mean(np.diff(spindles, axis=1)) / 100

    duration_gs = duration_(spindle_indices_gs)
    duration_a7 = duration_(spindle_indices_a7)
    duration = duration_(spindle_indices)

    return duration_gs, duration_a7, duration


def plot(overlaps_a7, overlaps, densities_gs, densities_a7, densities, durations_gs, durations_a7, durations):
    fig = plt.figure(figsize=(11, 6), constrained_layout=True)

    subfigures = fig.subfigures(1, 3)
    subfigures[0].suptitle('overlaps')
    subfigures[1].suptitle('density (spm)')
    subfigures[2].suptitle('duration (s)')

    axs = np.array([subfigures[0].subplots(2, 1, sharex=True),
                    subfigures[1].subplots(2, 1, sharex=True, sharey=True),
                    subfigures[2].subplots(2, 1, sharex=True, sharey=True)]).T

    def plot_hist_(axs):
        axs[0].hist((overlaps_a7[0] * 100).astype(int), bins=20, range=(0, 100), histtype='stepfilled', label='A7', alpha=0.5)
        axs[0].hist((overlaps[0] * 100).astype(int), bins=20, range=(0, 100), histtype='stepfilled', label='SUMO', alpha=0.5)

        axs[1].hist((overlaps_a7[1] * 100).astype(int), bins=20, range=(0, 100), histtype='stepfilled', label='A7', alpha=0.5)
        axs[1].hist((overlaps[1] * 100).astype(int), bins=20, range=(0, 100), histtype='stepfilled', label='SUMO', alpha=0.5)

        for ax in axs:
            ax.set_xlabel('overlap (%)')
            ax.set_ylabel('spindles')
            ax.label_outer()

            ax.set_xlim([0, 100])
            ax.set_xticks(list(range(0, 101, 20)))
            ax.set_xticks(list(range(10, 91, 20)), minor=True)

            ax.grid(which='both', alpha=0.5)
            ax.legend(loc='upper left')

    def plot_regression_(axs):
        regress = linregress(densities_gs[0], densities_a7[0])
        poly1d_fn = np.poly1d([regress.slope, regress.intercept])
        axs[0, 0].scatter(densities_gs[0], densities_a7[0], alpha=0.5)
        axs[0, 0].plot(densities_gs[0], poly1d_fn(densities_gs[0]), alpha=0.5, label=f'A7; $slope={round(regress.slope, 2)}$')

        regress = linregress(densities_gs[0], densities[0])
        poly1d_fn = np.poly1d([regress.slope, regress.intercept])
        axs[0, 0].scatter(densities_gs[0], densities[0], alpha=0.5)
        axs[0, 0].plot(densities_gs[0], poly1d_fn(densities_gs[0]), alpha=0.5, label=f'SUMO; $slope={round(regress.slope, 2)}$')

        regress = linregress(densities_gs[1], densities_a7[1])
        poly1d_fn = np.poly1d([regress.slope, regress.intercept])
        axs[1, 0].scatter(densities_gs[1], densities_a7[1], alpha=0.5)
        axs[1, 0].plot(densities_gs[1], poly1d_fn(densities_gs[1]), alpha=0.5, label=f'A7; $slope={round(regress.slope, 2)}$')

        regress = linregress(densities_gs[1], densities[1])
        poly1d_fn = np.poly1d([regress.slope, regress.intercept])
        axs[1, 0].scatter(densities_gs[1], densities[1], alpha=0.5)
        axs[1, 0].plot(densities_gs[1], poly1d_fn(densities_gs[1]), alpha=0.5, label=f'SUMO; $slope={round(regress.slope, 2)}$')

        regress = linregress(durations_gs[0], durations_a7[0])
        poly1d_fn = np.poly1d([regress.slope, regress.intercept])
        axs[0, 1].scatter(durations_gs[0], durations_a7[0], alpha=0.5)
        axs[0, 1].plot(durations_gs[0], poly1d_fn(durations_gs[0]), alpha=0.5, label=f'A7; $slope={round(regress.slope, 2)}$')

        regress = linregress(durations_gs[0], durations[0])
        poly1d_fn = np.poly1d([regress.slope, regress.intercept])
        axs[0, 1].scatter(durations_gs[0], durations[0], alpha=0.5)
        axs[0, 1].plot(durations_gs[0], poly1d_fn(durations_gs[0]), alpha=0.5, label=f'SUMO; $slope={round(regress.slope, 2)}$')

        regress = linregress(durations_gs[1], durations_a7[1])
        poly1d_fn = np.poly1d([regress.slope, regress.intercept])
        axs[1, 1].scatter(durations_gs[1], durations_a7[1], alpha=0.5)
        axs[1, 1].plot(durations_gs[1], poly1d_fn(durations_gs[1]), alpha=0.5, label=f'A7; $slope={round(regress.slope, 2)}$')

        regress = linregress(durations_gs[1], durations[1])
        poly1d_fn = np.poly1d([regress.slope, regress.intercept])
        axs[1, 1].scatter(durations_gs[1], durations[1], alpha=0.5)
        axs[1, 1].plot(durations_gs[1], poly1d_fn(durations_gs[1]), alpha=0.5, label=f'SUMO; $slope={round(regress.slope, 2)}$')

        for ax in axs.flatten():
            ax.set_xlabel('expert consensus')
            ax.set_ylabel('prediction')
            ax.label_outer()

            ax.grid(alpha=0.5)
            ax.legend()
            ax.set_aspect(1)

            r_min = min(*ax.get_xlim(), *ax.get_ylim())
            r_max = max(*ax.get_xlim(), *ax.get_ylim())
            ax.set_xlim([r_min, r_max])
            ax.set_ylim([r_min, r_max])

        axs[0, 0].set_xticks(list(range(0, 11, 2)))
        axs[0, 1].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])

    plot_hist_(axs[:, 0])
    plot_regression_(axs[:, 1:])

    output_dir = base_dir / 'output' / 'figures'
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / 'spindle_analysis.svg')


if __name__ == '__main__':
    plt.rcParams['svg.fonttype'] = 'none'

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

    overlaps, overlaps_a7 = [], []
    densities, durations = [], []
    for subjects, predictions in zip(datamodule.subjects_test, predictions_all):
        overlap, overlap_a7 = [], []
        density, duration = [], []

        for subject, prediction in zip(subjects, predictions):
            spindle_indices_gs = [spindle_vect_to_indices(v) for v in subject.spindles]

            spindle_indices_a7 = [detect_spindles(d, thresholds, win_length_sec, win_step_sec, sampling_rate)[1] for d in subject.data]

            prediction = prediction.detach().cpu().numpy()
            spindle_indices = [spindle_vect_to_indices(v) for v in prediction]

            overlap_a7.append(get_overlap(spindle_indices_gs, spindle_indices_a7))
            overlap.append(get_overlap(spindle_indices_gs, spindle_indices))

            density.append(get_density(spindle_indices_gs, spindle_indices_a7, spindle_indices))
            duration.append(get_duration(spindle_indices_gs, spindle_indices_a7, spindle_indices))

        overlap_a7 = np.concatenate(overlap_a7)
        overlap = np.concatenate(overlap)
        overlaps_a7.append(overlap_a7)
        overlaps.append(overlap)

        density = np.array(density).T
        duration = np.array(duration).T
        densities.append(density)
        durations.append(duration)

    densities_gs, densities_a7, densities = np.transpose(np.array(densities), axes=[1, 0, 2])
    durations_gs, durations_a7, durations = np.transpose(np.array(durations), axes=[1, 0, 2])

    plot(overlaps_a7, overlaps, densities_gs, densities_a7, densities, durations_gs, durations_a7, durations)
