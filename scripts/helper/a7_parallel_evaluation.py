"""
As the parallel execution using the ProcessPoolExecutor only seems to work if the parallelized function is imported,
the split_evaluation function is extracted into this helper file.
"""
import pickle

import numpy as np

from scripts.a7.detect_spindles import detect_spindles
from sumo.data import spindle_vect_to_indices
from sumo.evaluation import get_true_positives, metric_scores

sampling_rate = 100
win_length_sec = 0.3
win_step_sec = 0.1
thresholds = np.array([1.25, 1.6, 1.3, 0.69])

n_overlaps = 21
overlap_thresholds = np.linspace(0, 1, n_overlaps)


def split_evaluation(input_path):
    with open(input_path, 'rb') as input_file:
        subjects_test = pickle.load(input_file)['test']
    subjects_test = [subject for cohort in subjects_test for subject in cohort]

    n_spindles, n_spindles_gs = 0, 0
    n_true_positives = np.zeros_like(overlap_thresholds, dtype=int)
    for subject in subjects_test:
        data_blocks = subject.data
        spindle_blocks = subject.spindles
        for data_vect, spindle_vect in zip(data_blocks, spindle_blocks):
            spindles_gs = spindle_vect_to_indices(spindle_vect)
            spindles = detect_spindles(data_vect, thresholds, win_length_sec, win_step_sec, sampling_rate)[1]

            n_spindles += spindles.shape[0]
            n_spindles_gs += spindles_gs.shape[0]
            n_true_positives += get_true_positives(spindles, spindles_gs, overlap_thresholds)

    return metric_scores(n_spindles, n_spindles_gs, n_true_positives)[2].mean()
