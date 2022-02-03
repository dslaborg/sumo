import numpy as np
from scipy.stats import hmean

from sumo.data import spindle_vect_to_indices


def get_overlap(spindles_detected, spindles_gs):
    n_detected_spindles = spindles_detected.shape[0]
    n_gs_spindles = spindles_gs.shape[0]

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

    return overlap


def get_true_positives(spindles_detected, spindles_gs, overlap_thresholds):
    # If either there is no spindle detected or the gold standard doesn't contain any spindles, there can't be any true
    # positives
    if (spindles_detected.shape[0] == 0) or (spindles_gs.shape[0] == 0):
        return np.zeros_like(overlap_thresholds, dtype=np.int)

    # Get the overlaps in format (n_detected_spindles, n_gs_spindles)
    overlap = get_overlap(spindles_detected, spindles_gs)
    # Make sure there is at max one detection per gs event
    overlap_valid = np.where(overlap == np.max(overlap, axis=0), overlap, 0)
    # Make sure there is at max one gs event per detection
    overlap_valid = np.where(overlap_valid == np.max(overlap_valid, axis=1).reshape(-1, 1), overlap_valid, 0)

    n_true_positives = np.empty_like(overlap_thresholds, dtype=np.int)

    # Calculate the valid matches (true positives) depending on the overlap threshold
    for idx, overlap_threshold in enumerate(overlap_thresholds):
        # All remaining values > overlap_threshold are valid matches (true positives)
        matches = np.argwhere(overlap_valid > overlap_threshold)
        n_true_positives[idx] = matches.shape[0]

    return n_true_positives


def metric_scores(n_detected_spindles, n_gs_spindles, n_true_positives):
    if (n_detected_spindles == 0) & (n_gs_spindles == 0):
        # If there are no spindles detected and the gold standard doesn't contain any spindles, the precision, recall
        # and f1 score are defined as one
        precision = np.ones_like(n_true_positives)
        recall = np.ones_like(n_true_positives)
        f1 = np.ones_like(n_true_positives)
    elif (n_detected_spindles == 0) | (n_gs_spindles == 0):
        # If either there are no spindles detected or the gold standard doesn't contain any spindles, there can't be any
        # true positives and precision, recall and f1 score are defined as zero
        precision = np.zeros_like(n_true_positives)
        recall = np.zeros_like(n_true_positives)
        f1 = np.zeros_like(n_true_positives)
    else:
        # Precision is defined as TP/(TP+FP)
        precision = n_true_positives / n_detected_spindles
        # Recall is defined as TP/(TP+FN)
        recall = n_true_positives / n_gs_spindles
        # f1 score is defined as harmonic mean between precision and recall
        f1 = hmean(np.c_[precision, recall], axis=1)

    return precision, recall, f1


def f1_scores(spindles_detected, spindles_gs, overlap_thresholds):
    n_detected_spindles = spindles_detected.shape[0]
    n_gs_spindles = spindles_gs.shape[0]

    # Get the number of true positives per overlap threshold
    n_true_positives = get_true_positives(spindles_detected, spindles_gs, overlap_thresholds)

    return metric_scores(n_detected_spindles, n_gs_spindles, n_true_positives)


class PerformanceEvaluation:

    def __init__(self, spindle_vect, spindle_gs_vect, overlap_thresholds):
        self.spindle_vect = spindle_vect
        self.spindle_gs_vect = spindle_gs_vect

        self.overlap_thresholds = overlap_thresholds

        self.spindle_indices = spindle_vect_to_indices(spindle_vect)
        self.spindle_gs_indices = spindle_vect_to_indices(spindle_gs_vect)

    def evaluate_performance(self):
        true_positives = get_true_positives(self.spindle_indices, self.spindle_gs_indices, self.overlap_thresholds)

        return self.spindle_indices.shape[0], self.spindle_gs_indices.shape[0], true_positives
