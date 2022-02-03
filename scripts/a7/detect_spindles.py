import numpy as np

from .calculate_features import absolute_power_values, relative_power_values, covariance_values, correlation_values


def possible_spindle_indices(features, thresholds):
    # All four features must be above their respective thresholds at least once (simultaneously) during a spindle
    all_thresholds_exceeded_idx = np.where(np.all(features > thresholds, axis=1))[0]

    # Only the features a7_relative_sigma_power and a7_sigma_correlation (features indices [0,2]) exceeding their
    # respective thresholds are relevant for the start and end of a spindle detection (as long as all features exceed
    # their respective thresholds at least once simultaneously during the spindle)
    absolute_thresholds_exceeded = np.all(features[:, [0, 2]] > thresholds[[0, 2]], axis=1)

    # The changes of the features a7_relative_sigma_power and a7_sigma_correlation exceeding their thresholds are calculated
    # The .astype(int) conversion leads to 1 for True and 0 for False
    # The leading padded zero allows detecting a spindle that starts at the first sample, the trailing padded zero
    # allows detecting a spindle that ends at the last sample
    absolute_threshold_changes = np.diff(np.r_[0, absolute_thresholds_exceeded.astype(int), 0])

    # If a sample is the start of a spindle, there is a change from False to True (0 to 1) for
    # "features[:,[0,2]] > thresholds[[0,2]]", which means the value of the diff function has to be 1
    start_candidates_idx = np.where(absolute_threshold_changes == 1)[0]

    # If a sample is the first sample after an end of a spindle, there is a change from True to False (1 to 0) for
    # "features[:,[0,2]] > thresholds[[0,2]]", which means the value of the diff function has to be -1
    stop_candidates_idx = np.where(absolute_threshold_changes == -1)[0]

    # The (start_candidates_idx,stop_candidates_idx) pairs are only actual starts and stops of spindles, if there is at
    # least one sample in between at which all four thresholds are exceeded simultaneously
    start_stop_candidates_idx = np.c_[start_candidates_idx, stop_candidates_idx]
    start_stop_idx = np.array([[start, stop] for start, stop in start_stop_candidates_idx if
                               np.any((start <= all_thresholds_exceeded_idx) & (all_thresholds_exceeded_idx < stop))])
    # In case no spindle candidate is found, the dimension is (1, 0) and has to be set to (0, 2)
    start_stop_idx = start_stop_idx.reshape(-1, 2)

    # Return the (included) start indices and the (excluded) stop indices
    return start_stop_idx[:, 0], start_stop_idx[:, 1]


def detect_spindles(data, thresholds, win_length_sec, win_step_sec, sample_rate):
    bsl_length_sec = 30
    spindle_length_min_sec = 0.3
    spindle_length_max_sec = 2.5

    # Calculate the four features of the given data
    a7_absolute_sigma_power = absolute_power_values(data, win_length_sec, win_step_sec, sample_rate)
    a7_relative_sigma_power = relative_power_values(data, win_length_sec, win_step_sec, bsl_length_sec, sample_rate)
    a7_sigma_covariance = covariance_values(data, win_length_sec, win_step_sec, bsl_length_sec, sample_rate)
    a7_sigma_correlation = correlation_values(data, win_length_sec, win_step_sec, sample_rate)

    # Stack the features to a (n_samples, 4) matrix
    features = np.stack((a7_absolute_sigma_power, a7_relative_sigma_power, a7_sigma_covariance, a7_sigma_correlation),
                        axis=1)
    # With the features and the given thresholds calculate the start and stop indices of possible spindles
    start_idx, stop_idx = possible_spindle_indices(features, thresholds)

    spindle_length = stop_idx - start_idx
    # Only detected spindles whose length (in seconds) is between spindle_length_min_sec and spindle_length_max_sec are considered
    valid_idx = (spindle_length >= spindle_length_min_sec * sample_rate) & (
            spindle_length <= spindle_length_max_sec * sample_rate)

    # Only return the indices of valid spindles
    start_idx, stop_idx = start_idx[valid_idx], stop_idx[valid_idx]
    return features, np.c_[start_idx, stop_idx]
