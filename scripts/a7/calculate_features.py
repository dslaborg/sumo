import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import hann

from .butter_filter import butter_bandpass_filter


def sample_to_win(data_sample, win_length_sec, win_step_sec, sample_rate):
    n_samples = data_sample.shape[0]

    # Length of the window as number of samples
    n_samples_per_win = round(win_length_sec * sample_rate)
    # Length of the window step as number of samples
    n_samples_per_win_step = round(win_step_sec * sample_rate)
    # Total length of the time series; in seconds
    data_length_sec = n_samples / sample_rate
    # Number of windows, last one might be incomplete
    n_windows = np.ceil((data_length_sec - win_length_sec) / win_step_sec).astype(int) + 1

    # 2d array with dimensions (n_windows,n_samples_per_win) representing the indices of the sliding window
    indexer = np.arange(n_samples_per_win)[None, :] + n_samples_per_win_step * np.arange(n_windows)[:, None]

    # Make sure a possible incomplete window at the end is handled correctly
    # Therefore temporarily for indexer>=n_samples just repeat the last value...
    data_win = data_sample[np.minimum(indexer, n_samples - 1)]
    # ...and afterwards mask theses values before returning
    return np.ma.masked_array(data_win, indexer >= n_samples)


def win_to_sample(data_win, win_length_sec, win_step_sec, sample_rate, n_samples):
    n_windows = data_win.shape[0]

    # Length of the window as number of samples
    n_samples_per_win = round(win_length_sec * sample_rate)
    # The number of samples overlapping between two windows
    overlap = np.ceil(win_length_sec / win_step_sec).astype(int)

    data_sample = np.full((overlap, n_samples), np.nan)
    i = 0
    while i < n_windows:
        # Iterate over the overlapping windows, effectively iterating the currently considered window (see "i = i + 1" below)
        for j in range(overlap):
            # The indices have to be computed here, as i is modified within this inner loop
            # Start index of the currently considered window
            idx_start = round(i * win_step_sec * sample_rate)
            # Stop index of the currently considered window
            idx_stop = idx_start + n_samples_per_win

            if i < n_windows:
                if idx_stop < n_samples:
                    # Complete window
                    data_sample[j, idx_start:idx_stop] = data_win[i]
                else:
                    # Incomplete window
                    data_sample[j, idx_start:] = data_win[i]

            # Incremented within the inner loop!
            i = i + 1

    # Mask all remaining NaN values (which always occur at beginning and end of the samples)
    return np.ma.masked_array(data_sample, np.isnan(data_sample))


def power_spectral_density(data, win_length_sec, win_step_sec, sample_rate, zero_pad_sec):
    # Length of the FFT window as number of samples
    n_samples_per_fft_win = round(win_length_sec * sample_rate)
    # Length of the FFT window padded to zero_pad_sec as number of samples
    n_samples_per_padded_fft_win = round(zero_pad_sec * sample_rate)

    # Hann window to scale/smooth the signal before FFT
    win_hann_coeffs = hann(n_samples_per_fft_win)
    # Noise gain
    ng = np.square(win_hann_coeffs).mean()

    # Get matrix of sliding PSD windows
    windows = sample_to_win(data, win_length_sec, win_step_sec, sample_rate)
    # Calculate the mean for every window
    windows_mean = windows.mean(axis=1)

    # Remove DC offset from windows...
    windows = windows - np.c_[windows_mean]
    # and scale them using the Hann window
    windows_scaled = windows * win_hann_coeffs

    # Perform the FFT with the window zero padded to zero_pad_sec. As the input is always real, only the positive
    # frequency terms are considered (as the result of FFT is Hermitian-symmetric), so the rfft function is used
    # As windows_scaled is a masked array and the rfft function cannot handle these, the masked values are filled with
    # zeros to not influence the result of rfft
    fft_modules = np.abs(rfft(windows_scaled.filled(0), n_samples_per_padded_fft_win))
    # Get the frequency bins used by the rfft function
    freq_bins = rfftfreq(n_samples_per_padded_fft_win, 1 / sample_rate)

    # Apply the IntegSpectPow PSA normalization as described by Hanspeter Schmid
    fft_modules = np.square(fft_modules / n_samples_per_padded_fft_win)
    fft_modules[:, 1:] = fft_modules[:, 1:] * 2
    psd = fft_modules / ng

    # As the DC offset was removed early, it needs to be added as first frequency again
    psd[:, 0] = windows_mean

    # Return the PSD and the used frequency bins
    return psd, freq_bins


def baseline_windows(data_per_win, win_length_sec, win_step_sec, bsl_length_sec):
    n_windows = data_per_win.shape[0]
    # Number of windows per baseline (with bsl_length_sec), last one (per baseline) might be incomplete
    n_windows_per_bsl = np.ceil((bsl_length_sec - win_length_sec) / win_step_sec).astype(int) + 1

    # The baseline per sliding window value is usually centered around the value, but always bsl_length_sec long
    # Therefore the start index of every baseline is its index minus half the number of windows per baseline...
    bsl_win_start_idx = np.arange(n_windows) - np.ceil((n_windows_per_bsl - 1) / 2).astype(int)
    # ... but always at least 0 and at most n_windows - n_windows_per_bsl
    bsl_win_start_idx = np.minimum(np.maximum(bsl_win_start_idx, 0), n_windows - n_windows_per_bsl)
    # The stop index (excluded) is now simply the start index plus n_windows_per_bsl
    bsl_win_stop_idx = bsl_win_start_idx + n_windows_per_bsl

    # As a "multidimensional" call to np.arange() isn't possible, the following lines produce the same output as
    # "bsl_wins_idx = np.array([np.arange(bsl_win_start_idx[i], bsl_win_stop_idx[i]) for i in range(n_windows)])"
    # see https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop
    bsl_wins_idx = np.repeat(bsl_win_stop_idx - np.arange(1, n_windows + 1) * n_windows_per_bsl,
                             n_windows_per_bsl) + np.arange(n_windows * n_windows_per_bsl)
    bsl_wins_idx = bsl_wins_idx.reshape(n_windows, -1)

    # Return the baseline windows per value in data_per_win with format (n_windows,n_windows_per_bsl)
    return data_per_win[bsl_wins_idx]


def baseline_z_score(data_per_win, win_length_sec, win_step_sec, bsl_length_sec):
    # The baseline windows for every sliding window value (so one baseline per sliding window)
    bsl_per_win = baseline_windows(data_per_win, win_length_sec, win_step_sec, bsl_length_sec)

    # Only consider values in the baseline included in the 10th-90th percentile, all other values are set to NaN
    limits = np.percentile(data_per_win, [10, 90])
    bsl_per_win[(bsl_per_win < limits[0]) | (bsl_per_win > limits[1])] = np.nan

    # Calculate the mean and std per baseline window, ignoring the NaN values
    bsl_per_win_mean = np.nanmean(bsl_per_win, axis=1)
    bsl_per_win_std = np.nanstd(bsl_per_win, axis=1)

    # Transform every sliding window value to its z-score using the corresponding baseline window
    return (data_per_win - bsl_per_win_mean) / bsl_per_win_std


def unmask_result(result):
    mask = result.mask
    # Make sure all samples have a value
    if np.any(mask):
        # If there are still NaN values, use linear interpolation
        # see https://stackoverflow.com/a/9537830
        result[mask] = np.interp(np.nonzero(mask)[0], np.nonzero(~mask)[0], result[~mask])

    # Return the result as a normal ndarray (not masked)
    return result.compressed()


def absolute_power_values(data, win_length_sec, win_step_sec, sample_rate):
    n_samples = data.shape[0]

    # Filter the sigma signal as bandpass filter from 11 to 16 Hz
    sigma_data = butter_bandpass_filter(data, 11, 16, sample_rate, 20)
    # Get matrix of sliding windows
    win_sample_matrix = sample_to_win(sigma_data, win_length_sec, win_step_sec, sample_rate)
    # Calculate average squared power per window
    absolute_power_per_win = np.square(win_sample_matrix).mean(axis=1)

    # Calculate absolute sigma power per sample (multiple values for samples with overlapping windows)
    absolute_power_per_sample = win_to_sample(absolute_power_per_win, win_length_sec, win_step_sec, sample_rate,
                                              n_samples)
    # Return the average absolute sigma power per sample log10 transformed
    return unmask_result(np.log10(absolute_power_per_sample.mean(axis=0)))


def relative_power_values(data, win_length_sec, win_step_sec, bsl_length_sec, sample_rate):
    n_samples = data.shape[0]

    # The time on which the sliding windows should be zero padded before performing the FFT
    zero_pad_sec = 2
    # Sliding window PSD and the used frequency bins
    psd, freq_bins = power_spectral_density(data, win_length_sec, win_step_sec, sample_rate, zero_pad_sec)

    # Calculate the sigma power by summing the PSD windows in the sigma band
    # As freq_index_stop should be excluded, 1 is added
    freq_index_start, freq_index_stop = np.argmin(np.abs(freq_bins - 11)), np.argmin(np.abs(freq_bins - 16)) + 1
    psd_sigma_freq = psd[:, freq_index_start:freq_index_stop].sum(axis=1)

    # Calculate the total power by summing the PSD windows in the broadband signal excluding delta band
    # As freq_index_stop should be excluded, 1 is added
    freq_index_start, freq_index_stop = np.argmin(np.abs(freq_bins - 4.5)), np.argmin(np.abs(freq_bins - 30)) + 1
    psd_total_freq = psd[:, freq_index_start:freq_index_stop].sum(axis=1)

    # Calculate the relative ratio of sigma power and total power log10 transformed
    relative_power_per_win = np.log10(psd_sigma_freq / psd_total_freq)

    # Calculate the z-score of every relative power using a baseline window
    relative_power_per_win_z_score = baseline_z_score(relative_power_per_win, win_length_sec, win_step_sec,
                                                      bsl_length_sec)

    # Calculate relative power ratio per sample (multiple values for samples with overlapping windows)
    relative_power_per_sample = win_to_sample(relative_power_per_win_z_score, win_length_sec, win_step_sec, sample_rate,
                                              n_samples)

    # Return the average relative power ratio per sample
    return unmask_result(relative_power_per_sample.mean(axis=0))


def covariance_values(data, win_length_sec, win_step_sec, bsl_length_sec, sample_rate):
    n_samples = data.shape[0]

    # Filter the sigma signal as bandpass filter from 11 to 16 Hz
    sigma_data = butter_bandpass_filter(data, 11, 16, sample_rate, 20)

    # Get matrix of sliding windows for broadband signal
    win_sample_matrix_raw = sample_to_win(data, win_length_sec, win_step_sec, sample_rate)
    # Get matrix of sliding windows for sigma band
    win_sample_matrix_sigma = sample_to_win(sigma_data, win_length_sec, win_step_sec, sample_rate)

    n_windows = win_sample_matrix_raw.shape[0]

    # Calculate the covariance between the two signals for every window except the last one
    covariance_per_win = np.array(
        [np.cov(win_sample_matrix_raw[i], win_sample_matrix_sigma[i])[0, 1] for i in range(n_windows - 1)])
    # The last window might be incomplete, so use the np.ma.cov function which handles missing data, but is much
    # slower than the np.cov function
    covariance_last_win = np.ma.cov(win_sample_matrix_raw[-1], win_sample_matrix_sigma[-1])[0, 1]
    covariance_per_win = np.r_[covariance_per_win, covariance_last_win]

    # Covariance between the two signals log10 transformed
    covariance_per_win_no_negative = covariance_per_win
    covariance_per_win_no_negative[covariance_per_win_no_negative < 0] = 0
    covariance_per_win_log10 = np.log10(covariance_per_win + 1)

    # Calculate the z-score of every covariance using a baseline window
    covariance_per_win_z_score = baseline_z_score(covariance_per_win_log10, win_length_sec, win_step_sec,
                                                  bsl_length_sec)

    # Calculate covariance per sample (multiple values for samples with overlapping windows)
    covariance_per_sample = win_to_sample(covariance_per_win_z_score, win_length_sec, win_step_sec, sample_rate,
                                          n_samples)

    # Return the average covariance per sample
    return unmask_result(covariance_per_sample.mean(axis=0))


def correlation_values(data, win_length_sec, win_step_sec, sample_rate):
    n_samples = data.shape[0]

    # Filter the sigma signal as bandpass filter from 11 to 16 Hz
    sigma_data = butter_bandpass_filter(data, 11, 16, sample_rate, 20)

    # Get matrix of sliding windows for broadband signal
    win_sample_matrix_raw = sample_to_win(data, win_length_sec, win_step_sec, sample_rate)
    # Get matrix of sliding windows for sigma band
    win_sample_matrix_sigma = sample_to_win(sigma_data, win_length_sec, win_step_sec, sample_rate)

    n_windows = win_sample_matrix_raw.shape[0]

    # Calculate the correlation between the two signals for every window except the last one
    correlation_per_win = np.array(
        [np.corrcoef(win_sample_matrix_raw[i], win_sample_matrix_sigma[i])[0, 1] for i in range(n_windows - 1)])
    # The last window might be incomplete, so use the np.ma.corrcoef function which handles missing data, but is much
    # slower than the np.corrcoef function
    correlation_last_win = np.ma.corrcoef(win_sample_matrix_raw[-1], win_sample_matrix_sigma[-1])[0, 1]
    correlation_per_win = np.r_[correlation_per_win, correlation_last_win]

    # Calculate correlation per sample (multiple values for samples with overlapping windows)
    correlation_per_sample = win_to_sample(correlation_per_win, win_length_sec, win_step_sec, sample_rate, n_samples)

    # Return the average correlation per sample
    return unmask_result(correlation_per_sample.mean(axis=0))
