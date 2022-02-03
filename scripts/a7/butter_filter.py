import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


def butter_bandpass_filter(data, lowcut, highcut, sample_rate, order):
    """
    Bandpass filter the data using Butterworth IIR filters.

    Two digital Butterworth IIR filters with the specified order are created, one highpass filter for the lower critical
    frequency and one lowpass filter for the higher critical frequency. Both filters use second-order sections (SOS).
    Then first the highpass filter is applied on the given data and on its result the lowpass filter is applied.
    Both filters are applied as forward-backward digital filters to correct the non-linear phase.

    Parameters
    ----------
    data : ndarray
        The data to be filtered; format (n_samples,)
    lowcut : float
        The lower critical frequency
    highcut : float
        The higher critical frequency
    sample_rate : float
        The sampling rate of the given data
    order : int
        The order of the used filters

    Returns
    -------
    data : ndarray
        the bandpass filtered data; format (n_samples,)
    """

    sos_high = butter(order, lowcut, btype='hp', fs=sample_rate, output='sos')
    sos_low = butter(order, highcut, btype='lp', fs=sample_rate, output='sos')
    return sosfiltfilt(sos_low, sosfiltfilt(sos_high, data, padlen=3 * order), padlen=3 * order)


def butter_lowpass_filter(data, highcut, sample_rate, order):
    sos = butter(order, highcut, btype='lp', fs=sample_rate, output='sos')
    return sosfiltfilt(sos, data, padlen=3 * order)


def butter_highpass_filter(data, lowcut, sample_rate, order):
    sos = butter(order, lowcut, btype='hp', fs=sample_rate, output='sos')
    return sosfiltfilt(sos, data, padlen=3 * order)


def downsample(data, sample_rate, resampling_frequency):
    if (sample_rate != int(sample_rate)) | (resampling_frequency != int(resampling_frequency)):
        raise Exception('parameters "sample_rate" and "resampling_frequency" have to be integers')
    elif sample_rate < resampling_frequency:
        raise Exception('the original sample frequency must not be lower than the resample frequency')
    elif sample_rate == resampling_frequency:
        return data

    sample_rate = int(sample_rate)
    resampling_frequency = int(resampling_frequency)

    gcd = np.gcd(sample_rate, resampling_frequency)

    up = resampling_frequency // gcd
    down = sample_rate // gcd

    return resample_poly(data, up, down)
