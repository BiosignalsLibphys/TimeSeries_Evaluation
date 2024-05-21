### Packages ###
import numpy as np
from scipy.signal import welch
import fathon
from fathon import fathonUtils as fu
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from scipy.signal import detrend
from scipy.fft import fft, fftfreq


######## QCoD metrics ########
def QCod(fs, thresh, signal):
    """
    -----
    Brief
    -----
    Computes the Power Spectral Density of the signal. Next it divides the power spectrum into quartiles.
    Determines the QCoD threshold, i.e. it compares the first quartile where most power is expected to be with the
    third quartile, where there shouldn't be any spectral density contents.
    ----------
    Parameters
    ----------
    fs : int
        sampling frequency
    thresh : float
        maximum threshold acceptable for the presence of white noise in the PSD.
    signal : 1D-array
        A single signal or a dataset of signals.

    Returns
    -------
    mask : int
        1 (not contaminated) or 0 (contaminated).
     f : ND-array
        Sampling frequencies.
    psd : nd-array
        Power spectrum of timeseries.
    """
    # computing the spectral power density
    f, psd = welch(signal, fs, nperseg=(len(signal) // 2))
    # dividing psd into quartiles
    psdquarters = int(round(len(psd)) / 4)
    # Cumulative sum of the values in the 1st quartile
    q1 = np.sum(psd[:psdquarters])
    # Cumulative sum of the values in the 3rd quartile
    q3 = sum(psd[2 * psdquarters + 1:3 * psdquarters])

    qcod = (q1 - q3) / (q1 + q3)
    mask = 1 if qcod >= thresh else 0
    return mask, f, psd


######## Completness metrics ########
def completeness(signal):
    """
    -----
    Brief
    -----
    It computes the percentage of missing values in the time-series.
    ----------
    Parameters
    ----------
    signal : 1D-array or list
        The time series data.

    Returns:
    -------
    missing_percentage : float
        Percentage of the time-series that are missing values.
    """
    missing_values = np.isnan(signal).sum()
    total_values = len(signal)
    missing_percentage = (missing_values / total_values) * 100

    return missing_percentage


######## Uniqueness metrics ########
def uniqueness(signal):
    """
    -----
    Brief
    -----
    Analyzes the uniqueness of a signal or dataset of signals, defined as the percentage
    of consecutive unique values. If the signal has multiple dimensions, the function returns
    the average percentage of consecutive unique values across all dimensions.
    ----------
    Parameters:
    ----------
    signal: nd-array or list
        A single signal or a dataset of signals.

    Returns:
    -------
    uniqueness_percentage : float
        The uniqueness percentage of the signal.
    average_uniqueness : ND-array
        If the input is multi-dimensional, the average uniqueness percentage across dimensions is returned.
    """

    # Ensure signal is a numpy array
    signal = np.asarray(signal)

    # Initialize a list to store uniqueness percentages for each dimension
    uniqueness_percentages = []

    # Check if the signal is multi-dimensional
    if signal.ndim == 1:
        # For 1D signal, simply compare each element to the next
        unique_consecutive = np.sum(signal[:-1] != signal[1:])
        total_comparisons = max(len(signal) - 1, 1)  # Avoid division by zero
        uniqueness_percentage = (unique_consecutive / total_comparisons) * 100
        return uniqueness_percentage
    else:
        # For multi-dimensional signals, iterate over dimensions
        for dim in range(signal.shape[1]):
            unique_consecutive = np.sum(signal[:-1, dim] != signal[1:, dim])
            total_comparisons = max(signal.shape[0] - 1, 1)  # Avoid division by zero
            uniqueness_percentages.append((unique_consecutive / total_comparisons) * 100)

        # Calculate the average uniqueness percentage across dimensions
        average_uniqueness = np.mean(uniqueness_percentages)
        return average_uniqueness


######## Hurst Exponent analysis ########
def hurst_exponent(signal):
    """
    -----
    Brief
    -----
    Computes the Hurst exponent for the time-series.
    ----------
    Parameters
    ----------
    signal : nd-array or list
        The input timeseries.

    Returns:
    -------
    H : float
        Value of the Hurst exponent.
    """
    # zero - mean cumulative sum
    a = fu.toAggregated(signal)
    # Initialize the dfa object
    pydcca = fathon.DFA(a)
    polOrd = 1
    # compute fluctuation function
    winSizes = fu.linRangeByStep(16, len(signal) / 8, step=50)
    n, F = pydcca.computeFlucVec(winSizes, polOrd=polOrd)
    # compute the Hurst exponent
    H, _ = pydcca.fitFlucVec()
    return H


######## Amplitude analysis ########
def amplitude(data):
    """
    -----
    Brief
    -----
    Computes the signals' maximum amplitude.
    ----------
    Parameters
    ----------
    data : nd-array or list
        Input signal/ timeseries.
    Returns:
    -------
    max_amplitude : float
        maximum amplitude present in the input signal/timeseries.
    """
    # Detrend the signal to remove linear trend
    detrended_signal = detrend(data)

    # Calculate the maximum amplitude of the detrended signal
    max_amplitude = np.max(np.abs(detrended_signal))

    return max_amplitude


##### PCA noise analysis #####
def pca_and_auc(data, sampling_rate):
    """
    -----
    Brief
    -----
    Computes the PCA, cumulative variance and area under the curve from the number of components vs cumulated variance.
    ----------
    Parameters
    ----------
    data : nd-array
        array with the signal to be analyzed.
    sampling_rate : int
        signal's sampling frequency in Hz.
    Returns:
    -------
    rel_PCA_AUC : float
        value of the relationship between the area under the curve of the number of components vs cumulated variance
        and the total area.
    """
    try:
        # Check if data is a non-empty numpy array
        if not isinstance(data, np.ndarray) or data.size == 0:
            return print("Invalid data: Data should be a non-empty numpy array.")

        # Calculate the length of the data in seconds
        len_data_seconds = len(data) / sampling_rate

        # Reshape the data for PCA
        data = data.reshape(int(-len_data_seconds), int(len_data_seconds))

        # Apply PCA to the data
        pca = PCA()
        pca.fit(data)

        # Calculate the cumulative variance explained by the principal components
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.arange(len(cumulative_variance))

        # Calculate the area under the cumulative variance curve
        area_under_curve = auc(num_components, cumulative_variance)

        # Calculate the maximum possible AUC
        max_possible_auc = len(cumulative_variance) * 1.0

        # Calculate the relative PCA AUC
        rel_PCA_AUC = area_under_curve / max_possible_auc

        return rel_PCA_AUC

    except Exception as e:
        print(f"An error occurred: {e}")


##### SNR Classification #####
def calculate_snr(data):
    #todo i dont know if the base signal is only valid for EEG?
    """
    -----
    Brief
    -----
    Computes the Signal to Noise Ratio (SNR).
    ----------
    Parameters
    ----------
    data : nd-array or list
        The input signal.
    Returns:
    -------
    snr : float
        SNR in dB.
    """

    # Create a base signal for comparison
    fs = 1000  # Sampling frequency in Hz
    t = np.arange(0, 15, 1 / fs)  # 15 seconds duration
    # Create a base EEG-like signal (for simplicity, a sinusoidal wave)
    base_signal = 1.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave

    # Calculate RMS of signal and noise
    rms_base_signal = np.sqrt(np.mean(base_signal ** 2))
    rms_noise = np.sqrt(np.mean(np.array(data) ** 2))
    values = []
    # Calculate SNR
    if rms_noise > 0:
        snr = 20 * np.log10(rms_base_signal / rms_noise)
    else:
        snr = np.inf
    return snr


##### Saturation ######
def saturation(data, sampling_rate, th):
    """
    -----
    Brief
    -----
    Computes the duration for which the signal remains at its maximum amplitude.
    ----------
    Parameters
    ----------
    data : nd-array or list
        The input signal data.
    sampling_rate : int
        The sampling rate of the signal in Hz.
    th: float
        the threshold to consider the maximum amplitude possible.

    Returns:
    -------
    total_saturation_duration : float
                              Duration of saturated signal.
    """
    try:
        # Detrending the signal
        detrended_signal = detrend(data)
        max_amplitude = np.max(detrended_signal)

        # Saturation evaluation: calculate how many samples correspond to 200 milliseconds
        samples_needed_for_200ms = int(200 * sampling_rate / 1000)

        # Define a threshold for detecting the maximum amplitude, considering floating-point precision
        threshold = th
        is_max_amplitude = np.isclose(detrended_signal, max_amplitude, atol=threshold)

        # Identify consecutive segments of max amplitude
        changes = np.diff(is_max_amplitude.astype(int))
        segment_starts = np.where(changes == 1)[0] + 1
        segment_ends = np.where(changes == -1)[0] + 1

        # Handle the case where the signal starts or ends with a max amplitude segment
        if is_max_amplitude[0]:
            segment_starts = np.r_[0, segment_starts]
        if is_max_amplitude[-1]:
            segment_ends = np.r_[segment_ends, len(is_max_amplitude)]

        # Calculate the lengths of the segments
        segment_lengths = segment_ends - segment_starts

        # Check for any long max amplitude segments
        long_max_amplitude_segments = segment_lengths[segment_lengths >= samples_needed_for_200ms]

        # Calculate total duration of saturation
        total_saturation_duration = np.sum(long_max_amplitude_segments) / sampling_rate

        return total_saturation_duration

    except Exception as e:
        print(f"An error occurred: {e}")


##### Powerline Interference ######
def power_line_noise(data, sampling_rate):
    """
    -----
    Brief
    -----
    Computes the 50Hz powerline noise amplitude.
    ----------
    Parameters
    ----------
    data : nd-array or list
        The input signal data.
    sampling_rate : int
        The sampling rate of the signal in Hz.

    Returns:
    -------
    amplitude_50hz : float
                   The Power amplitude of the 50 Hz frequency contained in the signal.
    """

    # Compute the FFT
    n = len(data)
    fft_values = fft(data)
    fft_freq = fftfreq(n, 1 / sampling_rate)

    # Find the nearest index to 50 Hz in the frequency array
    idx_50hz = np.argmin(np.abs(fft_freq - 50))

    # Get the amplitude at 50 Hz
    amplitude_50hz = np.abs(fft_values[idx_50hz])

    return amplitude_50hz
