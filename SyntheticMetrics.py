import pickle
import numpy as np
from scipy.stats import kurtosis, skew, pearsonr, wasserstein_distance, entropy
from scipy.spatial.distance import jensenshannon
import math

# Load the data from the file
with open('synthetic_data.pkl', 'rb') as f:
    data = pickle.load(f)


## Organizing data into dictionary ##
# Create an empty dictionary
data_dict = {}
# Split the data into synthetic and real groups
synthetic_data = data[:10]  # The first 5 segments are synthetic
real_data = data[10:]  # The last 5 segments are real
# Assign the groups to the keys in the dictionary
data_dict['Synthetic'] = synthetic_data
data_dict['Real'] = real_data


def medium_wave(segment):
    """
    -----
    Brief
    -----
    Compute the mean of all timeseries at the same point.
    ----------
    Parameters
    ----------
    segment : nd-array or list
        Timeseries of the same lenght to be averaged.
    Returns
    -------
    mean_list : list
        mean value at each sample.
    std_list : list
        standard deviation at each sample.
    """
    mean_list = []
    std_list = []
    segment = np.array(segment)
    segment_t = segment.transpose()
    for i in range(len(segment_t)):
        mean = np.mean(segment_t[i])
        std = np.std(segment_t[i])
        mean_list.append(mean)
        std_list.append(std)
    return mean_list, std_list


def calculate_num_bins(data):
    """
    -----
    Brief
    -----
    Calculate number of bins using the Freedman-Diaconis rule.
    ----------
    Parameters
    ----------
    data : nd-array
        Input signal.
    Returns
    -------
    num_bins : int
        Number of bins to build histogram.
    """
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / len(data) ** (1 / 3)
    num_bins = int(np.ceil((np.max(data) - np.min(data)) / bin_width))
    return num_bins


#### HISTOGRAM ANALYSIS ####
def time_analysis(real_data, synthetic_data):
    """
    -----
    Brief
    -----
    Prints the time statistics for the input real data and for the input synthetic data.
    These include:
        - Mean
        -Standard Deviation
        - Maximum
        - Minimum
        - Kurtosis
        - Skewness
        - Correlation
    ----------
    Parameters
    ----------
    real_data : nd-array or list
        Input real signals.
    synthetic_data : nd-array or list
        Input synthetic signals.
    """
    # Multiple Signal Time Analysis
    if isinstance(real_data,(np.ndarray, list)) and np.array(real_data).ndim >= 2:
        real_data,_ = medium_wave(real_data)
        synthetic_data,_ = medium_wave(synthetic_data)

    # Signal Time Analysis
    print('Mean_Real:', np.mean(real_data))
    print('Mean_Synthetic:', np.mean(synthetic_data))
    print('STD_Real:', np.std(real_data))
    print('STD_Synthetic:', np.std(synthetic_data))
    print('Max_Real:', np.max(real_data))
    print('Max_Synthetic:', np.max(synthetic_data))
    print('Min_Real:', np.min(real_data))
    print('Min_Synthetic:', np.min(synthetic_data))
    print('Kurtosis_Real:', kurtosis(real_data))
    print('Kurtosis_Synthetic:', kurtosis(synthetic_data))
    print('Skewness_Real:', skew(real_data))
    print('Skewness_Synthetic:', skew(synthetic_data))
    correlation, _ = pearsonr(real_data, synthetic_data)
    print('Correlation:', correlation)


def wasserstein_distance_(time_series1, time_series2, num_bins=30, range_bins=(0, 1)):
    """
    -----
    Brief
    -----
    Computes the Earth Movers Distance between the distribution of two timeseries or the mean distance between
    multiple timeseries.
    ----------
    Parameters
    ----------
    time_series1 : nd-array or list
        Input signals 1.
    time_series2 : nd-array or list
        Input signals 2.
    num_bins : int
        Number of bins of the histogram. Default value is 30.
    range_bins : tuple
        The lower and upper range of the bins. Default is (0,1).

    Returns
    -------
    wasserstein_dist : float
        The mean Wasserstein distance between the input timeseries distributions.
    """
    # Convert to list if input is a single array, or if list only contains one signal, nest it into another list.
    if isinstance(time_series1, (np.ndarray, list)) and np.array(time_series1).ndim == 1:
        time_series1 = [time_series1]
    if isinstance(time_series2, (np.ndarray, list)) and np.array(time_series2).ndim == 1:
        time_series2 = [time_series2]

    wasserstein_dist = []
    n_signals1 = len(time_series1)
    n_signals2 = len(time_series2)

    for sig1 in range(n_signals1):
        for sig2 in range(n_signals2):
            hist_1, bin_edges_1 = np.histogram(time_series1[sig1], bins=num_bins, range=range_bins, density=True)
            hist_2, bin_edges_2 = np.histogram(time_series2[sig2], bins=num_bins, range=range_bins, density=True)
            bin_midpoints = (bin_edges_1[:-1] + bin_edges_1[1:]) / 2
            wasserstein_dist_ = wasserstein_distance(bin_midpoints, bin_midpoints, u_weights=hist_1, v_weights=hist_2)
            wasserstein_dist.append(wasserstein_dist_)

    mean_wasserstein_dist = np.mean(wasserstein_dist)
    std_wasserstein_dist = np.std(wasserstein_dist)

    print('WD:', mean_wasserstein_dist, 'STD:', std_wasserstein_dist)

    return mean_wasserstein_dist


def kl_divergence(time_series1, time_series2, num_bins=30, range_bins=(0, 1)):
    """
    -----
    Brief
    -----
    Compute the Kullback-Leibler divergence (difference between two probability distributions) between two time series.

    ----------
    Parameters
    ----------
    time_series1: 1D array-like, first time series.
    time_series2: 1D array-like, second time series.
    num_bins: int
        number of bins for the histograms.
    range_bins: tuple
        Range of the bins.

    Returns
    -------
    kl_divergence: float
        the KL divergence between the two distributions.
    """

    # Compute histograms
    hist_1, bin_edges_1 = np.histogram(time_series1, bins=num_bins, range=range_bins, density=True)
    hist_2, bin_edges_2 = np.histogram(time_series2, bins=num_bins, range=range_bins, density=True)

    # Add a small value to avoid division by zero or log of zero
    epsilon = 1e-10
    hist_1 += epsilon
    hist_2 += epsilon

    # Normalize histograms to form probability distributions
    hist_1 /= np.sum(hist_1)
    hist_2 /= np.sum(hist_2)

    # Compute KL divergence
    kl_div = entropy(hist_1, hist_2)

    print("Kullback-Leibler Divergence:", kl_div)

    return kl_div


def js_divergence(time_series1, time_series2, num_bins=30, range_bins=(0, 1)):
    """
    -----
    Brief
    -----
    Compute the Jensen-Shannon (JS) Distance (measure of the similarity between two probability distributions)
    between two time series. This is a symmetric and smoothed version of the KL divergence

    ----------
    Parameters
    ----------
    time_series1: 1D array-like
        first time series.
    time_series2: 1D array-like
        second time series.
    num_bins: int
        number of bins for the histograms.
    range_bins: tuple
        range of the bins.

    Returns:
    -------
    js_divergence: float
        the JS distance between the two distributions.
    """
    # Compute histograms
    hist_1, bin_edges_1 = np.histogram(time_series1, bins=num_bins, range=range_bins, density=True)
    hist_2, bin_edges_2 = np.histogram(time_series2, bins=num_bins, range=range_bins, density=True)

    # Add a small value to avoid division by zero or log of zero
    epsilon = 1e-10
    hist_1 += epsilon
    hist_2 += epsilon

    # Normalize histograms to form probability distributions
    hist_1 /= np.sum(hist_1)
    hist_2 /= np.sum(hist_2)

    # Compute KL divergence
    js_div = jensenshannon(hist_1, hist_2)

    print("Jensen-Shannon Distance:", js_div)
    return js_div


## Usage on multiple signals ##
time_analysis(real_data, synthetic_data)
wasserstein_distance_(real_data, synthetic_data)
wasserstein_distance_(real_data, real_data)
wasserstein_distance_(synthetic_data, synthetic_data)

## Usage on one signal ##
time_analysis(real_data[0], synthetic_data[0])
wasserstein_distance_(np.array(real_data[0]), np.array(synthetic_data[0]))
kl_divergence(real_data[0], synthetic_data[0])
js_divergence(real_data[0], synthetic_data[0])


def hellinger_distance(time_series1, time_series2, num_bins = 30, range_bins = (0,1)):
    """
    -----
    Brief
    -----
    The Hellinger Distance ranges from 0 to 1, where 0 indicates perfect similarity between distributions,
    and 1 is maximum dissimilarity.

    ----------
    Parameters
    ----------
    p: 1d-array
        distribution 1.
    q: 1d-array
        distribution 2.
    Returns:
    -------
    sosq / math.sqrt(2): float
        The Hellinger distance between the two distributions.
    """
    # Compute histograms
    p, bin_edges_1 = np.histogram(time_series1, bins=num_bins, range=range_bins, density=True)
    q, bin_edges_2 = np.histogram(time_series2, bins=num_bins, range=range_bins, density=True)

    # Add a small value to avoid division by zero or log of zero
    epsilon = 1e-10
    p += epsilon
    q += epsilon

    # Normalize histograms to form probability distributions
    p /= np.sum(p)
    q /= np.sum(q)
    list_of_squares = []
    for p_i, q_i in zip(p, q):
        # caluclate the square of the difference of ith distribution elements
        s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2

        # append
        list_of_squares.append(s)

    # calculate sum of squares
    sosq = sum(list_of_squares)
    print('Helinger Distance :',sosq / math.sqrt(2))
    return sosq / math.sqrt(2)
# Usage
hellinger_distance(real_data[0], synthetic_data[0])


def bhattacharyya_distance(time_series1, time_series2, num_bins, range_bins):
    """
    -----
    Brief
    -----
    The Bhattacharyya Distance ,measures the overlap between two probability distributions.
    ----------
    Parameters
    ----------
    time_series1 : 1d-array
        distribution 1.
    time_series2 : 1d-array
        distribution 2.
    num_bins : int
        Number of bins for the histogram
    range_bins : tuple
        The lower and upper range of the bins. Default is (0,1).
    Returns
    -------
    -np.log(bht): float
        The Bhattacharyya distance between the two distributions of the timeseries.
    """
    # Compute histograms
    hist_1, bin_edges_1 = np.histogram(time_series1, bins=num_bins, range=range_bins, density=True)
    hist_2, bin_edges_2 = np.histogram(time_series2, bins=num_bins, range=range_bins, density=True)

    bht = 0
    cX = np.concatenate((np.array(time_series1), np.array(time_series2)))
    print(cX)
    for i in range(num_bins):
        p1 = hist_1[i]
        p2 = hist_2[i]
        bht += math.sqrt(p1*p2) * (max(cX) - min(cX))/num_bins

    if bht == 0:
        print('Bhattacharyya Distance:', float('Inf'))
        return float('Inf')
    else:
        print('Bhattacharyya Distance:', -np.log(bht))
        return -np.log(bht)

# Usage
bhattacharyya_distance(real_data[0], synthetic_data[0], 30, (0,1))