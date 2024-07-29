import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import kurtosis, skew, pearsonr, wasserstein_distance, entropy
from scipy.spatial.distance import jensenshannon
import math
from scipy.signal import coherence, welch, cwt, morlet2
from scipy.stats import mode, entropy, kurtosis, skew, iqr, pearsonr
from scipy.integrate import simps

# Load the data from the file >>>> NO MEU CASO É HEALTHY
with open('healthy_synthetic_data.pkl', 'rb') as f:
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


#### HISTOGRAM ANALYSIS #### >>> OR TIME ANALYSIS?
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

#### FREQUENCY ANALYSIS ####

class FrequencyAnalysis:
    def __init__(self, fs=2048):
        self.fs = fs
        self.real_metrics = None
        self.synthetic_metrics = None
    def compute_relative_power(self,data, data_type):
        """
        Computes the relative power in different frequency bands for the given data.

        Parameters:
        data (list or np.ndarray): Input signals to analyze.
        data_type (str): Type of the data ('real' or 'synthetic').

        Returns:
        tuple: Computed frequency bands, power spectral densities, total power, and relative power in different bands.
        """
        freqs = []
        psd = []
        total_power = []
        slow_rel_power = []
        delta_rel_power = []
        theta_rel_power = []
        alpha_rel_power = []
        beta_rel_power = []
        dominant_freq = []
        fs = 2048
        win = 4 * self.fs
        bands = [0.5, 2, 4, 8, 13, 30]
        
        # Check if the input is a single signal or a list of signals
        if isinstance(data[0], (list, np.ndarray)):
            n_signals = len(data)
        else:
            data = [data]
            n_signals = 1

        for sig in range(n_signals):
            # Compute the Power Spectral Density (PSD) using Welch's method
            freqs_, psd_ = welch(data[sig], self.fs, nperseg=win)
            freqs.append(freqs_)
            psd.append(psd_)
            freq_res = freqs[sig][1] - freqs[sig][0]
            
            # Define frequency bands
            idx_total = np.logical_and(freqs[sig] >= 0, freqs[sig] <= 1024)
            idx_slow = np.logical_and(freqs[sig] >= bands[0], freqs[sig] <= bands[1])
            idx_delta = np.logical_and(freqs[sig] >= bands[1], freqs[sig] <= bands[2])
            idx_theta = np.logical_and(freqs[sig] >= bands[2], freqs[sig] <= bands[3])
            idx_alpha = np.logical_and(freqs[sig] >= bands[3], freqs[sig] <= bands[4])
            idx_beta = np.logical_and(freqs[sig] >= bands[4], freqs[sig] <= bands[5])
            
            # Calculate total power and relative power for each band
            total_power_sig = simps(psd[sig][idx_total], dx=freq_res)
            total_power.append(total_power_sig)
            for band, rel_power in zip([idx_slow, idx_delta, idx_theta, idx_alpha, idx_beta],
                                    [slow_rel_power, delta_rel_power, theta_rel_power, alpha_rel_power, beta_rel_power]):
                power = simps(psd[sig][band], dx=freq_res)
                rel_power.append(power / total_power_sig)
            # Determine the dominant frequency
            dominant_freq.append(freqs[sig][np.argmax(psd[sig])])

        # Print the mean and standard deviation of the relative powers
        print(f'{data_type} mean relative slow power: %.3f perc' % np.mean(slow_rel_power))
        print(f'{data_type} STD slow power: %.3f perc' % np.std(slow_rel_power))
        print(f'{data_type} mean relative delta power: %.3f perc' % np.mean(delta_rel_power))
        print(f'{data_type} STD delta power: %.3f perc' % np.std(delta_rel_power))
        print(f'{data_type} mean relative theta power: %.3f perc' % np.mean(theta_rel_power))
        print(f'{data_type} STD theta power: %.3f perc' % np.std(theta_rel_power))
        print(f'{data_type} mean relative alpha power: %.3f perc' % np.mean(alpha_rel_power))
        print(f'{data_type} STD alpha power: %.3f perc' % np.std(alpha_rel_power))
        print(f'{data_type} mean relative beta power: %.3f perc' % np.mean(beta_rel_power))
        print(f'{data_type} STD beta power: %.3f perc' % np.std(beta_rel_power))
        print(f'{data_type} Mean Dominant frequency: %.3f perc' % np.mean(dominant_freq))
        print(f'{data_type} STD Dominant frequency: %.3f perc' % np.std(dominant_freq))

        return freqs, psd, total_power, slow_rel_power, delta_rel_power, theta_rel_power, alpha_rel_power, beta_rel_power, dominant_freq, idx_slow, idx_delta

    def plot_psd(self,real_data, synthetic_data,x_limit1 = 0, x_limit2 = 8, y_limit1 = 0, y_limit2 = 0.04):  
        """
        Plots the power spectral density (PSD) for real and synthetic data.

        Parameters:
        real_data (list or np.ndarray): Real input signals.
        synthetic_data (list or np.ndarray): Synthetic input signals.
        x_limit1 (float): Lower limit for x-axis.
        x_limit2 (float): Upper limit for x-axis.
        y_limit1 (float): Lower limit for y-axis.
        y_limit2 (float): Upper limit for y-axis.
        """
        # Compute relative power for real and synthetic data
        freqs_r, psd_r, _, slow_rel_power_r, delta_rel_power_r, _, _, _, _,idx_slow,idx_delta = self.compute_relative_power(real_data,'real')
        freqs_s, psd_s, _, slow_rel_power_s, delta_rel_power_s, _, _, _, _,idx_slow,idx_delta = self.compute_relative_power(synthetic_data, 'synthetic')

        # Plot the PSD for real data
        plt.figure(figsize=(10, 8))
        plt.subplot(121)
        plt.text(5, 0.035, f'Slow: {slow_rel_power_r[0]:.2%}', fontsize=12)  # Adjust the position (5, 0.035) as needed
        plt.text(5, 0.032, f'Delta: {delta_rel_power_r[0]:.2%}', fontsize=12)  # Adjust the position (5, 0.032) as needed
        f_scale = np.mean(freqs_r, axis=0)
        plt.plot(f_scale, np.mean(psd_r, axis=0), lw=2, color='k')
        plt.fill_between(f_scale, np.mean(psd_r, axis=0), where=idx_slow, color='C1', alpha=0.3)
        plt.fill_between(f_scale, np.mean(psd_r, axis=0), where=idx_delta, color='skyblue')
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Power spectral density (uV^2 / Hz)', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim([x_limit1, x_limit2])
        plt.ylim([y_limit1, y_limit2])  # plt.ylim([0, np.max(psd_r) * 1.1])
        plt.title("Original", fontsize=14)
        plt.legend(["Mean Welch's periodogram", 'Slow Delta Band [0.5-2]Hz', 'Fast Delta Band [2-4]Hz'])

        # Plot the PSD for synthetic data
        plt.subplot(122)
        plt.text(5, 0.035, f'Slow: {slow_rel_power_s[0]:.2%}', fontsize=12)  # Adjust the position (5, 0.035) as needed
        plt.text(5, 0.032, f'Delta: {delta_rel_power_s[0]:.2%}', fontsize=12)  # Adjust the position (5, 0.032) as needed
        f_scale = np.mean(freqs_s, axis=0)
        plt.plot(f_scale, np.mean(psd_s, axis=0), lw=2, color='k')
        plt.fill_between(f_scale, np.mean(psd_s, axis=0), where=idx_slow, color='C1', alpha=0.3)
        plt.fill_between(f_scale, np.mean(psd_s, axis=0), where=idx_delta, color='skyblue')
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim([x_limit1, x_limit2])
        plt.ylim([y_limit1, y_limit2])  
        plt.title("Synthetic",fontsize=14)
        plt.legend(["Mean Welch's periodogram", 'Slow Delta Band [0.5-2]Hz', 'Fast Delta Band [2-4]Hz'])

        #plt.savefig('EEG_synthesiser_PSD.png')
        plt.show() 


    def plot_frequency_comparison(self,real_data,syntetic_data):
        """
        Plots a bar chart comparing the frequencies of two signals.

        Parameters:
        real_data (list or np.ndarray): Real input signals.
        synthetic_data (list or np.ndarray): Synthetic input signals.
        """ 
        labels = ['slow', 'delta','theta', 'alpha', 'beta']
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        # Compute relative power for real and synthetic data
        freqs_r, _, _, slow_rel_power_r, delta_rel_power_r, theta_rel_power_r, alpha_rel_power_r, beta_rel_power_r, _, _, _ = self.compute_relative_power(real_data, 'real')
        freqs_s, _, _, slow_rel_power_s, delta_rel_power_s, theta_rel_power_s, alpha_rel_power_s, beta_rel_power_s, _, _, _ = self.compute_relative_power(synthetic_data, 'synthetic')

       # Store the metrics for later use in histogram metrics
        self.real_metrics = {
            'slow': slow_rel_power_r,
            'delta': delta_rel_power_r,
            'theta': theta_rel_power_r,
            'alpha': alpha_rel_power_r,
            'beta': beta_rel_power_r
        }

        self.synthetic_metrics = {
            'slow': slow_rel_power_s,
            'delta': delta_rel_power_s,
            'theta': theta_rel_power_s,
            'alpha': alpha_rel_power_s,
            'beta': beta_rel_power_s
        }

        # Calculate mean and standard deviation for each band
        mean_r = [np.mean(slow_rel_power_r), np.mean(delta_rel_power_r), np.mean(theta_rel_power_r), np.mean(alpha_rel_power_r), np.mean(beta_rel_power_r)]
        std_r = [np.std(slow_rel_power_r), np.std(delta_rel_power_r), np.std(theta_rel_power_r), np.std(alpha_rel_power_r), np.std(beta_rel_power_r)]
        mean_s = [np.mean(slow_rel_power_s), np.mean(delta_rel_power_s), np.mean(theta_rel_power_s), np.mean(alpha_rel_power_s), np.mean(beta_rel_power_s)]
        std_s = [np.std(slow_rel_power_s), np.std(delta_rel_power_s), np.std(theta_rel_power_s), np.std(alpha_rel_power_s), np.std(beta_rel_power_s)]

        # Plot the bar chart comparing the frequency bands
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, mean_r, width, yerr=std_r, capsize=10, label='Real', color='c', alpha=0.7)
        rects2 = ax.bar(x + width/2, mean_s, width, yerr=std_s, capsize=10, label='Synthetic', color='black', alpha=0.7)

        # Add labels, title, and legend
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Frequency Band')
        ax.set_title('Frequency distribution comparison between real and synthetic signals')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.tight_layout()
        #plt.savefig('frequency_comparison.png')
        plt.show()

    def print_histogram_metrics(self, data, label):
        """
        Prints statistical metrics for the given data.

        Parameters:
        data (list or np.ndarray): Input signals to analyze.
        label (str): Label to identify the data type (e.g., 'real', 'synthetic').
        """
        # Check if the input is a list of signals or a single signal
        if isinstance(data[0], (list, np.ndarray)):
            data_combined = np.concatenate(data)
        else:
            data_combined = data

        # Compute statistical metrics
        mean = np.mean(data_combined)
        median = np.median(data_combined)
        mode_result = mode(data_combined, axis=None)
    
       # Check if mode_result is an array and has elements
        try:
            mode_value = mode_result.mode[0]
        except (IndexError, TypeError):
        mode_value = "undefined"
        data_range = np.ptp(data_combined)
        variance = np.var(data_combined)
        std_dev = np.std(data_combined)
        interquartile_range = iqr(data_combined)
        data_skewness = skew(data_combined)
        data_kurtosis = kurtosis(data_combined)

        # Print the metrics
        print(f"Metrics for {label} data:")
        print(f"Mean: {mean}")
        print(f"Median: {median}")
        print(f"Mode: {mode_value}")
        print(f"Range: {data_range}")
        print(f"Variance: {variance}")
        print(f"Standard Deviation: {std_dev}")
        print(f"Interquartile Range: {interquartile_range}")
        print(f"Skewness: {data_skewness}")
        print(f"Kurtosis: {data_kurtosis}")
        print("")

# Usage example

# Initialize the FrequencyAnalysis class
fa = FrequencyAnalysis(fs=2048)

# Compute relative power for real and synthetic data
fa.compute_relative_power(real_data, 'real')
fa.compute_relative_power(synthetic_data, 'synthetic')

# Compute relative power for real and synthetic data - one sample
fa.compute_relative_power(synthetic_data[0], 'synthetic')

# Plot power spectral density
fa.plot_psd(real_data, synthetic_data)

# Plot power spectral density - one sample
fa.plot_psd(real_data[0], synthetic_data[0])

# Plot frequency comparison
fa.plot_frequency_comparison(real_data, synthetic_data)

print('list of signals')
# Print histogram metrics for real data - list of signals
fa.print_histogram_metrics(real_data,'real')

# Plot power spectral density - one sample
fa.plot_frequency_comparison(real_data[0], synthetic_data[0])

print('1 signal')

# Print histogram metrics for real data - one sample
fa.print_histogram_metrics(real_data[0],'real')



