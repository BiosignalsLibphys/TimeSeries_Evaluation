import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from QualityMetrics import *
import copy
from SignalDictBuilder import structure_data

# Importing data #
path = '/Users/Asus/AISYM4MED_1/UMC_data/UMC_data'
data = structure_data(path, 'classifier')


#### Colormap functions ####
# Colormap with predefined levels of quality for both ECG and EEG data #
def quality_colormap(values, boundaries, name, metric, color):
    """
    -----
    Brief
    -----
    Plots the quality color map of the dataset based on the values list.
    ----------
    Parameters
    ----------
    values : list
         Classification of the signals according to the chosen metric.
    boundaries : list
        The boundary values of the metric.
    name : string
        Name of the metric that should appear in the plot.
    metric : int
        Discrete (0) or continuous (1) colorbar.
    color : list
        Strings with the colors to use in the map.
    """
    # Color mapping
    cmap = LinearSegmentedColormap.from_list(name, color)
    if metric == 0:  # The metric is discrete
        norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    else:  # The metric is continuous
        norm = plt.Normalize(np.min(boundaries), np.max(boundaries))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Determine the number of rows needed
    num_rows = (len(values) + 9) // 10

    # Create figure with subplots
    fig, axs = plt.subplots(num_rows, 1, figsize=(10, 1.5 * num_rows), constrained_layout=True)
    if num_rows == 1:
        axs = [axs]
    # Plotting the colorbars
    for i, ax in enumerate(axs):
        start_index = i * 10
        end_index = min((i + 1) * 10, len(values))
        x_values = range(1, 11)

        heights = [0.4] * (end_index - start_index) + [0] * (10 - (end_index - start_index))

        bars = ax.bar(x_values[:end_index - start_index], heights[:end_index - start_index],
                      color=cmap(norm(values[start_index:end_index])),
                      edgecolor='white', width=0.4)

        ax.set_xticks(x_values)
        ax.set_xticklabels(
            [f"Signal {j + 1}" for j in range(start_index, end_index)] + [''] * (10 - (end_index - start_index)))
        ax.set_ylim(0, 1)
        ax.set_yticks([])

    # Colorbar customization
    colorbar = fig.colorbar(sm, ax=axs, orientation='horizontal', pad=0.05, aspect=40)
    colorbar.set_label(name, fontweight='bold')

    plt.show()


# colormap to visualize signal quality as bad or good #
def binary_colormap(mask, title):
    """
    -----
    Brief
    -----
    Plots a colormap with green rectangles representing 1 in the provided mask and red rectangles are the 0 in mask.
    ----------
    Parameters
    ----------
    mask : nd-array
        Array of 0's and 1's representing if the signal follows the quality restrictions.
    title : string
        Plot title to be printed in the figure.
    """

    # Define colors for the plot: green for included, red for excluded
    colors = ['green' if m >= 1 else 'red' for m in mask]

    # Determine the number of rows needed for plotting
    num_rows = (len(mask) + 9) // 10

    # Create figure with subplots
    fig, axs = plt.subplots(num_rows, 1, figsize=(10, 1.5 * num_rows), constrained_layout=True)
    if num_rows == 1:
        axs = [axs]

    fig.suptitle(title, fontsize=16)
    # Plotting the colorbars
    for i, ax in enumerate(axs):
        start_index = i * 10
        end_index = min((i + 1) * 10, len(mask))
        x_values = range(1, 11)

        heights = [0.4] * (end_index - start_index) + [0] * (10 - (end_index - start_index))

        ax.bar(x_values[:end_index - start_index], heights[:end_index - start_index],
               color=colors[start_index:end_index], edgecolor='white', width=0.4)

        ax.set_xticks(x_values)
        ax.set_xticklabels(
            [f"Signal {j + 1}" for j in range(start_index, end_index)] + [''] * (10 - (end_index - start_index)))
        ax.set_ylim(0, 1)
        ax.set_yticks([])

    plt.show()


######## QCoD map ########
def noise_classify(signal, fs, signal_type='EEG'):
    """
    -----
    Brief
    -----
    Classification of the signals according to their level of white noise contamination,
    using QCod from QualityMetrics.
    ----------
    Parameters
    ----------
    signal : nd-array
        Array with the signals to be analyzed.
    fs : int
        Sampling frequency.
    signal_type : str, optional
        Type of signal being analyzed. Default is 'EEG'.

    Returns:
    -------
    results : int
        Classification for the signal:
        Level 4: High quality signal
        Level 3: Good quality signal
        Level 2: Medium quality signal
        Level 1: Bad quality signal
        Level 0: No quality signal
    """
    # Define thresholds for different signal types
    thresholds_dict = {'eeg': [0.3, 0.1, 0.06, 0.04, 0.03],
                       'ecg': [0.98, 0.9, 0.57, 0.37]}
    # Checking which signal is being analysed to retrieve its quality thresholds.
    thresholds = thresholds_dict.get(signal_type.lower(), [])
    num_levels = len(thresholds)

    masks = []
    # Determination of the signal quality from the multiple threshold levels.
    for threshold in thresholds:
        mask, _, _ = QCod(fs, threshold, data)
        masks.append(mask)

    # Determine result
    if all(mask == 0 for mask in masks):
        results = 0  # No quality signal
    elif masks[0] == 1:
        results = num_levels  # High quality signal
    else:
        results = num_levels - masks.index(1)

    return results


### Completness ###
def completeness_classify(signal):
    """
    -----
    Brief
    -----
    Classification of the signals according to their percentage of completeness.
    ----------
    Parameters
    ----------
    signal : nd-array or list
        The input signal.

    Returns:
    -------
    values : float
        Value of the signals' completeness in percentage.
        Level 1: Bad Quality, less than 80% of unique values.
        Level 2: Questionable quality, between 80 and 85% of unique values.
        Level 3: Medium quality, between 85 and 90% of unique values.
        Level 4: Good quality, between 90 and 95% of unique values.
        Level 5: Excellent quality, between 95-100%.
    """
    values = completeness(signal)
    return values


######## Uniqueness metrics ########
def uniqueness_classify(signal):
    """
    -----
    Brief
    -----
    Computes the uniqueness percentage.
    ----------
    Parameters
    ----------
    signal : nd-array or list
        The input signal.

    Returns:
    -------
    values : float
        Value of signals' uniqueness in percentage.
        Uniqueness levels
        Level 1: Bad Quality, less than 70% of unique values.
        Level 2: Questionable quality, between 70 and 80 % of unique values.
        Level 3: Medium quality, between 80 and 90% of unique values.
        Level 4: Good quality, between 90 and 95% of unique values.
        Level 5: Excellent quality, between 95-100%.
    """
    values = uniqueness(signal)
    return values


##### Hurst Exponent analysis #####
def hurst_classify(signal):
    """
    -----
    Brief
    -----
    Computes the Hurst exponent.
    H < 0.5 : Blue noise
    H = 0.5 : White noise, uncorrelated
    H = 1: 1/f noise, pink noise
    H > 1: red noise, non-stationary
    H = 1.5 : Brown noise
    ----------
    Parameters
    ----------
    signal : nd-array or list
        array with the timeseries values.

    Returns:
    -------
    values : float
        the Hurst exponent.
    """

    values = hurst_exponent(signal)
    return values


##### amplitude ######
def classify_amplitude(signal, signal_type='EEG'):
    """
    -----
    Brief
    -----
    Computes EEG quality based on its amplitude.
    ----------
    Parameters
    ----------
    signal : 1d-array or list
        array with the signal to be analyzed.
    signal_type : string
        If the signal is an 'EEG' or 'ECG'.

    Returns:
    -------
    c : int
        Classification of the signal according to its amplitude.
        The classification for EEG is based on amplitude thresholds in microvolts (uV):
        https://www.mdpi.com/1424-8220/19/3/601#B6-sensors-19-00601
        Level 4 - Excellent Quality if amplitude <= 100 uV.
        Level 3 - Good Quality if amplitude is between 100 and 200 uV.
        Level 2 - Questionable Quality if amplitude is between 200 and 300 uV.
        Level 1 - Poor Quality if amplitude >= 300 uV.

        The classification for ECG is based on amplitude thresholds in millivolts (mV):
        Level 4 - Excellent Quality if amplitude <= 5 mV.
        Level 3 - Good Quality if amplitude is between 5 and 10 mV.
        Level 2 - Moderate Quality if amplitude is between 10 and 15 mV.
        Level 1 - Poor Quality if amplitude > 15 mV.
    """

    # Type of signal has different treshold
    if signal_type.lower() == 'eeg':
        thresholds = [100, 200, 300]
    if signal_type == 'ecg':
        thresholds = [5, 10, 15]

    # List to save classification results
    max_amplitude = amplitude(signal)

    # Sort the thresholds to ensure they are in ascending order
    thresholds.sort()

    # Initialize classification value and description
    c = 0

    # Determine classification based on thresholds
    for i, threshold in enumerate(thresholds):
        if max_amplitude <= threshold:
            c = len(thresholds) - i
            break
    else:
        # If max_amplitude is greater than all thresholds
        c = 1

    return c


##### PCA noise analysis #####
def pca_classify(signal, sampling_rate, signal_type):
    """
    -----
    Brief
    -----
    Applies Principal Component Analysis (PCA) to the data, calculates the Area Under the Curve (AUC)
    of the cumulative variance explained by the principal components, and classifies the signal quality
    in terms of white noise and based on the relative AUC.

    ----------
    Parameters
    ----------
    signal : 1d-array or list
        array with the signal to be analyzed.
    sampling_rate : int
        The sampling rate of the signal in Hz.
    signal_type : string
        The type of signal, either 'EEG' or 'ECG'.

    Returns:
    -------
    c : int
        classification of the signal according to its number of PCA components vs accumulated variance.
        EEG:
        https://www.mdpi.com/1424-8220/19/3/601#B6-sensors-19-00601
        Level 4 - Excellent Quality if rel_pca >= 0.9.
        Level 3 - Good Quality if rel_pca is between 0.9 and 0.7519.
        Level 2 - Questionable Quality if rel_pca is between 0.7519 and 0.6211.
        Level 1 - Poor Quality' if rel_pca <= 0.6211.

        ECG:
        Level 4 - Excellent Quality if rel_pca >= 0.8679.
        Level 3 - Good Quality if rel_pca is between 0.8679 and 0.7519.
        Level 2 - Questionable Quality if rel_pca is between 0.7519 and 0.60.
        Level 1 - Poor Quality if rel_pca <= 0.60.
    """
    rel_PCA_AUC = pca_and_auc(signal, sampling_rate)
    thresholds = [] # avoiding thresholds referenced before assignment warning

    # Classification based on PCA relative AUC
    # Define thresholds for classification based on signal type
    if signal_type.lower() == 'eeg':
        thresholds = [0.90, 0.7519, 0.6211]
    elif signal_type.lower() == 'ecg':
        thresholds = [0.8679, 0.7698, 0.60]

    # Determine classification based on thresholds
    for i, threshold in enumerate(thresholds):
        if rel_PCA_AUC > threshold:
            c = len(thresholds) - i
            break
    else:
        c = 1
    return c


#### SNR ####
def snr_classify(signal):
    """
    -----
    Brief
    -----
    Classifies EEG and ECG quality based on their SNR, computed with calculate_snr from QualityMetrics.
    ----------
    Parameters
    ----------
    signal : nd-array
        array with the signal to be analyzed.
    show : boolean
        True if the dataset color map according to this metric is to be displayed. Default: False

    Returns:
    -------
    values : list
        The classification of the signal according to its SNR.
        ref: https://www.mdpi.com/1424-8220/19/3/601#B6-sensors-19-00601
        Level 4 - Excellent Quality. if SNR is more than 5 dB .
        Level 3 - Good Quality. if SNR is between 1 and 5 dB.
        Level 2 - Questionable Quality. if SNR is between -5 and 1 dB.
        Level 1 - Poor Quality. if SNR is less than -5 dB.
    """

    snr = calculate_snr(signal)
    # Categorize SNR into four levels
    if snr > 5:
        quality = 'Level 4 - Excellent Quality (> 5 dB)'
        values = 4
    elif 5 > snr > 1:
        quality = 'Level 3 - Good Quality (1 to 5 dB)'
        values = 3
    elif 1 > snr > -5:
        quality = 'Level 2 - Moderate Quality (-5 to 1 dB)'
        values = 2
    else:
        quality = 'Level 1 - Poor Quality (≤ -5 dB)'
        values = 1

    return values


# Classification based on the length of the saturation segments
def saturation_classify(signal, sampling_rate, signal_type):
    """
    -----
    Brief
    -----
    Evaluates the saturation level of a signal by determining the duration
    for which the signal remains at its maximum amplitude, using the saturation function from QualityMetrics.
    ----------
    Parameters
    ----------
    signal : 1d-array or list
        The input signal data.
    sampling_rate : int
        The sampling rate of the signal in Hz.
    signal_type : string
        What signal is being analyzed, 'EEG' or 'ECG'.

    Returns:
    -------
    c: int
        The classification is based on the total duration of saturation:
        Level 4 - Excellent Quality. if saturation is less than 25% of the signal duration.
        Level 3 - Good Quality. if saturation is between 25% and 50% of the signal duration.
        Level 2 - Questionable Quality. if saturation is between 50% and 75% of the signal duration.
        Level 1 - Poor Quality. if saturation is more than 75% of the signal duration.
    """

    if signal_type.lower() == 'eeg':
        # Thresholds for 50 Hz noise amplitude in EEG
        th = 1e-6

    elif signal_type.lower() == 'ecg':
        # Thresholds for 50 Hz noise amplitude in ECG
        th = 15

    total_saturation_duration = saturation(signal, sampling_rate, th)
    signal_duration = len(signal) / sampling_rate
    if total_saturation_duration < signal_duration / 4:
        c = 4
    elif signal_duration / 4 <= total_saturation_duration < signal_duration / 2:
        c = 3
    elif signal_duration / 2 <= total_saturation_duration < (signal_duration * 3) / 4:
        c = 2
    else:
        c = 1

    return c


def power_line_classify(signal, sampling_rate, signal_type):
    """
    -----
    Brief
    -----
    Classifies the signal quality based on the amplitude of 50 Hz power line noise.
    It uses the power_line_noise from Quality metrics to determine this measurement.
    ----------
    Parameters
    ----------
    signal : numpy.ndarray
        The input signal data.
    sampling_rate : int
        The sampling rate of the signal in Hz.
    signal_type : string
        The type of signal, either 'EEG' or 'ECG'.

    Returns :
    -------
    c :
        Classification of the signal, according to its powerline interference content.


    The classification for EEG is based on 50 Hz noise amplitude thresholds*:
    - 'Level 4 - Excellent Quality' if amplitude <= 0.843.
    - 'Level 3 - Good Quality' if amplitude is between 0.843 and 1.325.
    - 'Level 2 - Questionable Quality' if amplitude is between 1.325 and 2.660.
    - 'Level 1 - Poor Quality' if amplitude > 2.660.

    The classification for ECG is based on 50 Hz noise amplitude thresholds*:
    - 'Level 4 - Excellent Quality' if amplitude <= 0.189.
    - 'Level 3 - Good Quality' if amplitude is between 0.189 and 0.469.
    - 'Level 2 - Questionable Quality' if amplitude is between 0.469 and 1.055.
    - 'Level 1 - Poor Quality' if amplitude > 1.055.

    *These thresholds were defined based on SNR quality levels for each signal.
    """
    amplitude_50hz = power_line_noise(signal, sampling_rate)
    # Set thresholds based on signal type
    if signal_type.lower() == 'eeg':
        # Thresholds for 50 Hz noise amplitude in EEG
        thresholds = [0.843, 1.325, 2.660]

    elif signal_type.lower() == 'ecg':
        # Thresholds for 50 Hz noise amplitude in ECG
        thresholds = [0.189, 0.469, 1.055]

    # Classify the quality based on 50 Hz noise amplitude
    # Initialize classification value
    c = 0

    # Determine classification based on thresholds
    for i, threshold in enumerate(thresholds):
        if amplitude_50hz <= threshold:
            c = len(thresholds) - i
            break
    else:
        c = 1
    return c


##### Binarize Data #####
def binarize(values, lower_bound):
    """
    -----
    Brief :
    -----
    Changes the results to a binary mask, according to the defined lower_bound.
    ----------
    Parameters :
    ----------
    values : nd-array or list or int or float
        Array, list, or single value with the results with which to create a mask.
    lower_bound : int or float
        Minimum value of the results to be accepted in the mask.

    Returns :
    -------
    mask : list
         Mask of the signals that will be included: 1: included, 0: not included.
    """
    mask = None  # Initialize mask to avoid the 'mask is referenced before assignment'.

    # Checking what is the type of input
    if isinstance(values, (np.ndarray, list)):
        values = np.array(values)  # Ensure values is an np.ndarray
        mask = (values >= lower_bound).astype(int).tolist()
    elif isinstance(values, (int, float, np.number)):
        mask = [1] if values >= lower_bound else [0]
    else:
        raise TypeError("Unsupported input type. 'values' must be an ndarray, list, int, or float.")

    return mask


##### Application of functions to every dictionary #####
def apply_function_to_timeseries(signal, func, *args, **kwargs):
    """
    -----
    Brief
    -----
    Applies a function to each element of a data structure recursively.
    ----------
    Parameters
    ----------
    signal : dict or list or numpy.ndarray
        Input data structure to be processed.
    func : function
        Function to be applied to each element of the data structure.
    *args : positional arguments, optional
        Additional positional arguments to be passed to the function.
    **kwargs : keyword arguments, optional
        Additional keyword arguments to be passed to the function.

    Returns
    -------
    processed_data : dict or list or numpy.ndarray
        Processed data structure after applying the function to each element.
    """
    if isinstance(signal, dict):
        # Recurse into the dictionary
        return {key: apply_function_to_timeseries(value, func, *args, **kwargs) for key, value in signal.items()}
    elif isinstance(signal, (list, np.ndarray)):
        # Apply the function to each element if it's a list or array
        if all(isinstance(x, (int, float, np.number)) for x in signal):
            # It's a numerical array, apply func with additional arguments
            return func(signal, *args, **kwargs)
        else:
            # Recurse into each item of the list/array
            return [apply_function_to_timeseries(item, func, *args, **kwargs) for item in signal]
    else:
        # Return the data unchanged if it's not a list/array or dictionary
        return signal


##### Plotting the colormaps for the dictionaries #####
def results_plot(results, b, n, m, c):
    """
    -----
    Brief
    -----
    Auxiliary function that plots the colormaps with the different quality levels for each element
    within the data structure.
    Uses the quality_colormap function.
    ----------
    Parameters
    ----------
    results : dict
        Dictionary with the results from the quality metrics.
    b : list
        The boundary values of the metric.
    n : string
        Name of the metric that should appear in the plot.
    m : int
        Discrete (0) or continuous (1) colorbar.
    c : list
        Strings with the colors to use in the map.
    """
    if isinstance(results, dict):
        single_values = []
        for key, signals in results.items():
            # Checking if the list or array has more than one element
            if isinstance(signals, (list, np.ndarray)) and len(signals) > 1:
                quality_colormap(signals, b, n, m, c)
            # Checking if there are nested lists or arrays
            elif isinstance(signals, (list, np.ndarray)):
                # iterate over the values of the nested list/array
                for item in signals:
                    if isinstance(item, (list, np.ndarray)) and len(item) > 1:
                        # Plot the colormap with the key name in the title
                        quality_colormap(item, b, n + ' ' + str(key), m, c)
            # Checking if each key has only one value, so they can be plotted all together
            elif isinstance(signals, (float, int, np.number)):
                single_values.append(signals)

        if single_values:
            quality_colormap(single_values, b, n, m, c)
    else:
        print('The results parameter must be a dictionary.')


### Visualization of all the maps ###
def metric_map_visualizer(data_dict, sr, signal_type):
    """
    -----
    Brief
    -----
    Auxiliary function that plots the colormaps for all the metrics at the pre-defined quality levels.
    ----------
    Parameters
    ----------
    data_dict : dict
        Dictionary with the database.
    sr : int
        The sampling rate in Hz
    n : string
        Name of the metric that should appear in the plot.
    signal_type : string
        Which signal is being analysed 'EEG' or 'ECG'.
    """
    # Computing all the metrics for the data dictionary
    # QCOD_dict = apply_function_to_timeseries(dict, uniqueness_classify, sr = sr, type = signal)
    completeness_dict = apply_function_to_timeseries(data_dict, completeness_classify)
    uniqueness_dict = apply_function_to_timeseries(data_dict, uniqueness_classify)
    # hurst_dict = apply_function_to_timeseries(dict, hurst_classify)
    amplitude_dict = apply_function_to_timeseries(data_dict, classify_amplitude, type=signal_type)
    # pca_dict = apply_function_to_timeseries(dict, pca_classify, sampling_rate = sr, type = signal)
    snr_dict = apply_function_to_timeseries(data_dict, snr_classify)
    sat_dict = apply_function_to_timeseries(data_dict, saturation_classify, sampling_rate=sr, type=signal_type)
    pwl_dict = apply_function_to_timeseries(data_dict, power_line_classify, sampling_rate=sr, type=signal_type)

    # Plotting the results
    # results_plot(QCOD_dict, [0,4],'QCOD',1,["red", "yellow", "green"])
    results_plot(completeness_dict, [0, 80, 85, 90, 95, 100], 'Completness (%) ', 0, ["red", "yellow", "green"])
    results_plot(uniqueness_dict, [0, 70, 80, 90, 95, 100], 'Uniqueness (%) ', 0, ["red", "yellow", "green"])
    # results_plot(hurst_dict[0, 1.5], 'Hurst Exponent', 1, ["blue", "white", "red", "brown"])
    results_plot(amplitude_dict, [1, 4], 'Amplitude Quality ', 1, ["red", "yellow", "green"])
    # results_plot(pca_dict, [1, 4], 'PCA Quality', 1, ["red", "yellow", "green"])
    results_plot(snr_dict, [1, 4], 'SNR Quality ', 1, ["red", "yellow", "green"])
    results_plot(sat_dict, [1, 4], 'Saturation Quality ', 1, ["red", "yellow", "green"])
    results_plot(pwl_dict, [1, 4], 'Powerline noise Quality ', 1, ["red", "yellow", "green"])


#### Combining the masks that will have the same structure as the input data ####
def combine_nested_masks(masks):
    """
    -----
    Brief:
    -----
    Combine nested dictionary masks using bitwise AND operation recursively.
    ----------
    Parameters:
    ----------
    masks : list
        Dictionaries containing masks with values of 1's and 0's.

    Returns:
    -------
    combine_dicts : dict
        Dictionary with the combination of all inputed masks.
    """

    def combine_values(value1, value2):
        # Checking if masks to combine are dictionaries
        if isinstance(value1, dict) and isinstance(value2, dict):
            return combine_dicts([value1, value2])
        # Checking if the masks are lists
        elif isinstance(value1, list) and isinstance(value2, list):
            return combine_lists([value1, value2])
        else:
            return value1 & value2

    def combine_dicts(dicts):
        # Function to combine dictionaries
        result = copy.deepcopy(dicts[0])
        for d in dicts[1:]:
            if d is not None:
                for key, value in d.items():
                    if key in result:
                        result[key] = combine_values(result[key], value)
                    else:
                        result[key] = copy.deepcopy(value)
        return result

    def combine_lists(lists):
        # Function to combine lists
        result = copy.deepcopy(lists[0])
        for lst in lists[1:]:
            if lst is not None:
                for i in range(len(lst)):
                    if i < len(result):
                        result[i] = combine_values(result[i], lst[i])
                    else:
                        result.append(copy.deepcopy(lst[i]))
        return result

    return combine_dicts(masks)


#### Flatten the masks, so they can be plotted ####
def flatten_mask(mask):
    """
    -----
    Brief:
    -----
    Flattens a nested list of mask values into a single list of integers (0 and 1).
    Auxiliary function to plot the binary quality colormaps.
    ----------
    Parameters:
    ----------
    masks : list
        Dictionaries containing masks with values of 1's and 0's.

    Returns:
    -------
    flattened : list
       flattened mask values.
    """
    flattened = []
    for item in mask:
        if isinstance(item, list):
            # If the item is a list, flatten it recursively
            flattened.extend(flatten_mask(item))
        else:
            # If the item is an integer (0 or 1), append it directly
            flattened.append(item)
    return flattened


#### Final colormap ####
def plot_binary(results):
    """
    -----
    Brief
    -----
    Auxiliary function that plots the binary colormaps with the specified quality levels for each element
    within the data structure.
    Uses the binary_colormap function.
    ----------
    Parameters
    ----------
    results : dict
        Dictionary with the nested masks from the chosen quality metrics.
    """
    if isinstance(results, dict):
        single_values = []
        for key, signals in results.items():
            # Checking if the list or array has more than one element
            if isinstance(signals, (list, np.ndarray)) and len(signals) > 1:
                signals_f = flatten_mask(signals)
                binary_colormap(signals_f, key)
            # Checking if there are nested lists or arrays
            elif isinstance(signals, (list, np.ndarray)):
                # iterate over the values of the nested list/array
                for item in signals:
                    if isinstance(item, (list, np.ndarray)) and len(item) > 1:
                        signals_f = flatten_mask(item)
                        binary_colormap(signals_f, key)
            # Checking if each key has only one value, so they can be plotted all together
            elif isinstance(signals, (float, int, np.number)):
                single_values.append(signals)

        if single_values:
            binary_colormap(single_values, ' ')
    else:
        print('The results parameter must be a dictionary.')



#### Quality maps for the whole dataset and chosen metrics with specified levels of quality ####
def dummy_quality(dataset, metrics={'QCOD': 4, 'Completeness': 95, 'Uniqueness': 95, 'Hurst': 0.9,
                                    'SNR': 4}, signal_type='EEG', fs=2048):
    """
    -----
    Brief
    -----
    Classifies dataset quality based on the defined metrics and thresholds to be applied to the signals.
    It also plots the colormap of the quality mask.
    ----------
    Parameters
    ----------
    dataset : dict
        Dictionary with the signals to be analyzed.
    metrics : dict
        Dictionary with the metric names as keys and the selected thresholds as values. The default is maximum quality
        level for all metrics.
    signal_type : string
        Which signal is being analysed 'EEG' or 'ECG'.
    fs: float
        The signal's sampling frequency in Hz.

    Returns:
    -------
    final_mask : nd-array
        Array with the quality mask according to the specified metrics and quality level. 1 (green): the signal has enough
        quality according to the chosen parameters. 0 (red): the signal does not meet the specified quality standards.
    """
    qcod_mask = None
    c_mask = None
    u_mask = None
    h_mask = None
    a_mask = None
    pca_mask = None
    snr_mask = None
    sat_mask = None
    pwl_mask = None

    if 'QCOD' in metrics.keys():
        qcod = apply_function_to_timeseries(dataset, noise_classify, fs=fs, type=signal_type)
        qcod_mask = apply_function_to_timeseries(qcod, binarize, lower_bound=metrics['QCOD'])
        print('QCOD')

    if 'Completeness' in metrics.keys():
        c = apply_function_to_timeseries(dataset, completeness_classify)
        c_mask = apply_function_to_timeseries(c, binarize, lower_bound=metrics['Completeness'])
        print('Completeness')

    if 'Uniqueness' in metrics.keys():
        u = apply_function_to_timeseries(dataset, uniqueness_classify)
        u_mask = apply_function_to_timeseries(u, binarize, lower_bound=metrics['Uniqueness'])
        print('Uniqueness')

    if 'Hurst' in metrics.keys():
        h = apply_function_to_timeseries(dataset, hurst_classify)
        h_mask = apply_function_to_timeseries(h, binarize, lower_bound=metrics['Hurst'])
        print('Hurst')

    if 'Amplitude' in metrics.keys():
        a = apply_function_to_timeseries(dataset, classify_amplitude, type=signal_type)
        a_mask = apply_function_to_timeseries(a, binarize, lower_bound=metrics['Amplitude'])
        print('Amplitude')

    if 'PCA' in metrics.keys():
        pca = apply_function_to_timeseries(dataset, pca_classify, sampling_rate=fs, type=signal_type)
        pca_mask = apply_function_to_timeseries(pca, binarize, lower_bound=metrics['PCA'])
        print('PCA')

    if 'SNR' in metrics.keys():
        snr = apply_function_to_timeseries(dataset, snr_classify)
        snr_mask = apply_function_to_timeseries(snr, binarize, lower_bound=metrics['SNR'])
        print('SNR')

    if 'Saturation' in metrics.keys():
        sat = apply_function_to_timeseries(dataset, saturation_classify, sampling_rate=fs, type=signal_type)
        sat_mask = apply_function_to_timeseries(sat, binarize, lower_bound=metrics['Saturation'])
        print('Saturation')

    if 'Powerline' in metrics.keys():
        pwl = apply_function_to_timeseries(dataset, power_line_classify, sampling_rate=fs, type=signal_type)
        pwl_mask = apply_function_to_timeseries(pwl, binarize, lower_bound=metrics['Powerline'])
        print('Powerline Noise')

    # Combine all masks
    print('start combining')
    masks = [qcod_mask, c_mask, u_mask, h_mask, a_mask, pca_mask, snr_mask, sat_mask, pwl_mask]
    # Combining only not None masks
    masks = [m for m in masks if m is not None]
    final_mask = combine_nested_masks(masks)
    # Plotting the combined masks
    plot_binary(final_mask)
    return final_mask

### Checking the levels of quality for one metrics ###
result_dict = apply_function_to_timeseries(data, power_line_classify, sampling_rate=2048, type='EEG')
result_dict2 = apply_function_to_timeseries(data, saturation_classify, sampling_rate=2048, type='EEG')
results_plot(result_dict, [1, 4], 'Powerline Noise', 1, ["red", "yellow", "green"])

### Checking the levels of quality for all metrics ###
metric_map_visualizer(data, 2048, 'EEG')

## Testing for a dictionary that would only have one value per key ##
file_path = 'synthetic_data.pkl'
data1 = pd.read_pickle(file_path)
data1 = np.array(data1)

### Testing for one value per key in the data dictionary ###
dict1 = {}
for i in range(0, len(data1)):
    dict1[i] = data1[i]

result_dict1 = apply_function_to_timeseries(dict1, uniqueness_classify)
## Plotting results for this case ##
results_plot(result_dict1, [70, 80, 90, 95, 100], 'Uniqueness (%)', 0, ["red", "yellow", "green"])

## Usage for 1 metric
b1 = apply_function_to_timeseries(result_dict, binarize, lower_bound=3)
plot_binary(b1)

## Usage for more than one metric
sigs = dummy_quality(data, metrics={'Completeness': 95, 'Uniqueness': 95, 'Powerline': 3}, fs=2048, signal_type='EEG')
