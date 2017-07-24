import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
import peakutils
from Utils.DataHandling.data_processing import pd_to_np


# input is peak indices (idx1, idx2) and signal value at peaks (sig1, sig2)
# Output is DataFrame in the with columns ['idx', 'maxsignal', 'sig1_minus_2'])
def merge_adjacent_peaks_from_two_signals(idx1, idx2, sig1=None, sig2=None, p_type='keep_max', win_size=10):
    # Keep first signal out of two adjacent peaks
    if p_type == 'keep_first_signal':
        l = [range(i - win_size, i + win_size) for i in idx1]
        idx = [item for sublist in l for item in sublist]
        idx_add = np.setdiff1d(idx2, idx)
        return np.unique(np.concatenate((np.array(idx1), idx_add)))

    # Or, keep the stronger of signal out of two adjacent peaks
    elif p_type == 'keep_max':
        peaks_raw = []
        for i in range(len(idx1)):
            for j in range(len(idx2)):
                if abs(idx1[i] - idx2[j]) <= win_size:
                    # Prevent division by zero
                    if sig1.iloc[i] == 0:
                        sig1.iloc[i] += 0.00001
                    if sig2.iloc[j] == 0:
                        sig2.iloc[j] += 0.00001
                    peaks_raw.append((idx1[i], idx2[j], sig1.iloc[i], sig2.iloc[j], sig1.iloc[i] - sig2.iloc[j]))

        # keep only max signal
        peaks = []
        for i in range(len(peaks_raw)):
            if peaks_raw[i][2] >= peaks_raw[i][3]:
                peaks.append([peaks_raw[i][0], peaks_raw[i][2], peaks_raw[i][4]])
            else:
                peaks.append([peaks_raw[i][1], peaks_raw[i][3], peaks_raw[i][4]])

        peaks = pd.DataFrame(peaks, columns=['idx', 'maxsignal', 'sig1_minus_2'])
        peaks = peaks.sort_values(by='idx')

        # remove duplicates
        remove = []
        for i in range(peaks.shape[0]):
            for j in range(i + 1, peaks.shape[0]):
                if peaks['idx'].iloc[i] == peaks['idx'].iloc[j]:
                    if peaks['maxsignal'].iloc[i] >= peaks['maxsignal'].iloc[j]:
                        remove.append(j)
                    else:
                        remove.append(i)
        if len(remove) > 0:
            peaks = peaks.drop(peaks.index[np.unique(remove)], axis=0)
        return peaks
    else:
        return


# Input is DataFrame containing two columns ['idx', 'maxsignal'] and optionally additional columns as 'sig1_minus_2'
# Output is list of indices of remaining peaks after merging adjacent peaks
def merge_adjacent_peaks_from_single_signal(peaks, win_size=20):
    peaks.sort_values(by='idx')
    i = 0
    while i < peaks.shape[0] - 1:
        if peaks['idx'].iloc[i + 1] - peaks['idx'].iloc[i] > win_size:
            i += 1
        elif peaks['maxsignal'].iloc[i + 1] <= peaks['maxsignal'].iloc[i]:
            peaks = peaks.drop(peaks.index[i + 1], axis=0)
        else:
            peaks = peaks.drop(peaks.index[i], axis=0)
    idx = peaks['idx'].tolist()
    return idx


# Input is DataFrame containing two columns ['idx', 'maxsignal'] and optionally additional columns as 'sig1_minus_2'
# Output the same DataFrame after the small peaks have been removed
def remove_small_peaks(peaks, num_std=2):
    mean = peaks['maxsignal'].mean()
    std = peaks['maxsignal'].std()
    remove = []
    for i in range(peaks.shape[0]):
        if peaks['maxsignal'].iloc[i] < (mean - num_std * std):
            remove.append(i)
    if len(remove) == 0:
        return peaks
    else:
        return peaks.drop(peaks.index[np.unique(remove)], axis=0)


def score_max_peak_within_fft_frequency_range(signal, sampling_freq, min_hz, max_hz, show=False):
    signal_mean_normalized = signal - np.mean(signal)
    a = np.power(np.abs(np.fft.fft(signal_mean_normalized)), 2)
    a = a / np.sum(a)

    f = sampling_freq
    num_t = a.shape[0]
    tick = f / 2.0 / num_t

    if show:
        x_label = np.asarray(range(num_t)) * tick
        plt.plot(x_label, a)

    relevant_start = int(round(min_hz / tick))
    relevant_end = int(round(max_hz / tick))

    score = np.max(a[relevant_start:relevant_end])
    return score


def run_peak_utils_peak_detection(signal, param1, param2):
    # Check for zero lengths signal or for flat signal
    if len(signal) == 0:
        return np.array([])
    if max(signal) == min(signal):
        return np.array([])

    # Set threshold value
    if max(signal) == 0:
        val = 1
    else:
        val = param1 / max(signal)

    # Run peakutils
    res = peakutils.indexes(signal, thres=val, min_dist=param2)
    return res


def run_scipy_peak_detection(signal, param1, param2):
    return sig.find_peaks_cwt(signal, np.arange(param1, param2))


def max_filter(x, win_size):
    a = pd_to_np(x)
    max_val = np.max(a)
    b = np.asarray([max_val] * win_size)
    res = np.convolve(a, b, 'same')
    return res
