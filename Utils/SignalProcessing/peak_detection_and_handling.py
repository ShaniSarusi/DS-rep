import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt
import peakutils
from Utils.DataHandling.data_processing import pd_to_np


def merge_peaks_from_two_signals(idx1, idx2, sig1=None, sig2=None, merge_type='union', intersect_win_size=10,
                                 union_min_dist=40, union_min_thresh=1):
    if len(idx1) == 0:
        return
    elif len(idx2) == 0:
        return
    elif sig1 is None:
        return
    elif sig2 is None:
        return

    sig1 = pd_to_np(sig1)
    sig2 = pd_to_np(sig2)

    # Sort signals
    a = np.argsort(idx1)
    idx1 = idx1[a]
    sig1 = sig1[a]
    b = np.argsort(idx2)
    idx2 = idx2[b]
    sig2 = sig2[b]

    # Intersect
    peaks = []
    merged_idx = []
    for i in range(len(idx1)):
        nearest = np.min(np.abs(idx2 - idx1[i]))
        if nearest <= intersect_win_size:
            j = np.argmin(np.abs(idx2 - idx1[i]))
            other_nearest = np.min(np.abs(idx1 - idx2[j]))
            if other_nearest < nearest:
                continue
            else:
                merged_idx.append(idx1[i])
                merged_idx.append(idx2[j])
                if sig1[i] >= sig2[j]:
                    peaks.append(idx1[i])
                else:
                    peaks.append(idx2[j])

    # Union
    if merge_type == 'union':
        merged_idx = np.asarray(merged_idx)
        idx_left_out = []
        sig_left_out = []
        for i in range(len(idx1)):
            if idx1[i] not in merged_idx:
                idx_left_out.append(idx1[i])
                sig_left_out.append(sig1[i])
        for j in range(len(idx2)):
            if idx2[j] not in merged_idx:
                idx_left_out.append(idx2[j])
                sig_left_out.append(sig2[j])

        for i in range(len(idx_left_out)):
            nearest = np.min(np.abs(merged_idx - idx_left_out[i]))
            if nearest > union_min_dist:
                if sig_left_out[i] >= union_min_thresh:
                    peaks.append(idx_left_out[i])

    return peaks



def merge_adjacent_peaks_from_two_signals_opt2(idx1, idx2, sig1=None, sig2=None, win_size=10, z=0.5):
    # input is peak indices (idx1, idx2) and signal value at peaks (sig1, sig2).
    # Output is DataFrame in the with columns ['idx', 'maxsignal', 'sig1_minus_2'])

    if len(idx1) == 0:
        return
    elif len(idx2) == 0:
        return
    elif sig1 is None:
        return
    elif sig2 is None:
        return

    sig1 = pd_to_np(sig1)
    sig2 = pd_to_np(sig2)

    # Sort signals
    a = np.argsort(idx1)
    idx1 = idx1[a]
    sig1 = sig1[a]
    b = np.argsort(idx2)
    idx2 = idx2[b]
    sig2 = sig2[b]

    cols = ['idx', 'maxsignal', 'sig1_minus_2']

    # Merge******
    adjacent = []
    k = 0
    for i in range(len(idx1)):
        for j in range(k, len(idx2)):
            if idx1[i] - idx2[j] > win_size:
                k = j
            if idx2[j] - idx1[i] > win_size:
                break
            if abs(idx1[i] - idx2[j]) <= win_size:
                # Prevent division by zero
                if sig1[i] == 0:
                    sig1[i] += 0.00001
                if sig2[j] == 0:
                    sig2[j] += 0.00001
                adjacent.append((idx1[i], idx2[j], sig1[i], sig2[j], sig1[i] - sig2[j]))
    if len(adjacent) == 0:
        return pd.DataFrame(columns=cols)

    # keep only max signal
    p_idx = [adjacent[i][0] if adjacent[i][2] >= adjacent[i][3] else adjacent[i][1] for i in
             range(len(adjacent))]
    p_max_sig = [adjacent[i][2] if adjacent[i][2] >= adjacent[i][3] else adjacent[i][3] for i in
                 range(len(adjacent))]
    p_sig_diff = [adjacent[i][4] for i in range(len(adjacent))]

    peaks = pd.DataFrame(list(zip(p_idx, p_max_sig, p_sig_diff)), columns=cols)
    peaks = peaks.sort_values(['idx', 'maxsignal'])
    peaks = peaks.drop_duplicates(keep='first')

    idx1_in = [adjacent[i][0] for i in range(len(adjacent))]
    idx2_in = [adjacent[i][1] for i in range(len(adjacent))]
    idx1_left_out = [(idx1[i], sig1[i]) for i in range(len(idx1)) if idx1[i] not in idx1_in]
    idx2_left_out = [(idx2[i], sig2[i]) for i in range(len(idx2)) if idx2[i] not in idx2_in]

    mean_max_signal = peaks['maxsignal'].mean()
    std_max_signal = peaks['maxsignal'].std()
    for idx1_left_out_i in idx1_left_out:
        if idx1_left_out_i[1] > (mean_max_signal + z * std_max_signal):
            peaks.loc[len(peaks)] = [idx1_left_out_i[0], idx1_left_out_i[1], 0]
    for idx2_left_out_i in idx2_left_out:
        if idx2_left_out_i[1] > (mean_max_signal + z * std_max_signal):
            peaks.loc[len(peaks)] = [idx2_left_out_i[0], idx2_left_out_i[1], 0]

    peaks = peaks.sort_values(['idx', 'maxsignal'])
    peaks = peaks.drop_duplicates(keep='first')

    return peaks


def merge_adjacent_peaks_from_two_signals(idx1, idx2, sig1=None, sig2=None, p_type='keep_max', win_size=10):
    # input is peak indices (idx1, idx2) and signal value at peaks (sig1, sig2).
    # Output is DataFrame in the with columns ['idx', 'maxsignal', 'sig1_minus_2'])

    if len(idx1) == 0:
        return
    elif len(idx2) == 0:
        return
    elif sig1 is None:
        return
    elif sig2 is None:
        return

    cols = ['idx', 'maxsignal', 'sig1_minus_2']
    # Keep first signal out of two adjacent peaks
    if p_type == 'keep_first_signal':
        l = [range(i - win_size, i + win_size) for i in idx1]
        idx = [item for sublist in l for item in sublist]
        idx_add = np.setdiff1d(idx2, idx)
        return np.unique(np.concatenate((np.array(idx1), idx_add)))

    # Or, keep the stronger of signal out of two adjacent peaks
    sig1 = pd_to_np(sig1)
    sig2 = pd_to_np(sig2)

    # Sort signals
    a = np.argsort(idx1)
    idx1 = idx1[a]
    sig1 = sig1[a]
    b = np.argsort(idx2)
    idx2 = idx2[b]
    sig2 = sig2[b]
    if p_type == 'keep_max':
        adjacent = []
        k = 0
        for i in range(len(idx1)):
            for j in range(k, len(idx2)):
                if idx1[i] - idx2[j] > win_size:
                    k = j
                if idx2[j] - idx1[i] > win_size:
                    break
                if abs(idx1[i] - idx2[j]) <= win_size:
                    # Prevent division by zero
                    if sig1[i] == 0:
                        sig1[i] += 0.00001
                    if sig2[j] == 0:
                        sig2[j] += 0.00001
                    adjacent.append((idx1[i], idx2[j], sig1[i], sig2[j], sig1[i] - sig2[j]))
        if len(adjacent) == 0:
            return pd.DataFrame(columns=cols)

        # keep only max signal
        p_idx = [adjacent[i][0] if adjacent[i][2] >= adjacent[i][3] else adjacent[i][1] for i in
                 range(len(adjacent))]
        p_max_sig = [adjacent[i][2] if adjacent[i][2] >= adjacent[i][3] else adjacent[i][3] for i in
                     range(len(adjacent))]
        p_sig_diff = [adjacent[i][4] for i in range(len(adjacent))]

        peaks = pd.DataFrame(list(zip(p_idx, p_max_sig, p_sig_diff)), columns=cols)
        peaks = peaks.sort_values(['idx', 'maxsignal'])
        peaks = peaks.drop_duplicates(keep='first')
        return peaks
    else:
        return


# Input is DataFrame containing two columns ['idx', 'maxsignal'] and optionally additional columns as 'sig1_minus_2'
# Output is list of indices of remaining peaks after merging adjacent peaks
def merge_adjacent_peaks_from_single_signal(peaks, win_size=20):
    if peaks is None:
        return []
    if peaks.shape[0] == 0:
        return []
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
    # if max(signal) == 0:
    #     val = 1
    # else:
    #     val = param1 / max(signal)

    # Run peakutils
    res = peakutils.indexes(signal, thres=param1, min_dist=param2)
    return res


def run_scipy_peak_detection(signal, param1, param2):
    return find_peaks_cwt(signal, np.arange(param1, param2))


def max_filter(x, win_size):
    a = pd_to_np(x)
    max_val = np.max(a)
    b = np.asarray([max_val] * win_size)
    res = np.convolve(a, b, 'same')
    return res
