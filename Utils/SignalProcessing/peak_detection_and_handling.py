import numpy as np
import peakutils
from Utils.DataHandling.data_processing import pd_to_np


def merge_peaks_from_two_signals(idx1, idx2, sig1=None, sig2=None, merge_type='union', intersect_win_size=10,
                                 union_min_dist=40, union_min_thresh=1):
    peaks = []
    if len(idx1) == 0:
        return peaks
    elif len(idx2) == 0:
        return peaks
    elif sig1 is None:
        return peaks
    elif sig2 is None:
        return peaks

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
    if merge_type == 'union' and len(merged_idx) > 0:
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


def run_peak_utils_peak_detection(signal, min_thresh, min_dist_between_peaks):
    # Check for zero lengths signal or for flat signal
    if len(signal) == 0:
        return np.array([])
    if max(signal) == min(signal):
        return np.array([])

    return peakutils.indexes(signal, thres=min_thresh, min_dist=min_dist_between_peaks)
