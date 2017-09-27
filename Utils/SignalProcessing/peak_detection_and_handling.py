import numpy as np
import peakutils
import pandas as pd
from Utils.DataHandling.data_processing import pd_to_np


def merge_peaks_from_two_signals_union_one_stage(idx1, idx2, sig1=None, sig2=None, union_min_dist=40):
    if len(idx1) == 0:
        return []
    elif len(idx2) == 0:
        return []
    elif sig1 is None:
        return []
    elif sig2 is None:
        return []

    sig = np.concatenate((pd_to_np(sig1), pd_to_np(sig2)))
    idx = np.concatenate((np.asarray(idx1), np.asarray(idx2)))
    a = np.argsort(sig)
    sig = sig[a[::-1]]
    idx = idx[a[::-1]]
    keep = np.ones((len(idx)), dtype=bool)
    for i in range(len(idx)):
        if not keep[i]:
            continue
        keep = np.abs(idx - idx[i]) >= union_min_dist
        keep[i] = True
    res = idx[keep]
    res.sort()
    return res


def merge_peaks_from_two_signals(idx1, idx2, sig1=None, sig2=None, merge_type='union_two_stages', intersect_win_size=10,
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
    if merge_type == 'union_two_stages' and len(merged_idx) > 0:
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

        # remove peaks not in intersection that are below min threshold
        left_out = pd.DataFrame(list(zip(idx_left_out, sig_left_out)), columns=['idx', 'sig'])
        left_out = left_out[left_out['sig'] >= union_min_thresh].reset_index(drop=True)

        # remove peaks not in intersection that are closer to intersection than min dist
        left_out['nearest_intersect_idx'] = np.nan
        for i in range(len(left_out)):
            left_out['nearest_intersect_idx'].set_value(i, np.min(np.abs(merged_idx - left_out['idx'].iloc[i])))
        left_out = left_out[left_out['nearest_intersect_idx'] >= union_min_dist]

        # remove peaks that are too close to each other
        left_out = left_out.sort_values('sig', ascending=False).reset_index(drop=True)
        sig = np.asarray(left_out['sig'])
        idx = np.asarray(left_out['idx'])
        keep = np.ones((len(idx)), dtype=bool)
        for i in range(len(idx)):
            if not keep[i]:
                continue
            keep = np.abs(idx - idx[i]) >= union_min_dist
            keep[i] = True
        union_idx = idx[keep]
        peaks = peaks + union_idx.tolist()

    peaks.sort()
    return peaks


def run_peak_utils_peak_detection(signal, min_thresh, min_dist_between_peaks):
    # Check for zero lengths signal or for flat signal
    if len(signal) == 0:
        return np.array([])
    if max(signal) == min(signal):
        return np.array([])

    return peakutils.indexes(signal, thres=min_thresh, min_dist=min_dist_between_peaks)
