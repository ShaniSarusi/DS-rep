# code for calculating features
import scipy.stats as st
import numpy as np
import pandas as pd
import scipy.integrate as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import peakutils
from Utils import Utils as Utils


def add_feature(df, sensor, sensor_name, axes, sides, what):
    for sample_id in range(df.shape[0]):
        for axis in axes:
            for side in sides:
                ft_name = side + '_' + sensor_name + '_' + axis + '_' + what
                # set data
                if side == 'both':
                    data = [sensor[sample_id]['lhs'][axis], sensor[sample_id]['rhs'][axis]]
                else:
                    data = sensor[sample_id][side][axis]
                # apply the feature calculation
                df.loc[sample_id, ft_name] = calc_feature(data, what)
    return df


def calc_feature(data, what):
    res = []
    if what == 'mean':
        res = data.mean()
    if what == 'median':
        res = data.median()
    if what == 'std':
        res = data.std()
    if what == 'mean_diff':
        res = data[0].mean() - data[1].mean()
    if what == 'median_diff':
        res = data[0].median() - data[1].median()
    if what == 'std_diff':
        res = data[0].std() - data[1].std()
    if what == 'cross_corr':
        # normalize by substracting mean [check if this is needed] - http://dsp.stackexchange.com/questions/9491/normalized-square-error-vs-pearson-correlation-as-similarity-measures-of-two-sig?noredirect=1&lq=1
        # should we subtract std also?
        res = st.pearsonr(data[0] - data[0].mean(), data[1] - data[1].mean())[0]

    # if what == 'wavelets':
    #     df_ft_vals, df_ft_name = wavelets(data, ft_name)
    #     df.loc[sample_id, df_ft_name] = df_ft_vals
    return res


def single_arm_ft(gyr, acc, time):
    crossings_idx = np.where(np.diff(np.signbit(gyr)))[0] + 1
    cols = ['start_idx', 'end_idx', 'duration', 'degrees', 'max_velocity', 'jerk']
    df = pd.DataFrame(index=range(len(crossings_idx) - 1), columns=cols)
    for i in range(len(crossings_idx)-1):
        st = crossings_idx[i]; en = crossings_idx[i + 1]
        df.iloc[i]['start_idx'] = st
        df.iloc[i]['end_idx'] = en
        df.iloc[i]['duration'] = (time.iloc[en]['value'] - time.iloc[st]['value']).total_seconds()
        df.iloc[i]['max_velocity'] = np.max(gyr[st:en])

    # degrees and jerk
    for i in range(len(crossings_idx)-1):
        st = df.iloc[i]['start_idx']
        en = df.iloc[i]['end_idx']
        t = (time.iloc[st:en]['value']) - time.iloc[st]['value']
        df.iloc[i]['degrees'] = sp.trapz(gyr[st:en], t)

        dt = []
        for j in range(df.iloc[i]['start_idx'], df.iloc[i]['end_idx']):
            dt.append((time.iloc[j+1]['value'] - time.iloc[j]['value']).total_seconds())
        dx = np.abs(np.diff(acc[st:en+1]))
        df.iloc[i]['jerk'] = np.mean(dx/dt)  # This assumes evenly spaced time series
    return df


# arm swing asymmetry (ASA)
def asa(a, b):
    return 100 * (45 - np.arctan(np.max([a/b, b/a]))) / 90


def zero_crossings(a):
    # returns indices of zero crossings, and area between each consecutive zero crossings
    # a = np.asarray(a)
    # find zero crossings
    crossings_idx = np.where(np.diff(np.signbit(a)))[0] + 1
    if len(crossings_idx) == 0:
        return None, None

    # remove crossing at point zero if it exists. Also, no crossing should exist at the last point if it is zero.
    if crossings_idx[0] == 1:
        crossings_idx = np.delete(crossings_idx, 0)

    idx = np.unique(np.append(0, crossings_idx, len(a)))
    area = []
    for i in range(len(idx) - 1):
        st = idx[i]
        en = idx[i+1]
        area.append(np.sum(a[st:en]))
    return crossings_idx, np.asarray(area)


def merge_peaks(idx1, idx2, sig1=None, sig2=None, p_type='keep_max', win_size=10):
    if p_type == 'keep_max':
        return merge_peaks_keep_max_signal(idx1, idx2, sig1, sig2, win_size)
    elif p_type =='keep_first_signal':
        return merge_peaks_keep_first_signal(idx1, idx2, win_size)
    else:
        pass


def merge_peaks_keep_max_signal(idx1, idx2, sig1=None, sig2=None, win_size=10):
    peaks_raw = []
    for i in range(len(idx1)):
        for j in range(len(idx2)):
            if abs(idx1[i] - idx2[j]) <= win_size:
                #prevent division by zero
                if sig1.iloc[i] == 0:
                    sig1.iloc[i] += 0.00001
                if sig2.iloc[j] == 0:
                    sig2.iloc[j] += 0.00001
                peaks_raw.append((idx1[i], idx2[j], sig1.iloc[i], sig2.iloc[j], sig1.iloc[i]- sig2.iloc[j]))

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
        for j in range(i+1,peaks.shape[0]):
            if peaks['idx'].iloc[i] == peaks['idx'].iloc[j]:
                if peaks['maxsignal'].iloc[i] >= peaks['maxsignal'].iloc[j]:
                    remove.append(j)
                else:
                    remove.append(i)
    if len(remove) == 0:
        return peaks
    else:
        return peaks.drop(peaks.index[np.unique(remove)], axis=0)


def merge_peaks_keep_first_signal(a, b, win_size=20):
    l = [range(i - win_size, i + win_size) for i in a]
    idx = [item for sublist in l for item in sublist]
    idx_add = np.setdiff1d(b, idx)
    concat = np.concatenate((np.array(a), idx_add))
    res = np.unique(concat)
    return res


def remove_adjacent_peaks(peaks, win_size=20):
    peaks.sort_values(by='idx')
    i = 0
    while i < peaks.shape[0] - 1:
        if peaks['idx'].iloc[i + 1] - peaks['idx'].iloc[i] > win_size:
            i += 1
        elif peaks['maxsignal'].iloc[i + 1] <= peaks['maxsignal'].iloc[i]:
            peaks = peaks.drop(peaks.index[i+1], axis=0)
        else:
            peaks = peaks.drop(peaks.index[i], axis=0)
    idx = peaks['idx'].tolist()
    return idx


def combine_both_single(b, l, r, win_size=30):
    single = np.unique([l,r])
    for i in range(len(single)):
        if np.min(np.abs(b-single[i])) > win_size:
            b.append(single[i])
    return b


def remove_small_peaks(peaks, num_stds=2):
    mean = peaks['maxsignal'].mean()
    std = peaks['maxsignal'].std()
    remove = []
    for i in range(peaks.shape[0]):
        if peaks['maxsignal'].iloc[i] < (mean - num_stds*std):
            remove.append(i)
    if len(remove) == 0:
        return peaks
    else:
        return peaks.drop(peaks.index[np.unique(remove)], axis=0)


def typefilter(data, filt_data, string):
    filt = (filt_data == string).as_matrix()
    a = [i for (i, v) in zip(data, filt) if v]
    b = [i for (i, v) in zip(data, ~filt) if v]
    return a, b


def score_max_peak(the_signal, sampling_freq, min_hz, max_hz, show=False):
    sig_mean_norm = the_signal - np.mean(the_signal)
    a = np.power(np.abs(np.fft.fft(sig_mean_norm)), 2)
    a = a / np.sum(a)

    f = sampling_freq
    num_t = a.shape[0]
    tick = f / 2.0 / num_t

    if show:
        x_label = np.asarray(range(num_t))*tick
        plt.plot(x_label, a)

    relevant_start = int(round(min_hz / tick))
    relevant_end = int(round(max_hz / tick))

    score = np.max(a[relevant_start:relevant_end])
    return score


def mva_no_nans(x, win_size):
    a = [1.0 / win_size] * win_size
    res = sig.filtfilt(a, 1, x)
    return pd.Series(res)


def detect_peaks(input_signal, peak_type, param1, param2):
    if peak_type == 'p_utils':
        if max(input_signal) == 0:
            val = 1
        else:
            val = param1 / max(input_signal)
        if max(input_signal) == min(input_signal):
            return np.array([])
        else:
            return peakutils.indexes(input_signal, thres=val, min_dist=param2)
    elif peak_type == 'scipy':
        return sig.find_peaks_cwt(input_signal, np.arange(param1, param2))
    else:
        return


def max_filter(x, win_size):
    a = Utils.pd_to_np(x)
    max_val = np.max(a)
    b = np.asarray([max_val] * win_size)
    res = np.convolve(a, b, 'same')
    return res
