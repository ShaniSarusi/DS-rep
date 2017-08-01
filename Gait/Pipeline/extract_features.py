import pickle
from os.path import join

import numpy as np
import pandas as pd
from scipy.integrate import trapz
from scipy.stats import pearsonr

import Gait.Resources.config as c
from Gait.Pipeline.gait_utils import truncate
from Utils.Preprocessing.denoising import butter_lowpass_filter


def _add_feature(df, sensor, sensor_name, axes, sides, what):
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
                df.loc[sample_id, ft_name] = _calc_feature(data, what)
    return df


def _calc_feature(data, what):
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
        # normalize by substracting mean [check if this is needed].  Also, should we subtract std as well?
        res = pearsonr(data[0] - data[0].mean(), data[1] - data[1].mean())[0]
    # if what == 'wavelets':
    return res


def _single_arm_features(gyr, acc, time):
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
        df.iloc[i]['degrees'] = trapz(gyr[st:en], t)

        dt = []
        for j in range(df.iloc[i]['start_idx'], df.iloc[i]['end_idx']):
            dt.append((time.iloc[j+1]['value'] - time.iloc[j]['value']).total_seconds())
        dx = np.abs(np.diff(acc[st:en+1]))
        df.iloc[i]['jerk'] = np.mean(dx/dt)  # This assumes evenly spaced time series
    return df


# arm swing asymmetry (ASA)
def _asa(a, b):
    return 100 * (45 - np.arctan(np.max([a/b, b/a]))) / 90


def extract_arm_swing_features():
    # load data
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp: sample = pickle.load(fp)
    with open(join(c.pickle_path, 'gyr'), 'rb') as fp: gyr = pickle.load(fp)
    with open(join(c.pickle_path, 'acc'), 'rb') as fp: acc = pickle.load(fp)
    with open(join(c.pickle_path, 'time'), 'rb') as fp: time = pickle.load(fp)

    # TODO using only z axis for gyro calculation. Maybe norm or the euler stuff is better
    col = ['mean_range', 'mean_max_angular_velocity']
    ft_arm_lhs = pd.DataFrame(index=range(sample['SampleId'].shape[0]), columns=col)
    ft_arm_rhs = pd.DataFrame(index=range(sample['SampleId'].shape[0]), columns=col)
    for i in sample['SampleId']:
        df_lhs = _single_arm_features(gyr[i]['lhs']['z'].as_matrix(), acc[i]['lhs']['n'], time[i]['lhs'])
        df_rhs = _single_arm_features(gyr[i]['rhs']['z'].as_matrix(), acc[i]['rhs']['n'], time[i]['rhs'])

        ft_arm_lhs.iloc[i]['mean_range'] = df_lhs['degrees'].mean()*c.radian
        ft_arm_lhs.iloc[i]['mean_max_angular_velocity'] = df_lhs['max_velocity'].mean() * c.radian
        ft_arm_rhs.iloc[i]['mean_range'] = df_rhs['degrees'].mean()*c.radian
        ft_arm_rhs.iloc[i]['mean_max_angular_velocity'] = df_rhs['max_velocity'].mean() * c.radian

    # TODO implement the funcion below: (ASA, variability/symmetry, etc.
    col = ['asa', 'other']
    ft_arm_both = pd.DataFrame(index=range(sample['SampleId'].shape[0]), columns=col)
    for i in sample['SampleId']:
        ft_arm_both.iloc[i]['asa'] = _asa(ft_arm_lhs.iloc[i]['mean_range'], ft_arm_rhs.iloc[i]['mean_range'])
        ft_arm_both.iloc[i]['other'] = 1234  # just a temporary placeholder

    # save results
    with open(join(c.pickle_path, 'ft_arm_lhs'), 'wb') as fp: pickle.dump(ft_arm_lhs, fp)
    with open(join(c.pickle_path, 'ft_arm_rhs'), 'wb') as fp: pickle.dump(ft_arm_rhs, fp)
    with open(join(c.pickle_path, 'ft_arm_both'), 'wb') as fp: pickle.dump(ft_arm_both, fp)


def extract_generic_features():
    # load metadata and sensor data
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp: sample = pickle.load(fp)
    with open(join(c.pickle_path, 'acc'), 'rb') as fp: acc = pickle.load(fp)
    with open(join(c.pickle_path, 'gyr'), 'rb') as fp: gyr = pickle.load(fp)

    # pre-processing - truncate
    fr_pct = 10
    bk_pct = 10
    acc = truncate(acc, fr_pct, bk_pct)
    gyr = truncate(gyr, fr_pct, bk_pct)

    sides = ['lhs', 'rhs']
    axes = ['x', 'y', 'z', 'n']
    for i in range(len(acc)):
        for side in sides:
            for axis in axes:
                acc[i][side][axis] = butter_lowpass_filter(acc[i][side][axis], lowcut=15, sampling_rate=128, order=10)
                gyr[i][side][axis] = butter_lowpass_filter(gyr[i][side][axis], lowcut=15, sampling_rate=128, order=10)

    # calculate features
    ft = pd.DataFrame(index=range(sample.shape[0]))
    # single side accelerometer  # TODO which features need absolute value? Where does gravity need to be removed?
    ft = _add_feature(ft, sensor=acc, sensor_name='acc', axes=['x', 'y', 'z', 'n'], sides=['lhs', 'rhs'], what='mean')
    ft = _add_feature(ft, sensor=acc, sensor_name='acc', axes=['x', 'y', 'z', 'n'], sides=['lhs', 'rhs'], what='median')
    ft = _add_feature(ft, sensor=acc, sensor_name='acc', axes=['x', 'y', 'z', 'n'], sides=['lhs', 'rhs'], what='std')
    # both sides accelerometer
    ft = _add_feature(ft, sensor=acc, sensor_name='acc', axes=['x', 'y', 'z', 'n'], sides=['both'], what='mean_diff')
    ft = _add_feature(ft, sensor=acc, sensor_name='acc', axes=['x', 'y', 'z', 'n'], sides=['both'], what='median_diff')
    ft = _add_feature(ft, sensor=acc, sensor_name='acc', axes=['x', 'y', 'z', 'n'], sides=['both'], what='std_diff')
    ft = _add_feature(ft, sensor=acc, sensor_name='acc', axes=['x', 'y', 'z', 'n'], sides=['both'], what='cross_corr')

    # single side gyro
    ft = _add_feature(ft, sensor=gyr, sensor_name='gyr', axes=['x', 'y', 'z', 'n'], sides=['lhs', 'rhs'], what='mean')
    ft = _add_feature(ft, sensor=gyr, sensor_name='gyr', axes=['x', 'y', 'z', 'n'], sides=['lhs', 'rhs'], what='median')
    ft = _add_feature(ft, sensor=gyr, sensor_name='gyr', axes=['x', 'y', 'z', 'n'], sides=['lhs', 'rhs'], what='std')
    # both sides gyro
    ft = _add_feature(ft, sensor=gyr, sensor_name='gyr', axes=['x', 'y', 'z', 'n'], sides=['both'], what='mean_diff')
    ft = _add_feature(ft, sensor=gyr, sensor_name='gyr', axes=['x', 'y', 'z', 'n'], sides=['both'], what='median_diff')
    ft = _add_feature(ft, sensor=gyr, sensor_name='gyr', axes=['x', 'y', 'z', 'n'], sides=['both'], what='std_diff')
    ft = _add_feature(ft, sensor=gyr, sensor_name='gyr', axes=['x', 'y', 'z', 'n'], sides=['both'], what='cross_corr')

    # save features
    with open(join(c.pickle_path, 'features_generic'), 'wb') as fp:
        pickle.dump(ft, fp)


if __name__ == '__main__':
    extract_arm_swing_features()
    extract_generic_features()

# OLD code
# sides = ['lhs', 'rhs']
# axes = ['x', 'y', 'z', 'n']
# for i in sample['SampleId']:
#     sw = {}
#     for side in sides:
#         sw[side] = {}
#         for axis in axes:
#             g = gyr[i][side][axis].as_matrix()
#             a = acc[i][side]['n']
#             sw[side][axis] = ft.arm_swing_ft(g, a, time[i][side])
#     swing.append(sw)


#######################
# gyro view
# id = 55
# gl = gyr[id]['lhs']
# gr = gyr[id]['rhs']
# plt.plot(gl['z'])
# plt.plot(gr['z'])
#
# #sample to check
# id = 5
# aclhs = acc[id]['lhs']
# gylhs = gyr[id]['lhs']
#
# a_l = aclhs['n']
# a_r = aclhs['n']
# lp = sig.find_peaks_cwt(a_l,np.arange(10,20))
# rp = sig.find_peaks_cwt(a_r,np.arange(10,20))
#
# plt.plot(gylhs['x'])
# plt.plot(aclhs['z']+15)
# plt.plot(aclhs['n']+25)
#
# plt.plot(aclhs['x'] - aclhs['x'].mean())
#
# # try filter
# sampling_freq = 128 #hz
# nyquist_freq = sampling_freq/2
#
# gravity_freq = 0.2/nyquist_freq
# tremor_freq = 3.5/nyquist_freq
#
# b, a = sig.butter(5, gravity_freq, btype='lowpass')
# b, a = sig.butter(5, tremor_freq, btype='highpass')
#
# b, a = sig.butter(5, [gravity_freq, tremor_freq], btype='band')
#
# y = sig.lfilter(b, a, aclhs['n'])
# plt.plot(y)
# plt.plot(aclhs['n'])
