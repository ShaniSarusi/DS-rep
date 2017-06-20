import pickle
from os.path import join
import pandas as pd
import Gait.GaitUtils.algorithms as ft
import Gait.config as c


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
        df_lhs = ft.single_arm_ft(gyr[i]['lhs']['z'].as_matrix(), acc[i]['lhs']['n'], time[i]['lhs'])
        df_rhs = ft.single_arm_ft(gyr[i]['rhs']['z'].as_matrix(), acc[i]['rhs']['n'], time[i]['rhs'])

        ft_arm_lhs.iloc[i]['mean_range'] = df_lhs['degrees'].mean()*c.radian
        ft_arm_lhs.iloc[i]['mean_max_angular_velocity'] = df_lhs['max_velocity'].mean() * c.radian
        ft_arm_rhs.iloc[i]['mean_range'] = df_rhs['degrees'].mean()*c.radian
        ft_arm_rhs.iloc[i]['mean_max_angular_velocity'] = df_rhs['max_velocity'].mean() * c.radian

    # TODO implement the funcion below: (ASA, variability/symmetry, etc.
    col = ['asa', 'other']
    ft_arm_both = pd.DataFrame(index=range(sample['SampleId'].shape[0]), columns=col)
    for i in sample['SampleId']:
        ft_arm_both.iloc[i]['asa'] = ft.asa(ft_arm_lhs.iloc[i]['mean_range'], ft_arm_rhs.iloc[i]['mean_range'])
        ft_arm_both.iloc[i]['other'] = 1234  # just a temporary placeholder

    # save results
    with open(join(c.pickle_path, 'ft_arm_lhs'), 'wb') as fp: pickle.dump(ft_arm_lhs, fp)
    with open(join(c.pickle_path, 'ft_arm_rhs'), 'wb') as fp: pickle.dump(ft_arm_rhs, fp)
    with open(join(c.pickle_path, 'ft_arm_both'), 'wb') as fp: pickle.dump(ft_arm_both, fp)

if __name__ == '__main__':
    extract_arm_swing_features()

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

