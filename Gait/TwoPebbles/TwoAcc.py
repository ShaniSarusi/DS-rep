import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import scipy.stats as st


class TwoAcc:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        self.ft = pd.DataFrame()
        self.time_offset = None
        self.shift_by = None

    def add_norm(self, ptype='both'):
        if (ptype == 'left') or (ptype == 'both'):
            self.lhs['n'] = (self.lhs['x'] ** 2 + self.lhs['y'] ** 2 + self.lhs['z'] ** 2) ** 0.5
        if (ptype == 'right') or (ptype == 'both'):
            self.rhs['n'] = (self.rhs['x'] ** 2 + self.rhs['y'] ** 2 + self.rhs['z'] ** 2) ** 0.5

    # Synchronize signal
    def calc_offset(self, lhs_shake_range, rhs_shake_range):
        a = self.lhs.loc[lhs_shake_range, 'n'].as_matrix()
        b = self.rhs.loc[rhs_shake_range, 'n'].as_matrix()
        offset = np.argmax(signal.correlate(a, b))
        self.shift_by = a.shape[0] - offset
        self.time_offset = self.rhs.loc[rhs_shake_range[0] + self.shift_by, 'ts'] - self.lhs.loc[lhs_shake_range[0], 'ts']
        # The link below says you need to subtract 1 from shift_by, but based on my plotting it looks like you don't need to
        # link: http://stackoverflow.com/questions/4688715/find-time-shift-between-two-similar-waveforms

    def plot_shift(self, lhs_range, show_shift=True):
        a = self.lhs.loc[lhs_range, 'n'].as_matrix()
        b_idx = self.get_closest_index(self.lhs, value = self.lhs.loc[lhs_range[0], 'ts'])
        rhs_range = range(b_idx, b_idx + len(lhs_range))
        b = self.rhs.loc[rhs_range, 'n'].as_matrix()

        plt.plot(a)
        if show_shift:
            plt.plot(b)
        else:
            plt.plot(b[range(b_idx + self.shift_by, b_idx + len(lhs_range) + self.shift_by)])

    @staticmethod
    def get_closest_index(data, value):
        #below
        tmp = (data <= value)
        ind = None
        if np.any(tmp): #there is at least one True
            ind = tmp[tmp == True].index[-1]
        else:
            tmp = (data > value)
            ind = tmp[tmp == True].index[0]
        return ind




# Segment data into samples *****************************************
# plt.plot(lhs.loc[range(0,50000),'n'].as_matrix())
# #metadata_path = common_path + '\Gait\Exp2\Metadata.xlsx'
#
# metadata_path = common_path + '\Gait\Exp2\Metadata_short.xlsx'
# metadata = pd.read_excel(metadata_path)
#
# for i in range(metadata.shape[0]):
#     # start
#     tmp = (rhs['ts'] <= lhs.loc[metadata.loc[i, 'lhs_st'], 'ts'] + time_offset)
#     metadata.loc[i, 'rhs_st'] = tmp[tmp == True].index[-1]
#     # end
#     tmp = (rhs['ts'] <= lhs.loc[metadata.loc[i, 'lhs_en'], 'ts'] + time_offset)
#     metadata.loc[i, 'rhs_en'] = tmp[tmp == True].index[-1]
# metadata['rhs_st'] = metadata['rhs_st'].astype('int')
# metadata['rhs_en'] = metadata['rhs_en'].astype('int')
#
# # Extract features
# # this may be relevant
# # also tshred package
#
#
# # 4 Extract a few features on each sample.
# def pearson_corr(lhs_p, rhs_p):
#     # TODO need to align signals first due to changing sampling rates and missing data
#     # can do this via a loop that goes over ts of 1 side and compares with the other
#
#     #tmp solution instead of alignment
#     idx = np.min([lhs_p.shape[0], rhs_p.shape[0]])
#     lhs_p = lhs_p.reset_index()
#     rhs_p = rhs_p.reset_index()
#     lhs_p = lhs_p.loc[range(idx)]
#     rhs_p = rhs_p.loc[range(idx)]
#
#     # normalize by substracting mean [check if this is needed] - http://dsp.stackexchange.com/questions/9491/normalized-square-error-vs-pearson-correlation-as-similarity-measures-of-two-sig?noredirect=1&lq=1
#     # should we subtract std also?
#     lhs_x = lhs_p['x'] - np.mean(lhs_p['x'])
#     rhs_x = rhs_p['x'] - np.mean(rhs_p['x'])
#     lhs_y = lhs_p['y'] - np.mean(lhs_p['y'])
#     rhs_y = rhs_p['y'] - np.mean(rhs_p['y'])
#     lhs_z = lhs_p['z'] - np.mean(lhs_p['z'])
#     rhs_z = rhs_p['z'] - np.mean(rhs_p['z'])
#     lhs_n = lhs_p['n'] - np.mean(lhs_p['n'])
#     rhs_n = rhs_p['n'] - np.mean(rhs_p['n'])
#     import scipy.stats as st
#     xcorr = st.pearsonr(lhs_x, rhs_x)[0]
#     ycorr = st.pearsonr(lhs_y, rhs_y)[0]
#     zcorr = st.pearsonr(lhs_z, rhs_z)[0]
#     ncorr = st.pearsonr(lhs_n, rhs_n)[0]
#     return xcorr, ycorr, zcorr, ncorr
#
#
#
# ft = pd.DataFrame()
# for i in range(metadata.shape[0]):
#     lhs_st = metadata.loc[i, 'lhs_st']
#     lhs_en = metadata.loc[i, 'lhs_en']
#     rhs_st = metadata.loc[i, 'rhs_st']
#     rhs_en = metadata.loc[i, 'rhs_en']
#     lhs_i = lhs.loc[lhs_st:lhs_en]
#     rhs_i = rhs.loc[rhs_st:rhs_en]
#
#     xcorr, ycorr, zcorr, ncorr = pearson_corr(lhs_i, rhs_i)
#     ft.loc[i, 'xcorr'] = xcorr
#     ft.loc[i, 'ycorr'] = ycorr
#     ft.loc[i, 'zcorr'] = zcorr
#     ft.loc[i, 'ncorr'] = ncorr
#
# #ALSO CALCULATE THE BELOW...
# #    Difference in mean axis and norm values.  [xr mean vs xl mean, etc.]
# #    Difference between std of each axis
# pass
#
#
# # 5 classifier to distinguish between labels
# #temp label
# y = np.array([0, 0, 0, 1, 1, 1, 1])
#
# # do svm
# from sklearn import svm
# clf = svm.SVC()
# clf.fit(ft.as_matrix(), y)
#
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()
# clf.fit(ft.as_matrix(), y)
# importance = clf.feature_importances_
#
# # 6 Visualization
# #  a) show video of example
# #  b) show signal off both hands [overlap norms + overlap a single axis example - say z]
# #  c) show feature importance
# #  d) show classification - roc
#
#

