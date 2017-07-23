# #FROM STEP1
#
# # save results
# step_features = dict()
# step_features['cadence'] = cadence
# step_features['step_durations'] = step_durations
# step_features['mean_step_duration'] = mean_step_durations
# step_features['mean_step_time_diff'] = mean_step_time_diff
#
# step_features = pd.DataFrame.from_dict(step_features)
# with open(join(common_input, 'features_steps'), 'wb') as fp:
#     pickle.dump(step_features, fp)

import peakutils
def step_detection_wpd_brajdic(sig, movavg_winsize=40, min_peak_thresh=0.02, min_peak_distance=30):
    #default window size is 0.31sec (from Brajdic paper) * 128Hz (APDM sampling frequency) = 40
    a = sig.rolling(window=movavg_winsize, center=True).mean()
    indexes = peakutils.indexes(a, thres=min_peak_thresh / max(sig), min_dist=min_peak_distance)
    return indexes


# def mva(self, win_size=30, p_type='nans', which=None):
#     print("\rRunning: Moving average")
#     for i in range(len(self.lhs)):
#         if which == 'combined':
#             if p_type == 'regular':
#                 self.both[i] = pd.Series(self.both[i]).rolling(window=win_size, center=True).mean()
#                 self.both_abs[i] = pd.Series(self.both_abs[i]).rolling(window=win_size, center=True).mean()
#             if p_type == 'nans':
#                 self.both[i] = moving_average_no_nans(self.both[i], win_size)
#                 self.both_abs[i] = moving_average_no_nans(self.both_abs[i], win_size)
#         else:
#             if p_type == 'regular':
#                 self.lhs[i] = pd.Series(self.lhs[i]).rolling(window=win_size, center=True).mean()
#                 self.rhs[i] = pd.Series(self.rhs[i]).rolling(window=win_size, center=True).mean()
#             if p_type == 'nans':
#                 self.lhs[i] = moving_average_no_nans(self.lhs[i], win_size)
#                 self.rhs[i] = moving_average_no_nans(self.rhs[i], win_size)
