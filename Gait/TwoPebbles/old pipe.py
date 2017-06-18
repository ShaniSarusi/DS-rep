from Utils import Utils as ut
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from Gait.TwoPebbles import TwoAcc as ta
import scipy.stats as st


# Read data***************************
common_path = 'C:\Users\zwaks\Documents\Data\TwoPebbles_Exp2'
left_watch = common_path + '\w_lft_rawa.csv'
right_watch = common_path + '\w_rt_rawb.csv'
lhs = ut.read_export_tool_csv(left_watch)
rhs = ut.read_export_tool_csv(right_watch)
lhs['n'] = (lhs['x'] ** 2 + lhs['y'] ** 2 + lhs['z'] ** 2) ** 0.5
rhs['n'] = (rhs['x'] ** 2 + rhs['y'] ** 2 + rhs['z'] ** 2) ** 0.5

# Synchronize signal********************
# Find shake area using protocol and by plotting the signal  #example: plt.plot(lft.loc[range(100,8000),'norm'])
lhs_shake_range = range(3000, 4000)
rhs_shake_range = range(2000, 3000)

# find offset
#time_offset = ta.find_offset(lhs.loc[lhs_shake_range,'n'].as_matrix(), rhs.loc[rhs_shake_range,'n'].as_matrix())


a = lhs.loc[lhs_shake_range,'n'].as_matrix()
b = rhs.loc[rhs_shake_range,'n'].as_matrix()
offset = np.argmax(signal.correlate(a,b))
shift_by = a.shape[0] - offset
time_offset = rhs.loc[rhs_shake_range[0] + shift_by, 'ts'] - lhs.loc[lhs_shake_range[0],'ts']
# The link below says you need to subtract 1 from shift_by, but based on my plotting it looks like you don't need to
# link: http://stackoverflow.com/questions/4688715/find-time-shift-between-two-similar-waveforms

# Plot shift
st = 500;  en = 550
plt.plot(a[range(st, en)])
#plt.plot(b[range(st, en)])

plt.plot(b[range(st + shift_by, en + shift_by)])


# Segment data into samples *****************************************
plt.plot(lhs.loc[range(0,50000),'n'].as_matrix())
#metadata_path = common_path + '\Gait\Exp2\Metadata.xlsx'

metadata_path = common_path + '\Gait\Exp2\Metadata_short.xlsx'
metadata = pd.read_excel(metadata_path)

for i in range(metadata.shape[0]):
    # start
    tmp = (rhs['ts'] <= lhs.loc[metadata.loc[i, 'lhs_st'], 'ts'] + time_offset)
    metadata.loc[i, 'rhs_st'] = tmp[tmp == True].index[-1]
    # end
    tmp = (rhs['ts'] <= lhs.loc[metadata.loc[i, 'lhs_en'], 'ts'] + time_offset)
    metadata.loc[i, 'rhs_en'] = tmp[tmp == True].index[-1]
metadata['rhs_st'] = metadata['rhs_st'].astype('int')
metadata['rhs_en'] = metadata['rhs_en'].astype('int')

# Extract features
# this may be relevant
# also tshred package


# 4 Extract a few features on each sample.
def pearson_corr(lhs_p, rhs_p):
    # TODO need to align signals first due to changing sampling rates and missing data
    # can do this via a loop that goes over ts of 1 side and compares with the other

    #tmp solution instead of alignment
    idx = np.min([lhs_p.shape[0], rhs_p.shape[0]])
    lhs_p = lhs_p.reset_index()
    rhs_p = rhs_p.reset_index()
    lhs_p = lhs_p.loc[range(idx)]
    rhs_p = rhs_p.loc[range(idx)]

    # normalize by substracting mean [check if this is needed] - http://dsp.stackexchange.com/questions/9491/normalized-square-error-vs-pearson-correlation-as-similarity-measures-of-two-sig?noredirect=1&lq=1
    # should we subtract std also?
    lhs_x = lhs_p['x'] - np.mean(lhs_p['x'])
    rhs_x = rhs_p['x'] - np.mean(rhs_p['x'])
    lhs_y = lhs_p['y'] - np.mean(lhs_p['y'])
    rhs_y = rhs_p['y'] - np.mean(rhs_p['y'])
    lhs_z = lhs_p['z'] - np.mean(lhs_p['z'])
    rhs_z = rhs_p['z'] - np.mean(rhs_p['z'])
    lhs_n = lhs_p['n'] - np.mean(lhs_p['n'])
    rhs_n = rhs_p['n'] - np.mean(rhs_p['n'])
    import scipy.stats as st
    xcorr = st.pearsonr(lhs_x, rhs_x)[0]
    ycorr = st.pearsonr(lhs_y, rhs_y)[0]
    zcorr = st.pearsonr(lhs_z, rhs_z)[0]
    ncorr = st.pearsonr(lhs_n, rhs_n)[0]
    return xcorr, ycorr, zcorr, ncorr



ft = pd.DataFrame()
for i in range(metadata.shape[0]):
    lhs_st = metadata.loc[i, 'lhs_st']
    lhs_en = metadata.loc[i, 'lhs_en']
    rhs_st = metadata.loc[i, 'rhs_st']
    rhs_en = metadata.loc[i, 'rhs_en']
    lhs_i = lhs.loc[lhs_st:lhs_en]
    rhs_i = rhs.loc[rhs_st:rhs_en]

    xcorr, ycorr, zcorr, ncorr = pearson_corr(lhs_i, rhs_i)
    ft.loc[i, 'xcorr'] = xcorr
    ft.loc[i, 'ycorr'] = ycorr
    ft.loc[i, 'zcorr'] = zcorr
    ft.loc[i, 'ncorr'] = ncorr

#ALSO CALCULATE THE BELOW...
#    Difference in mean axis and norm values.  [xr mean vs xl mean, etc.]
#    Difference between std of each axis
pass


# 5 classifier to distinguish between labels
#temp label
y = np.array([0, 0, 0, 1, 1, 1, 1])

# do svm
from sklearn import svm
clf = svm.SVC()
clf.fit(ft.as_matrix(), y)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(ft.as_matrix(), y)
importance = clf.feature_importances_

# 6 Visualization
#  a) show video of example
#  b) show signal off both hands [overlap norms + overlap a single axis example - say z]
#  c) show feature importance
#  d) show classification - roc



