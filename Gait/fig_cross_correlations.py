import pickle
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import Gait.config as config

with open(join(config.pickle_path, 'metadata_sample'), 'rb') as fp:
    sample = pickle.load(fp)
with open(join(config.pickle_path, 'features_generic'), 'rb') as fp:
    ft = pickle.load(fp)


# analyze axis cross correlation - arm swing
sensor = 'acc'
xcr = ft['both_' + sensor + '_x_cross_corr']
ycr = ft['both_' + sensor + '_y_cross_corr']
zcr = ft['both_' + sensor + '_z_cross_corr']
ncr = ft['both_' + sensor + '_n_cross_corr']
xgyr = ft['both_' + 'gyr' + '_x_cross_corr']
ygyr = ft['both_' + 'gyr' + '_y_cross_corr']
zgyr = ft['both_' + 'gyr' + '_z_cross_corr']
ngyr = ft['both_' + 'gyr' + '_n_cross_corr']

#filters
reg = sample['WalkType'] == 'Regular'
pace = sample['PaceInstructions'] != 'H'
straight = sample['WalkDirection'] != 'Straight'
hold = sample['ItemHeld'] != 'None'
not_hold = sample['ItemHeld'] == 'None'

nh = reg & pace & not_hold
h = reg & pace & hold

nh = reg  & not_hold
h = reg  & hold

xcr1 = xcr[nh]
xcr2 = xcr[h]

ycr1 = ycr[nh]
ycr2 = ycr[h]

zcr1 = zcr[nh]
zcr2 = zcr[h]

ncr1 = ncr[nh]
ncr2 = ncr[h]

xgyr1 = xgyr[nh]
xgyr2 = xgyr[h]

ygyr1 = ygyr[nh]
ygyr2 = ygyr[h]

zgyr1 = zgyr[nh]
zgyr2 = zgyr[h]

ngyr1 = ngyr[nh]
ngyr2 = ngyr[h]

ks_x = stats.ks_2samp(xcr1, xcr2)
ks_y = stats.ks_2samp(ycr1, ycr2)
ks_z = stats.ks_2samp(zcr1, zcr2)
ks_n = stats.ks_2samp(ncr1, ncr2)
ks_gx = stats.ks_2samp(xgyr1, xgyr2)
ks_gy = stats.ks_2samp(ygyr1, ygyr2)
ks_gz = stats.ks_2samp(zgyr1, zgyr2)
ks_gn = stats.ks_2samp(ngyr1, ngyr2)

#plt.scatter(xcr1, ycr1, color='b')
#plt.scatter(xcr2, ycr2, color='r')
pval = [ks_x[1], ks_y[1], ks_z[1], ks_n[1], ks_gx[1], ks_gy[1], ks_gz[1], ks_gn[1]]

# do triple bar point graph
fig = plt.figure()
i = 0
xt = np.array([1, 1.2, 2, 2.2, 3, 3.2, 4, 4.2, 5, 5.2, 6, 6.2, 7, 7.2, 8, 8.2])
for y in [xcr1, xcr2, ycr1, ycr2, zcr1, zcr2, ncr1, ncr2, xgyr1, xgyr2, ygyr1, ygyr2, zgyr1, zgyr2, ngyr1, ngyr2]:
# for y in [xcr1, xcr2, ycr1, ycr2, zcr1, zcr2, zgyr1, zgyr2]:
    # Add some random "jitter" to the x-axis
    x = np.random.normal(xt[i], 0.04, size=len(y))
    i += 1
    if i % 2:
        col = 'r.'
        plt.text(xt[i]-0.6, -1.1, 'p=' + str(round(pval[i/2], 3)), fontsize=11)
    else:
        col = 'b.'
    plt.plot(x, y, col, alpha=0.2, ms=9)

xt2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
my_xticks = ['','X','Y','Z', 'N', 'GyrX', 'GyrY', 'GyrZ', 'GyrN']
plt.xticks(xt2, my_xticks, fontsize=16)
plt.ylabel('Pearson correlation (R)', fontsize=16)
plt.yticks(fontsize=16)
plt.axis([0.5, 8.5, -1.2, 1.2])




