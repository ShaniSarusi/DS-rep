import math as m
import pickle
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

import Gait.Resources.config as c

with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
    sample = pickle.load(fp)
with open(join(c.pickle_path, 'features_steps'), 'rb') as fp:
    step_features = pickle.load(fp)


# plot step time differences
reg = (sample['WalkType'] == 'Regular').as_matrix()
y = [x*100 for x in step_features['mean_step_time_diff']]
y1 = [i for (i, v) in zip(y, reg) if v]
y2 = [i for (i, v) in zip(y, ~reg) if v]

fig = plt.figure()
data = [y1, y2]
statistic, pval = st.ks_2samp(y1, y2)
plt.boxplot(data)
plt.ylabel('Mean percent difference in step time:left vs right', fontsize=16)
plt.xticks([1,2], ['Regular, n= ' + str(len(y1)), 'PD simulation, n=' + str(len(y2))], fontsize=16)
plt.text(1.3, 100, 'ks_2 pval = ' + str(round(pval,3)), fontsize=14)

# plot cadence
reg = (sample['WalkType'] == 'Regular').as_matrix()
y = step_features['cadence']
y1 = [i for (i, v) in zip(y, reg) if v]
y2 = [i for (i, v) in zip(y, ~reg) if v]

fig = plt.figure()
data = [y1, y2]
statistic, pval = st.ks_2samp(y1, y2)
plt.boxplot(data)
plt.ylabel('Cadence (steps/min)', fontsize=16)
plt.xticks([1, 2], ['Regular, n= ' + str(len(y1)), 'PD simulation, n=' + str(len(y2))], fontsize=16)
plt.text(1.3, 100, 'ks_2, pval = ' + str(round(pval,4)), fontsize=14)


# plot step detection result
# select ids to plot
ids = sample[sample['StepCount'].notnull()].index.tolist()  # all step count ids
steps_true = sample['StepCount'][ids]
steps_alg = list(step_features['step_count'][i] for i in ids)
plt.scatter(steps_alg, steps_true, color='b')
corr, pval = st.pearsonr(steps_alg, steps_true)
plt.xlabel('Algorithm result', fontsize=18)
plt.ylabel('True count', fontsize=16)
highest = max(max(steps_alg), max(steps_true))
ax_max = int(m.ceil(highest / 10.0)) * 10
plt.ylim(0, ax_max)
plt.xlim(0, ax_max)
x = np.arange(0, ax_max)
plt.plot(x,x)
plt.text(0.05*ax_max, 0.9*ax_max, 'correlation = ' + str(round(corr,2)), fontsize=14)
plt.text(0.05*ax_max, 0.85*ax_max, 'pval = ' + str(pval), fontsize=14)




