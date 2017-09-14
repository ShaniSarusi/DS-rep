import math as m
import pickle
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from Sandbox.Zeev import Gait_old as c

with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
    sample = pickle.load(fp)
with open(join(c.pickle_path, 'features_armswing'), 'rb') as fp:
    swing_features = pickle.load(fp)


#filters
reg = sample['WalkType'] == 'Regular'
fast = sample['PaceInstructions'] == 'H'
med = sample['PaceInstructions'] == 'M'
slow = sample['PaceInstructions'] == 'L'
hold = sample['ItemHeld'] != 'None'
not_hold = sample['ItemHeld'] == 'None'

regfast = reg & fast
regmed = reg & med
regslow = reg & slow

lhs_deg = []
for i in sample['SampleId']:
    d = swing_features[i]['lhs']['z']['degrees']
    abs_d = [abs(i) for i in d]
    av = sum(abs_d)/len(abs_d)
    lhs_deg.append(av)

a1 = [i for (i, v) in zip(lhs_deg, regfast) if v]
a2 = [i for (i, v) in zip(lhs_deg, regmed) if v]
a3 = [i for (i, v) in zip(lhs_deg, regslow) if v]

# plot 1
fig = plt.figure()
plt.boxplot([a1, a2, a3])
plt.xticks(np.array([0, 1, 2, 3]), ['', 'fast', 'medium', 'slow'], fontsize=16)
plt.ylabel('Average arm swing degrees', fontsize=16)


# plot 2 - time distribution
lhs_time = []
for i in sample['SampleId']:
    d = swing_features[i]['lhs']['z']['duration']
    av = sum(d) / len(d)
    lhs_time.append(av)
fig = plt.figure()
a1 = [i for (i, v) in zip(lhs_time, reg) if v]
plt.hist(a1, 20)


fig = plt.figure()
i = 0
xt = np.array([1, 1.2, 1.4])
for y in [a1, a2, a3]:
# for y in [xcr1, xcr2, ycr1, ycr2, zcr1, zcr2, zgyr1, zgyr2]:
    # Add some random "jitter" to the x-axis
    x = np.random.normal(xt[i], 0.04, size=len(y))
    i += 1
    if i == 1:
        col = 'r.'
    if i == 2:
        col = 'g.'
    else:
        col = 'b.'
    plt.plot(x, y, col, alpha=0.2, ms=9)

xt2 = np.array([0, 1])
my_xticks = ['','degrees']
plt.xticks(xt2, my_xticks, fontsize=16)
plt.ylabel('Average degrees', fontsize=16)
plt.yticks(fontsize=16)




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
plt.xticks([1,2], ['Regular, n= ' + str(len(y1)), 'PD simulation, n=' + str(len(y2))], fontsize=16)
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
x = np.arange(0,ax_max)
plt.plot(x,x)
plt.text(0.05*ax_max, 0.9*ax_max, 'correlation = ' + str(round(corr,2)), fontsize=14)
plt.text(0.05*ax_max, 0.85*ax_max, 'pval = ' + str(pval), fontsize=14)




