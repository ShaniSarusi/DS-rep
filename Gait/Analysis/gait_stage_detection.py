from os.path import join
import pickle
import Gait.Resources.config as c
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def string_to_int_list(str, replace_dot_zero=True):
    z = str.replace('\n', '')
    z = z.replace('[', '')
    z = z.replace(']', '')
    z = z.replace(' ', ',')
    if replace_dot_zero:
        z = z.replace('.0', '')
    z = z.split(sep=',')
    z = list(filter(None, z))
    z = [int(i) for i in z]
    return z


# APDM events TODO need to sort
with open(join(c.pickle_path, 'apdm_events'), 'rb') as fp:
    apdm_events = pickle.load(fp)
apdm_events['initial'] = apdm_events['Gait - Lower Limb - Initial Contact L (s)'] + apdm_events['Gait - Lower Limb - Initial Contact R (s)']
apdm_events['off'] = apdm_events['Gait - Lower Limb - Toe Off L (s)'] + apdm_events['Gait - Lower Limb - Toe Off R (s)']

df = pd.concat([apdm_events['initial'], apdm_events['off']], axis=1)

# Wrist events
with open(join(c.pickle_path, 'acc'), 'rb') as fp:
    acc = pickle.load(fp)
ts = [(acc[i]['lhs']['ts'] - acc[i]['lhs']['ts'].iloc[0])/np.timedelta64(1, 's') for i in range(len(acc))]

input_file = join('C:\\Users\\zwaks\\Desktop\\apdm-june2017\\param6_2000ev_tpe',
                  'gait_measures_all.csv')

data = pd.read_csv(input_file)

algs = [data.columns[i] for i in range(len(data.columns)) if 'idx_' in data.columns[i]]
for alg in algs:
    ts_idx = []
    for j in range(len(df)):
        if j not in data['SampleId'].tolist():
            ts_idx.append([])
            continue
        data_idx = data[data['SampleId'] == j].index[0]
        idx = string_to_int_list(data[alg][data_idx])
        ts_idx.append(ts[j][idx].tolist())
    t = pd.Series(ts_idx, name=alg)
    df = pd.concat([df, t], axis=1)


gait_stages = ['initial', 'off']
for gait_stage in gait_stages:
    res = pd.DataFrame(index=range(len(df)))
    for alg in algs:
        alg_series = []
        for j in range(len(df)):
            cell_vals = []
            if j not in data['SampleId'].tolist():
                alg_series.append(cell_vals)
                continue
            if np.any(np.isnan(df[gait_stage][j])):
                alg_series.append(cell_vals)
                continue
            a = df[alg][j]
            b = np.asanyarray(df[gait_stage][j])
            for i in range(len(a)):
                dt = np.negative(b - a[i])
                actual_value = dt[(np.abs(dt)).argmin()]
                cell_vals.append(actual_value)
            alg_series.append(cell_vals)
        t = pd.Series(alg_series, name=alg)
        res = pd.concat([res, t], axis=1)
    if gait_stage == 'initial':
        on = res
    if gait_stage == 'off':
        off = res

# data for boxplots
lhs_on = [item for sublist in on['idx_lhs'].tolist() for item in sublist]
lhs_off = [item for sublist in off['idx_lhs'].tolist() for item in sublist]

rhs_on = [item for sublist in on['idx_rhs'].tolist() for item in sublist]
rhs_off = [item for sublist in off['idx_rhs'].tolist() for item in sublist]

ov_st_on = [item for sublist in on['idx_overlap_strong'].tolist() for item in sublist]
ov_st_off = [item for sublist in off['idx_overlap_strong'].tolist() for item in sublist]

plt.boxplot([lhs_off, lhs_on])
plt.ylim(-0.4, 0.4)

plt.boxplot([ov_st_off, ov_st_on], 0, '')
plt.ylim(-0.4, 0.4)

A= [lhs_off, lhs_on]
B = [ov_st_off, ov_st_on]
C = [rhs_off, rhs_on]
all = A + B + C


_, pval = ks_2samp(lhs_off, lhs_off)

plt.boxplot(all, 0, '', positions= [1, 1.4, 2, 2.4, 3, 3.4])
plt.ylim(-0.4, 0.4)
