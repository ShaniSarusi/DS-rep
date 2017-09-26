import pickle
from os.path import join, sep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Gait.Resources.config as c
from Utils.DataHandling.data_processing import string_to_int_list


# APDM events TODO need to sort
with open(join(c.pickle_path, 'apdm_events'), 'rb') as fp:
    apdm_events = pickle.load(fp)
apdm_events['initial'] = apdm_events['Gait - Lower Limb - Initial Contact L (s)'] + apdm_events['Gait - Lower Limb - Initial Contact R (s)']
apdm_events['initial_lhs'] = apdm_events['Gait - Lower Limb - Initial Contact L (s)']
apdm_events['initial_rhs'] = apdm_events['Gait - Lower Limb - Initial Contact R (s)']
apdm_events['off'] = apdm_events['Gait - Lower Limb - Toe Off L (s)'] + apdm_events['Gait - Lower Limb - Toe Off R (s)']
apdm_events['off_lhs'] = apdm_events['Gait - Lower Limb - Toe Off L (s)']
apdm_events['off_rhs'] = apdm_events['Gait - Lower Limb - Toe Off R (s)']
df = pd.concat([apdm_events['initial'], apdm_events['initial_lhs'], apdm_events['initial_rhs'], apdm_events['off'],
                apdm_events['off_lhs'], apdm_events['off_rhs']], axis=1)

# Wrist events******************************************
# Get time stamps
with open(join(c.pickle_path, 'acc'), 'rb') as fp:
    acc = pickle.load(fp)
    ts = [(acc[i]['lhs']['ts'] - acc[i]['lhs']['ts'].iloc[0])/np.timedelta64(1, 's') for i in range(len(acc))]

save_dir = join('C:', sep, 'Users', 'zwaks', 'Desktop', 'GaitPaper')
input_file = join(save_dir, 'aa_param1_1k_sc_09262017', 'gait_measures.csv')
data = pd.read_csv(input_file)
max_dist_between_apdm_to_wrist_alg = 0.3

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


apdm_gait_stages = ['initial', 'initial_lhs', 'initial_rhs', 'off', 'off_lhs', 'off_rhs']
for apdm_gait_stage in apdm_gait_stages:
    res = pd.DataFrame(index=range(len(df)))
    for alg in algs:
        alg_series = []
        for j in range(len(df)):
            cell_vals = []
            if j not in data['SampleId'].tolist():
                alg_series.append(cell_vals)
                continue
            if np.any(np.isnan(df[apdm_gait_stage][j])):
                alg_series.append(cell_vals)
                continue
            alg_event = df[alg][j]
            apdm_event = np.asanyarray(df[apdm_gait_stage][j])

            # option one - Distance from each APDM event detected
            if len(alg_event) > 0:
                for i in range(len(apdm_event)):
                    dt = alg_event - apdm_event[i]
                    min_dist_from_apdm_event_i = dt[(np.abs(dt)).argmin()]
                    if np.abs(min_dist_from_apdm_event_i) > max_dist_between_apdm_to_wrist_alg:
                        continue
                    cell_vals.append(min_dist_from_apdm_event_i*1000)

            # option two - distance from each alg event
            # for i in range(len(alg_event)):
            #     dt = np.negative(apdm_event - alg_event[i])
            #     min_dist_from_alg_event_i = dt[(np.abs(dt)).argmin()]
            #     cell_vals.append(min_dist_from_alg_event_i)

            alg_series.append(cell_vals)
        t = pd.Series(alg_series, name=alg)
        res = pd.concat([res, t], axis=1)
    # Save res data
    if apdm_gait_stage == 'initial':
        initial = res
    if apdm_gait_stage == 'initial_lhs':
        initial_lhs = res
    if apdm_gait_stage == 'initial_rhs':
        initial_rhs = res
    if apdm_gait_stage == 'off':
        off = res
    if apdm_gait_stage == 'off_lhs':
        off_lhs = res
    if apdm_gait_stage == 'off_rhs':
        off_rhs = res

# alg to plot
alg_to_plot = 'lhs'
alg_to_plot = 'fusion_high_level_union'

on_lhs_bp = [item for sublist in initial_lhs['idx_' + alg_to_plot].tolist() for item in sublist]
on_rhs_bp = [item for sublist in initial_rhs['idx_' + alg_to_plot].tolist() for item in sublist]
off_lhs_bp = [item for sublist in off_lhs['idx_' + alg_to_plot].tolist() for item in sublist]
off_rhs_bp = [item for sublist in off_rhs['idx_' + alg_to_plot].tolist() for item in sublist]
plt.boxplot([off_lhs_bp, off_rhs_bp, on_lhs_bp, on_rhs_bp], 0, '', positions=[1, 1.5, 2, 2.5],
            labels=['Toe off-L', 'Toe off-R', 'Initial-L', 'Initial-R'])
plt.ylabel('Time from nearest wrist event\n(ms)', fontsize=16)
plt.xlabel('Leg detected gait events', fontsize=18)
plt.ylim(-125, 250)
plt.yticks(fontsize=12)
plt.xticks(fontsize=14)
#plt.axhline(0, color='black', linestyle='--')
plt.gca().yaxis.grid(True)
plt.tight_layout()
plt.savefig('gait_stage_' + alg_to_plot + '.png')


alg1 = 'lhs'
alg2 = 'fusion_high_level_union'
off_lhs_alg1 = [item for sublist in off_lhs['idx_' + alg1].tolist() for item in sublist]
off_rhs_alg1 = [item for sublist in off_rhs['idx_' + alg1].tolist() for item in sublist]
off_lhs_alg2 = [item for sublist in off_lhs['idx_' + alg2].tolist() for item in sublist]
off_rhs_alg2 = [item for sublist in off_rhs['idx_' + alg2].tolist() for item in sublist]
a = plt.boxplot([off_lhs_alg1, off_lhs_alg2, off_rhs_alg1, off_rhs_alg2], 0, '', positions=[1, 1.4, 2, 2.4],
            labels=['Lhs', 'Union', 'Lhs', 'Union'], patch_artist=True)

colors = ['pink', 'lightblue', 'pink', 'lightblue']
for patch, color in zip(a['boxes'], colors):
    patch.set_facecolor(color)
plt.ylabel('\u0394t from nearest wrist event\n(ms)', fontsize=16)
plt.xlabel('Left toe             Right toe', fontsize=18)
plt.ylim(-100, 75)
plt.axhline(0, color='black', linestyle='--')
plt.yticks(fontsize=12)
plt.xticks(fontsize=14)
plt.gca().yaxis.grid(True)
plt.tight_layout()
plt.savefig('gait_stage_comparison.png')


#### Plotting personal
with open(join(c.pickle_path, 'apdm_measures'), 'rb') as fp:
    apdm_measures_tmp = pickle.load(fp)
    apdm_measures = apdm_measures_tmp.iloc[[i for i in data['SampleId'].tolist()]]
    apdm_vals = apdm_measures['apdm_toe_off_asymmetry_median']
with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
    sample = pickle.load(fp)

df2 = pd.DataFrame(columns=['SampleId', 'l_mean', 'r_mean', 'l_med', 'r_med', 'l_std', 'r_std', 'apdm_asym', 'alg_asym', 'walk_task'], index=data['SampleId'])
for id in data['SampleId']:
    l = off_lhs.iloc[id]['idx_fusion_high_level_union']
    r = off_rhs.iloc[id]['idx_fusion_high_level_union']
    row = [id, np.mean(l), np.mean(r), np.median(l), np.median(r), np.std(l), np.std(r), apdm_vals.loc[id],
           data[data['SampleId'] == id]['step_time_asymmetry_median_fusion_high_level_union'].iloc[0], sample['TaskName'].iloc[id]]
    df2.loc[id] = row


df2.to_csv(join(save_dir, 'asymmetry_evaluation.csv'), index=False)


