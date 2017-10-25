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
input_file = join(save_dir, 'a_cad_param3small_1000_1017_v1', 'gait_measures.csv')
input_file = join(save_dir, 'a_cad10k_param4small_oct22_final', 'gait_measures.csv')
data = pd.read_csv(input_file)
max_dist_between_apdm_to_wrist_alg = 0.5

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
just_union = list()
just_union.append([item/1000.0 for sublist in initial_lhs['idx_fusion_high_level_union_one_stage'].tolist() for item in sublist])
just_union.append([item/1000.0 for sublist in initial_rhs['idx_fusion_high_level_union_one_stage'].tolist() for item in sublist])
just_union.append([item/1000.0 for sublist in off_lhs['idx_fusion_high_level_union_one_stage'].tolist() for item in sublist])
just_union.append([item/1000.0 for sublist in off_rhs['idx_fusion_high_level_union_one_stage'].tolist() for item in sublist])
labels=['L', 'R', 'L', 'R']
fig, ax = plt.subplots()
x = [1, 1.5, 2.5, 3]
box = plt.boxplot([just_union[0], just_union[1], just_union[2], just_union[3]], 0, '', positions=x,
                  labels=labels, widths=0.4, whis=[5, 95], patch_artist=True)
colors = ['cyan', 'cyan', 'lightgreen', 'lightgreen']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.setp(box['medians'], color='k')
plt.yticks(np.arange(-0.1, 0.2, 0.05), fontsize=10)
plt.ylabel('\u0394t after event (s)', fontsize=11)
plt.xticks(fontsize=10)
plt.xlabel('Heel strike          Toe off')
ax.xaxis.set_label_coords(0.46, -0.17)
plt.tight_layout()
ax = fig.gca()
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
fig = plt.gcf()
fig.set_size_inches(3, 2.5)
fig.tight_layout()
plt.savefig(join(save_dir, 'gait_phase_just_union.png'), dpi=600)
plt.show()


# Per task
tasks = ['aa', 'bb']
toe = []
with open(join(c.pickle_path, 'task_filters'), 'rb') as fp:
    task_filters = pickle.load(fp)
# tasks = ['all'] + task_filters['Task Name'].tolist()
tasks = task_filters['Task Name'].tolist()
tasks.remove('Tug')
tasks.remove('Tug')
a, b = tasks.index('Walk - Regular'), tasks.index('Walk - Fast')
tasks[b], tasks[a] = tasks[a], tasks[b]
xlabs = ['Cane' if x == 'Asymmetry - Imagine you have a cane in the right hand' else x for x in tasks]
xlabs = ['No shoe' if x == 'Asymmetry - No right shoe' else x for x in xlabs]
xlabs = ['Hands on side' if x == 'Walk - Both hands side' else x for x in xlabs]
xlabs = [x[6:] if 'Walk - ' in x else x for x in xlabs]
xlabs = ['Right\nbag' if x == ' Right bag' else x for x in xlabs]
xlabs = ['Hands\non side' if x == 'Hands on side' else x for x in xlabs]
with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
    sample = pickle.load(fp)
sample['TaskName'] = sample['TaskName'].replace('Walk - Imagine you have a cane in the right hand',
                                                'Asymmetry - Imagine you have a cane in the right hand')
sample['TaskName'] = sample['TaskName'].replace('Walk - Without right shoe', 'Asymmetry - No right shoe')
for i in range(len(tasks)):
    sample_ids = sample[sample['TaskName'] == tasks[i]]['SampleId'].as_matrix()
    off_tmp = off['idx_fusion_high_level_union_one_stage'].iloc[sample_ids].as_matrix()
    tmp = [item/1000.0 for sublist in off_tmp for item in sublist]
    toe.append(tmp)
#boxplot here
fig, ax = plt.subplots()
vals = toe
x = [1, 2, 3, 4, 5, 6, 7, 8]
box = plt.boxplot([vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]], 0, '', positions=x,
            labels=xlabs, widths=0.4, whis=[5, 95], patch_artist=True)
# colors = ['cyan', 'cyan', 'lightgreen', 'lightgreen', 'pink', 'pink']
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
plt.setp(box['medians'], color='k')
plt.yticks(np.arange(-0.15, 0.2, 0.05), fontsize=10)
plt.ylim(-0.15, 0.15)
plt.ylabel('\u0394t after toe off (s)', fontsize=11)
plt.xticks(fontsize=10)
plt.tight_layout()
ax = fig.gca()
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

fig = plt.gcf()
fig.set_size_inches(6, 2.5)
fig.tight_layout()
plt.savefig(join(save_dir, 'toe_tasks_union.png'), dpi=600)
plt.show()

