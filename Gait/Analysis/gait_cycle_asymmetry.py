import pickle
from os.path import join, sep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Gait.Resources.config as c
from Utils.DataHandling.data_processing import string_to_int_list, string_to_float_list
from scipy.stats import pearsonr
from Gait.Resources.gait_utils import bp_me
from scipy.stats import pearsonr


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
alg_to_plot = 'fusion_high_level_union_two_stages'
alg_to_plot = 'fusion_high_level_union_one_stage'


just_union = []
just_union.append([item/1000.0 for sublist in initial_lhs['idx_fusion_high_level_union_one_stage'].tolist() for item in sublist])
just_union.append([item/1000.0 for sublist in initial_rhs['idx_fusion_high_level_union_one_stage'].tolist() for item in sublist])
just_union.append([item/1000.0 for sublist in off_lhs['idx_fusion_high_level_union_one_stage'].tolist() for item in sublist])
just_union.append([item/1000.0 for sublist in off_rhs['idx_fusion_high_level_union_one_stage'].tolist() for item in sublist])
labels=['Heel-L', 'Heel-R', 'Toe off-L', 'Toe off-R']
fig, ax = plt.subplots()
x = [1, 1.5, 2.5, 3]
box = plt.boxplot([just_union[0], just_union[1], just_union[2], just_union[3]], 0, '', positions=x,
                  labels=labels, widths=0.4, whis=[5, 95], patch_artist=True)
colors = ['cyan', 'cyan', 'lightgreen', 'lightgreen']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.setp(box['medians'], color='k')
plt.yticks(fontsize=10)
plt.ylabel('\u0394t after nearest toe off event (s)', fontsize=11)
plt.xticks(fontsize=9)
plt.tight_layout()
ax = fig.gca()
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
fig = plt.gcf()
fig.set_size_inches(3.5, 2.75)
fig.tight_layout()
plt.savefig(join(save_dir, 'gait_phase_just_union.png'), dpi=300)
plt.show()



#NEW
algs = ['lhs', 'rhs', 'fusion_low_level_sum', 'fusion_low_level_diff', 'fusion_high_level_intersect', 'fusion_high_level_union_one_stage']
heel_l = []
for alg in algs:
    tmp = [item/1000.0 for sublist in initial_lhs['idx_' + alg].tolist() for item in sublist]
    heel_l.append(tmp)
bp_me(heel_l, save_name=join(save_dir, 'heel_L.png'), ylabel='\u0394t after nearest heel strike (s)')

heel_r = []
for alg in algs:
    tmp = [item/1000.0 for sublist in initial_rhs['idx_' + alg].tolist() for item in sublist]
    heel_r.append(tmp)
bp_me(heel_r, save_name=join(save_dir, 'heel_R.png'), ylabel='\u0394t after nearest heel strike (s)')

toe_l = []
for alg in algs:
    tmp = [item/1000.0 for sublist in off_lhs['idx_' + alg].tolist() for item in sublist]
    toe_l.append(tmp)
bp_me(toe_l, save_name=join(save_dir, 'toe_L.png'), ylabel='\u0394t after nearest toe off event (s)')

toe_r = []
for alg in algs:
    tmp = [item/1000.0 for sublist in off_rhs['idx_' + alg].tolist() for item in sublist]
    toe_r.append(tmp)
bp_me(toe_r, save_name=join(save_dir, 'toe_R.png'), ylabel='\u0394t after nearest toe off event (s)')


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

alg = 'fusion_high_level_union_one_stage'
for i in range(len(tasks)):
    sample_ids = sample[sample['TaskName'] == tasks[i]]['SampleId'].as_matrix()
    off_tmp = off['idx_' + alg].iloc[sample_ids].as_matrix()
    tmp = [item for sublist in off_tmp for item in sublist]
    toe.append(tmp)
#boxplot here
fig, ax = plt.subplots()
vals = toe
x = [1, 2,3,4,5,6,7,8]
box = plt.boxplot([vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]], 0, '', positions=x,
            labels=xlabs, widths=0.4, whis=[5, 95], patch_artist=True)
# colors = ['cyan', 'cyan', 'lightgreen', 'lightgreen', 'pink', 'pink']
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
plt.setp(box['medians'], color='k')
plt.yticks(fontsize=10)
plt.ylabel('\u0394t from toe off (ms)', fontsize=11)
plt.xticks(fontsize=9)
plt.tight_layout()
ax = fig.gca()
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

fig = plt.gcf()
fig.set_size_inches(6, 3)
fig.tight_layout()
plt.savefig(join(save_dir, 'toe_tasks_union.png'), dpi=300)
plt.show()




# Per person
alg = 'fusion_high_level_union_one_stage'
toe = []
people = sample['Person'].unique()
std_sort=[]
for i in range(len(people)):
    sample_ids = sample[sample['Person'] == people[i]]['SampleId'].as_matrix()
    off_tmp = off['idx_' + alg].iloc[sample_ids].as_matrix()
    tmp = [item for sublist in off_tmp for item in sublist]
    std_sort.append(np.std(tmp))
    toe.append(tmp)
toe = np.array(toe)
a = np.argsort(std_sort)
toe = toe[a]
#boxplot here
fig, ax = plt.subplots()
vals = toe.tolist()
box = plt.boxplot(vals, 0, '', widths=0.4, patch_artist=True)
# colors = ['cyan', 'cyan', 'lightgreen', 'lightgreen', 'pink', 'pink']
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
plt.setp(box['medians'], color='k')
plt.yticks(fontsize=10)
plt.ylabel('\u0394t from toe off (ms)', fontsize=11)
plt.xticks([])
plt.xlabel('People sorted by variability of \u0394t')
plt.tight_layout()
ax = fig.gca()
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

fig = plt.gcf()
fig.set_size_inches(6, 3)
fig.tight_layout()
plt.savefig(join(save_dir, 'toe_people_union.png'), dpi=300)
plt.show()


#scatter diff l-r
alg = 'fusion_high_level_union_one_stage'
med_l = np.array([np.median(x) for x in off_lhs['idx_' + alg].as_matrix()])
med_r = np.array([np.median(x) for x in off_rhs['idx_' + alg].as_matrix()])
diff = med_l - med_r
diff = diff[~np.isnan(diff)]
plt.hist(diff, bins=30)
plt.ylabel('count', fontsize=11)
plt.xlabel('\u0394t toe off offset between L and R\n(median_l - median_r) (ms)', fontsize=11)
fig = plt.gcf()
fig.set_size_inches(4, 3)
fig.tight_layout()
plt.savefig(join(save_dir, 'diff_union_hist.png'), dpi=300)
plt.show()




#Stride time var correlation....
input_file = join(save_dir, 'a_cad_param3small_1000_1017_v1', 'gait_measures.csv')
alg = 'fusion_high_level_union_one_stage'
#alg = 'rhs'
a = pd.read_csv(input_file)
ids = a['SampleId']
strides = a['stride_durations_all_' + alg]
alg_var = []
# for j in range(50,91):
strides = a['stride_durations_all_' + alg]
alg_var = []
pctile_high = 100
pctile_low = 0
for i in range(len(strides)):
    strides_i = np.array(string_to_float_list(strides[i]))
    tmp = strides_i
    if pctile_high < 100:
        tmp = tmp[tmp < np.percentile(tmp, pctile_high)]
    if pctile_low > 0:
        tmp = tmp[tmp > np.percentile(tmp, pctile_low)]
    cv = np.std(tmp)/np.mean(tmp)
    alg_var.append(cv)

with open(join(c.pickle_path, 'apdm_measures'), 'rb') as fp:
    apdm_measures = pickle.load(fp)
apdm_var = [np.mean([apdm_measures['stride_time_var_lhs'].iloc[i], apdm_measures['stride_time_var_rhs'].iloc[i]])
            for i in range(len(apdm_measures))]
apdm_var = [apdm_var[i] for i in ids]

diff_var = np.array(alg_var) - np.array(apdm_var)
std_l = [np.std(off_lhs['idx_' + alg][i]) for i in ids]
std_r = [np.std(off_rhs['idx_' + alg][i]) for i in ids]
std_lr = [std_l[i] + std_r[i] for i in range(len(std_l))]
std_b = [np.std(off['idx_' + alg][i]) for i in ids]
r, _ = pearsonr(diff_var, std_b)
print(r)
plt.scatter(diff_var, std_b)

r, _ = pearsonr(apdm_var, alg_var)
print('pctile ' + str(pctile_high) + ' corr: ' + str(r))

ids = [i for i in range(len(alg_var)) if alg_var[i] < 0.12]
alg_var_small = [alg_var[i] for i in ids]
apdm_var_small = [apdm_var[i] for i in ids]
r, _ = pearsonr(apdm_var_small, alg_var_small)
print(r)








# OLD
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
plt.savefig(join(save_dir, 'gait_stage_' + alg_to_plot + '.png'))


# alg1 = 'lhs'
# alg2 = 'fusion_high_level_union_two_stages'
# off_lhs_alg1 = [item for sublist in off_lhs['idx_' + alg1].tolist() for item in sublist]
# off_rhs_alg1 = [item for sublist in off_rhs['idx_' + alg1].tolist() for item in sublist]
# off_lhs_alg2 = [item for sublist in off_lhs['idx_' + alg2].tolist() for item in sublist]
# off_rhs_alg2 = [item for sublist in off_rhs['idx_' + alg2].tolist() for item in sublist]
# a = plt.boxplot([off_lhs_alg1, off_lhs_alg2, off_rhs_alg1, off_rhs_alg2], 0, '', positions=[1, 1.4, 2, 2.4],
#             labels=['Lhs', 'Union', 'Lhs', 'Union'], patch_artist=True)
#
# colors = ['pink', 'lightblue', 'pink', 'lightblue']
# for patch, color in zip(a['boxes'], colors):
#     patch.set_facecolor(color)
# plt.ylabel('\u0394t from nearest wrist event\n(ms)', fontsize=16)
# plt.xlabel('Left toe             Right toe', fontsize=18)
# plt.ylim(-100, 75)
# plt.axhline(0, color='black', linestyle='--')
# plt.yticks(fontsize=12)
# plt.xticks(fontsize=14)
# plt.gca().yaxis.grid(True)
# plt.tight_layout()
# plt.savefig(join(save_dir, 'gait_stage_comparison.png'))


#### Plotting personal
with open(join(c.pickle_path, 'apdm_measures'), 'rb') as fp:
    apdm_measures_tmp = pickle.load(fp)
    apdm_measures = apdm_measures_tmp.iloc[[i for i in data['SampleId'].tolist()]]
    apdm_vals = apdm_measures['apdm_toe_off_asymmetry_median']
with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
    sample = pickle.load(fp)

alg_name = 'fusion_high_level_union_two_stages'
alg_name = 'fusion_high_level_union_one_stage'
df2 = pd.DataFrame(columns=['SampleId', 'l_all', 'r_all', 'l_mean', 'r_mean', 'l_med', 'r_med', 'l_std', 'r_std', 'apdm_asym', 'alg_asym',
                            'diff', 'sc_pct_error', 'walk_task'], index=data['SampleId'])
for id in data['SampleId']:
    l = off_lhs.iloc[id]['idx_' + alg_name]
    r = off_rhs.iloc[id]['idx_' + alg_name]

    apdm_asym = apdm_vals.loc[id]
    alg_asym = data[data['SampleId'] == id]['step_time_asymmetry_median_' + alg_name].iloc[0]
    diff = alg_asym - apdm_asym

    sc_manual = data[data['SampleId'] == id]['sc_manual'].iloc[0]
    sc_alg = data[data['SampleId'] == id]['sc_' + alg_name].iloc[0]
    err = (sc_alg - sc_manual)/sc_manual

    l.sort()
    r.sort()
    row = [id, str(l), str(r), np.mean(l), np.mean(r), np.median(l), np.median(r), np.std(l), np.std(r), apdm_asym,
           alg_asym, diff, err, sample['TaskName'].iloc[id]]

    df2.loc[id] = row
df2.to_csv(join(save_dir, 'asymmetry_evaluation.csv'), index=False)

df_corr = pd.DataFrame(columns=['walk_task', 'apdm_med', 'alg_med', 'corr'], index=range(len(df2['walk_task'].unique())))
df_corr['walk_task'] = df2['walk_task'].unique()
for i in range(len(df_corr)):
    task = df_corr.iloc[i]['walk_task']
    idx_bool = df2['walk_task'] == task
    a = df2[idx_bool]['apdm_asym'].as_matrix()
    b = df2[idx_bool]['alg_asym'].as_matrix()
    # values
    df_corr.iloc[i]['apdm_med'] = np.median(a)
    df_corr.iloc[i]['alg_med'] = np.median(b)
    df_corr.iloc[i]['corr'] = pearsonr(a, b)[0]

df_corr.to_csv(join(save_dir, 'asymmetry_corr_per_task.csv'), index=False)


