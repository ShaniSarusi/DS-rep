import pickle
from os.path import join, sep
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import Gait.Resources.config as c
import numpy as np


def plot_sc_error_by_task(data_file, save_name='sc_err_tasks.png', hide_splines=False, sc_label='apdm', whisk='5_95'):
    # Set tasks
    with open(join(c.pickle_path, 'task_filters'), 'rb') as fp:
        task_filters = pickle.load(fp)
        tasks = task_filters['Task Name'].tolist()
        tasks.remove('Tug')
        tasks.remove('Tug')
        a, b = tasks.index('Walk - Regular'), tasks.index('Walk - Fast')
        tasks[b], tasks[a] = tasks[a], tasks[b]
        x_labels = ['Cane' if x == 'Asymmetry - Imagine you have a cane in the right hand' else x for x in tasks]
        x_labels = ['No shoe' if x == 'Asymmetry - No right shoe' else x for x in x_labels]
        x_labels = ['Hands on side' if x == 'Walk - Both hands side' else x for x in x_labels]
        x_labels = [x[6:] if 'Walk - ' in x else x for x in x_labels]
        x_labels = ['Right\nbag' if x == ' Right bag' else x for x in x_labels]
        x_labels = ['Hands\non side' if x == 'Hands on side' else x for x in x_labels]

    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
        sample = pickle.load(fp)
        sample['TaskName'] = sample['TaskName'].replace('Walk - Imagine you have a cane in the right hand',
                                                        'Asymmetry - Imagine you have a cane in the right hand')
        sample['TaskName'] = sample['TaskName'].replace('Walk - Without right shoe', 'Asymmetry - No right shoe')

    # Set algorithms and their order
    data = pd.read_csv(data_file)
    algs = [data.columns[i] for i in range(len(data.columns)) if 'sc_' in data.columns[i]]
    if 'sc_manual' in algs: algs.remove('sc_manual')
    # Set order
    if 'sc_fusion_high_level_union_one_stage' in algs:
        algs.insert(0, algs.pop(algs.index('sc_fusion_high_level_union_one_stage')))
    if 'sc_fusion_high_level_union_two_stages' in algs:
        algs.insert(0, algs.pop(algs.index('sc_fusion_high_level_union_two_stages')))
    if 'sc_fusion_high_level_intersect' in algs:
        algs.insert(0, algs.pop(algs.index('sc_fusion_high_level_intersect')))
    if 'sc_fusion_low_level_diff' in algs:
        algs.insert(0, algs.pop(algs.index('sc_fusion_low_level_diff')))
    if 'sc_fusion_low_level_sum' in algs:
        algs.insert(0, algs.pop(algs.index('sc_fusion_low_level_sum')))
    if 'sc_rhs' in algs:
        algs.insert(0, algs.pop(algs.index('sc_rhs')))
    if 'sc_lhs' in algs:
        algs.insert(0, algs.pop(algs.index('sc_lhs')))
    # remove ***************************************************************
    remove = ['sc_fusion_high_level_union_two_stages', 'sc_lhs', 'sc_fusion_high_level_intersect',
              'sc_fusion_low_level_diff', 'sc_fusion_low_level_sum']
    for rem in remove:
        algs.pop(algs.index(rem))

    # Calculate value to plot
    vals = []
    for j in range(len(tasks)):
        for i in range(len(algs)):
            sample_ids = sample[sample['TaskName'] == tasks[j]]['SampleId'].as_matrix()
            if sc_label == 'apdm':
                sample_ids = sample_ids[sample_ids != 66]
                true_vals = sample.loc[np.intersect1d(data['SampleId'], sample_ids)]['CadenceApdmMean'].as_matrix()
                alg_vals = data.loc[data['SampleId'].isin(sample_ids)][algs[i].replace('sc_', 'cadence_apdm_')].as_matrix()
            else:
                true_vals = data.loc[data['SampleId'].isin(sample_ids)]['sc_manual'].as_matrix()
                alg_vals = data.loc[data['SampleId'].isin(sample_ids)][algs[i]].as_matrix()
            vals.append(100 * (alg_vals - true_vals) / true_vals)

    # Plotting
    # x locations
    x = []
    gap = 3
    inner_gap = 0.6
    xtick_locs = []
    for i in range(len(tasks)):
        xtick_locs.append(gap*i + 0.5*len(algs)*inner_gap)
        for j in range(len(algs)):
            x_tmp = i*gap + j*inner_gap
            x.append(x_tmp)

    fig, ax = plt.subplots()
    if whisk == '5_95':
        box = plt.boxplot(vals, 0, '', positions=x, patch_artist=True, whis=[5, 95])
    else:
        box = plt.boxplot(vals, 0, '+', positions=x, patch_artist=True)
    plt.xticks(xtick_locs, x_labels, fontsize=10)

    # Y axis stuff
    plt.yticks(np.arange(-15, 30, 5), fontsize=10)
    plt.ylabel('Step count error (%)', fontsize=10)
    # for label in ax.yaxis.get_ticklabels()[::2]:
    #     label.set_visible(False)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    plt.ylim(-15, 20)

    col = ['#1f77b4', '#ff7f0e']
    if len(algs) == 2:
        colors = col * len(tasks)
    else:
        colors = ['cyan', 'lightgreen', 'pink'] * len(tasks)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.setp(box['medians'], color='k')

    # Set legend
    right = mpatches.Patch(facecolor=col[0], label='Right only (no fusion)', edgecolor='black')
    union = mpatches.Patch(facecolor=col[1], label='Union (high-level) fusion', edgecolor='black')
    plt.legend(handles=[right, union], fontsize=9.5)

    # Hide the right and top spines
    ax = plt.gca()
    if hide_splines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    fig = plt.gcf()
    fig.set_size_inches(5.5, 2.2)
    fig.tight_layout()
    plt.savefig(save_name, dpi=600)
    plt.show()


if __name__ == '__main__':
    dirpath = join('C:', sep, 'Users', 'zwaks', 'Desktop', 'GaitPaper')
    input_file = join(dirpath, 'a_cad10k_param4small_oct22_final', 'gait_measures.csv')
    save_name = join(dirpath, 'sc_err_tasks.png')
    plot_sc_error_by_task(input_file, save_name, hide_splines=True)
