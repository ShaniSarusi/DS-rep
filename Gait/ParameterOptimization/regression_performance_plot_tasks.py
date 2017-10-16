import pickle
from math import sqrt
from os.path import join, sep
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import Gait.Resources.config as c
from Utils.BasicStatistics.statistics_functions import mean_absolute_percentage_error
import numpy as np


def create_regression_performance_plot(data_file, metric, save_name='alg_performance.png', rotate=True, show_plot=False,
                                       set_y_lim=True, y_min=0, y_max=30, inp2=None):
    # Set tasks
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

    # Set plot values
    means = []
    stds = []
    vals = []
    algs = None
    data = pd.read_csv(data_file)
    if 'Algorithm' in data.columns:
        analysis_type = 'folds'
    elif 'sc_manual' in data.columns:
        analysis_type = 'all'
    else:
        analysis_type = 'unknown'

    if analysis_type == 'folds':
        algs = data['Algorithm'].unique().tolist()
        algs.sort()
        for i in range(len(algs)):
            means_i = []
            stds_i = []
            for j in range(len(tasks)):
                vals = data[(data['Algorithm'] == algs[i]) & (data['WalkingTask'] == tasks[j])][metric]
                means_i.append(vals.mean())
                stds_i.append(vals.std())
            means.append(means_i)
            stds.append(stds_i)
    if analysis_type == 'all':
        algs = [data.columns[i] for i in range(len(data.columns)) if 'sc_' in data.columns[i]]
        true_label = 'sc_manual'
        if true_label in algs: algs.remove(true_label)
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
        algs.pop(algs.index('sc_fusion_high_level_union_one_stage'))
        algs.pop(algs.index('sc_lhs'))
        algs.pop(algs.index('sc_fusion_high_level_intersect'))
        algs.pop(algs.index('sc_fusion_low_level_diff'))
        #and
        algs.pop(algs.index('sc_fusion_low_level_sum'))

        with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
            sample = pickle.load(fp)
        sample['TaskName'] = sample['TaskName'].replace('Walk - Imagine you have a cane in the right hand',
                                                        'Asymmetry - Imagine you have a cane in the right hand')
        sample['TaskName'] = sample['TaskName'].replace('Walk - Without right shoe', 'Asymmetry - No right shoe')

        for j in range(len(tasks)):
            means_i = []
            stds_i = []
            for i in range(len(algs)):
                if tasks[j] == 'all':
                    sample_ids = data['SampleId'].as_matrix()
                else:
                    sample_ids = sample[sample['TaskName'] == tasks[j]]['SampleId'].as_matrix()
                if inp2 is not None:
                    true_vals = sample.loc[np.intersect1d(data['SampleId'], sample_ids)]['CadenceApdmMean'].as_matrix()
                    alg_vals = data.loc[data['SampleId'].isin(sample_ids)][algs[i].replace('sc_', 'cadence_apdm_')].as_matrix()
                else:
                    true_vals = data.loc[data['SampleId'].isin(sample_ids)][true_label].as_matrix()
                    alg_vals = data.loc[data['SampleId'].isin(sample_ids)][algs[i]].as_matrix()

                if metric == 'PE':  # percent error
                    vals_j = 100 * (alg_vals - true_vals) / true_vals
                    mean_val = vals_j.mean()
                    std_val = vals_j.std()
                elif metric == 'MAPE':
                    mean_val, std_val = mean_absolute_percentage_error(true_vals, alg_vals, handle_zeros=True,
                                                                       return_std=True)
                else:  # default is 'RMSE'
                    mean_val = sqrt(mean_squared_error(true_vals, alg_vals))
                    std_val = 0

                means_i.append(mean_val)
                stds_i.append(std_val)
                vals.append(vals_j)
            means.append(means_i)
            stds.append(stds_i)

    # Plotting
    fig, ax = plt.subplots()

    # Set legend
    # algs = [alg.replace('sc_', '') for alg in algs]
    # algs = ['Left side only' if alg == 'lhs' else alg for alg in algs]
    # algs = ['Right side only' if alg == 'rhs' else alg for alg in algs]
    # algs = ['Fusion - Sum raw signal' if 'sum' in alg else alg for alg in algs]
    # algs = ['Fusion - Diff raw signal' if 'diff' in alg else alg for alg in algs]
    # algs = ['Fusion - Intersection of steps' if 'intersect' in alg else alg for alg in algs]
    # algs = ['Fusion - Union two stages' if 'two_sta' in alg else alg for alg in algs]
    # algs = ['Fusion - Union one stage' if 'one_sta' in alg else alg for alg in algs]

    algs = [alg.replace('sc_', '') for alg in algs]
    algs = ['Left side only' if alg == 'lhs' else alg for alg in algs]
    algs = ['Right only' if alg == 'rhs' else alg for alg in algs]
    algs = ['Sum raw signal' if 'sum' in alg else alg for alg in algs]
    algs = ['Fusion - Diff raw signal' if 'diff' in alg else alg for alg in algs]
    algs = ['Fusion - Intersection of steps' if 'intersect' in alg else alg for alg in algs]
    algs = ['Union of events' if 'two_sta' in alg else alg for alg in algs]
    algs = ['Union of events' if 'one_sta' in alg else alg for alg in algs]
    legends = algs

    if len(algs) == 2:
        colors = ['r', 'b']
    else:
        colors = ['r', 'g', 'b']

    x = []
    gap = 3
    inner_gap = 0.6
    xtick_locs = []
    for i in range(len(tasks)):
        xtick_locs.append(gap*i + 0.5*len(algs)*inner_gap)
        for j in range(len(algs)):
            x_tmp = i*gap + j*inner_gap
            x.append(x_tmp)

    box = plt.boxplot(vals, 0, '', positions=x, patch_artist=True)
    xlabs = ['Right\nbag' if x == ' Right bag' else x for x in xlabs]
    xlabs = ['Hands\non side' if x == 'Hands on side' else x for x in xlabs]
    plt.xticks(xtick_locs, xlabs)

    plt.yticks(fontsize=10)
    if inp2 is not None:
        ylab = 'Cadence\n(percent difference)'
    else:
        ylab = 'Step count\n(percent error)'
    plt.ylabel(ylab, fontsize=11)
    plt.tight_layout()
    ax = fig.gca()
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    if len(algs) == 2:
        colors = ['cyan', 'pink'] * len(tasks)
    else:
        colors = ['cyan', 'lightgreen', 'pink'] * len(tasks)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.setp(box['medians'], color='k')
    plt.legend(algs)
    ax = plt.gca()
    leg = ax.get_legend()
    for i in range(len(algs)):
        leg.legendHandles[i].set_color(colors[i])

    fig = plt.gcf()
    fig.set_size_inches(7.2, 3.2)
    fig.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()



if __name__ == '__main__':
    dirpath = join('C:', sep, 'Users', 'zwaks', 'Desktop', 'GaitPaper')
    #input_file = join(dirpath,'aa_param3small_10k_big_sc_1008_v1', 'gait_measures.csv')
    input_file = join(dirpath, 'aa_param3small_5000_195_sc_1008_v1', 'gait_measures.csv')
    inp2 = join(c.pickle_path, 'metadata_sample')
    rotate = False
    metric = 'PE'  # 'MAPE' or 'RMSE'
    show_plot = True
    save_name = join(dirpath, 'sc_tasks.png')
    # create_regression_performance_plot(input_file, metric, save_name, rotate, show_plot, y_min=-30)
    create_regression_performance_plot(input_file, metric, save_name, rotate, show_plot, y_min=-30, y_max=30, inp2=inp2)





