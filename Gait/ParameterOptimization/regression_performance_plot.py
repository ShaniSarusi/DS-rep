import pickle
from math import sqrt
from os.path import join, dirname
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import Gait.Resources.config as c
from Utils.BasicStatistics.statistics_functions import mean_absolute_percentage_error


def create_regression_performance_plot(data_file, metric, save_name='alg_performance.png', rotate=True, show_plot=False,
                                       set_y_lim=True, y_min=0, y_max=30):
    # Set tasks
    with open(join(c.pickle_path, 'task_filters'), 'rb') as fp:
        task_filters = pickle.load(fp)
    tasks = ['all'] + task_filters['Task Name'].tolist()
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
        algs.insert(0, algs.pop(algs.index('sc_combined')))
        algs.insert(0, algs.pop(algs.index('sc_rhs')))
        algs.insert(0, algs.pop(algs.index('sc_lhs')))

        with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
            sample = pickle.load(fp)
        sample['TaskName'] = sample['TaskName'].replace('Walk - Imagine you have a cane in the right hand',
                                                        'Asymmetry - Imagine you have a cane in the right hand')
        sample['TaskName'] = sample['TaskName'].replace('Walk - Without right shoe', 'Asymmetry - No right shoe')

        for i in range(len(algs)):
            means_i = []
            stds_i = []
            for j in range(len(tasks)):
                if tasks[j] == 'all':
                    sample_ids = data['SampleId'].as_matrix()
                else:
                    sample_ids = sample[sample['TaskName'] == tasks[j]]['SampleId'].as_matrix()
                alg_vals = data.loc[data['SampleId'].isin(sample_ids)][algs[i]].as_matrix()
                true_vals = data.loc[data['SampleId'].isin(sample_ids)][true_label].as_matrix()

                if metric == 'PE':  # percent error
                    vals = 100 * (alg_vals - true_vals) / true_vals
                    mean_val = vals.mean()
                    std_val = vals.std()
                elif metric == 'MAPE':
                    mean_val, std_val = mean_absolute_percentage_error(true_vals, alg_vals, handle_zeros=True,
                                                                       return_std=True)
                else:  # default is 'RMSE'
                    mean_val = sqrt(mean_squared_error(true_vals, alg_vals))
                    std_val = 0

                means_i.append(mean_val)
                stds_i.append(std_val)
            means.append(means_i)
            stds.append(stds_i)

    # Plotting
    fig, ax = plt.subplots()

    # Set legend
    algs = [alg.replace('sc_', '') for alg in algs]
    algs = ['Left side only' if alg == 'lhs' else alg for alg in algs]
    algs = ['Right side only' if alg == 'rhs' else alg for alg in algs]
    algs = ['Fusion - Sum raw signal' if 'sum' in alg else alg for alg in algs]
    algs = ['Fusion - Diff raw signal' if 'diff' in alg else alg for alg in algs]
    algs = ['Fusion - Intersection of steps' if 'intersect' in alg else alg for alg in algs]
    algs = ['Fusion - Union of steps' if 'union' in alg else alg for alg in algs]
    legends = algs

    groups = means
    errors = stds
    colors = ['r', 'g', 'b', 'm', 'k']

    x = range(len(groups[0]))
    list_dots = list()
    gap = 1.5
    for idx, group in enumerate(groups):
        left = [0.2 + i * gap + idx * 0.15 for i in x]
        dots = ax.errorbar(left, group, color=colors[idx], yerr=errors[idx], ecolor='k', capsize=5, fmt='o')
        list_dots.append(dots)

    # Legend
    list_dots = [list_dot[0] for list_dot in list_dots]
    ax.legend(list_dots, legends, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2)

    # set X tick labels
    ax.set_xticks([gap*i+0.4 for i in x])
    if rotate:
        ax.set_xticklabels(xlabs, rotation=45, fontsize=11)
    else:
        xlabs = ['Right\nbag' if x == ' Right bag' else x for x in xlabs]
        xlabs = ['Hands\non side' if x == 'Hands on side' else x for x in xlabs]
        ax.set_xticklabels(xlabs, fontsize=12)
    ax.set_xlim(0, gap*(max(x)+1))
    ax.set_xlabel('Task type', fontsize=16)
    if set_y_lim:
        ax.set_ylim(y_min, y_max)

    # Hide the right and top spines
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the bottom spine
    ax.xaxis.set_ticks_position('bottom')

    # y label
    if metric == 'PE':
        metric = 'Percent error'
    ax.set_ylabel(metric, fontsize=18)
    plt.yticks(fontsize=12)

    # y=0 line
    plt.axhline(0, color='black', linestyle='--')

    plt.subplots_adjust(top=0.9, right=0.9, bottom=0.2)
    # plt.tight_layout()
    save_path = join(dirname(data_file), save_name)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    plt.savefig(save_path)
    if show_plot:
        plt.show()

if __name__ == '__main__':
    input_file = join('C:\\Users\\zwaks\\Desktop\\apdm-june2017\\small_search_space', 'Summary.csv')

    input_file = join('C:\\Users\\zwaks\\Desktop\\apdm-june2017\\param_fast3_1000evals',
                'gait_measures_all.csv')

    input_file = join('C:\\Users\\zwaks\\Desktop\\apdm-june2017\\param6_2000ev_tpe',
                'gait_measures_all.csv')

    input_file = join('C:\\Users\\zwaks\\Desktop\\GaitPaper\\param_per5', 'gait_measures_all.csv')

    # input_file = join('C:\\Users\\zwaks\\Desktop\\GaitPaper\\param_per5',
    #               'gait_measures_split.csv')

    rotate = False
    metric = 'MAPE'  # 'MAPE' or 'RMSE'
    metric = 'PE'  # 'MAPE' or 'RMSE'
    show_plot = True
    save_name = 'pe_all2.png'
    # create_regression_performance_plot(input_file, metric, save_name, rotate, show_plot, y_min=-30)
    create_regression_performance_plot(input_file, metric, save_name, rotate, show_plot, y_min=-30, y_max=30)

