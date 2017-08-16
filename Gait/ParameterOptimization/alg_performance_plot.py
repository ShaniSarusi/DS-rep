import pickle
from os.path import join, dirname
import matplotlib.pyplot as plt
import pandas as pd
import Gait.Resources.config as c
from math import sqrt
from Utils.BasicStatistics.statistics_functions import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error


def create_alg_performance_plot(data_file, metric, save_name='alg_performance.png', rotate=True, show_plot=False,
                                set_y_lim=True, y_min=0, y_max=30, analysis_type='folds'):

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
        algs.sort()

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

                if metric == 'MAPE':
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
    legends = algs
    groups = means
    errors = stds
    colors = ['r', 'g', 'b', 'm', 'k']

    x = range(len(groups[0]))
    list_dots = list()
    for idx, group in enumerate(groups):
        left = [0.2 + i+idx*0.15 for i in x]
        dots = ax.errorbar(left, group, color=colors[idx], yerr=errors[idx], ecolor='k', capsize=5, fmt='o')
        list_dots.append(dots)

    # Legend
    ax.legend(list_dots, legends, loc='upper right')

    # set X tick labels
    ax.set_xticks([i+0.4 for i in x])
    if rotate:
        ax.set_xticklabels(xlabs, rotation=45)
    else:
        ax.set_xticklabels(xlabs)
    ax.set_xlim(0, max(x)+1-0.2)
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
    ax.set_ylabel(metric)

    plt.tight_layout()
    save_path = join(dirname(data_file), save_name)
    plt.savefig(save_path)
    if show_plot:
        plt.show()

if __name__ == '__main__':
    input_file = join('C:\\Users\\zwaks\\Desktop\\apdm-june2017\\small_search_space', 'Summary.csv')

    input_file = join('C:\\Users\\zwaks\\Desktop\\apdm-june2017\\param_fast3_1000evals',
                'gait_measures_all.csv')

    input_file = join('C:\\Users\\zwaks\\Desktop\\apdm-june2017\\param_fast3_1000evals',
                'Summary_search_fast3_alg_tpe_evals_1000_folds_5.csv')


    rotate = True
    metric = 'MAPE'  # 'MAPE' or 'RMSE'
    show_plot = True
    save_name = 'alg_performance.png'
    # create_alg_performance_plot(input_file, metric, save_name, rotate, show_plot, analysis_type='all')
    create_alg_performance_plot(input_file, metric, save_name, rotate, show_plot)
