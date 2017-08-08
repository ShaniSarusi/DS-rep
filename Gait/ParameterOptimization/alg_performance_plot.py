import pickle
from os.path import join, dirname
import matplotlib.pyplot as plt
import pandas as pd
import Gait.Resources.config as c


def create_alg_performance_plot(data_file, metric, save_name='alg_performance.png', rotate=True, show_plot=False,
                                set_y_lim=True, y_min=0, y_max=30):
    data = pd.read_csv(data_file)

    with open(join(c.pickle_path, 'task_filters'), 'rb') as fp:
        task_filters = pickle.load(fp)
    tasks = ['all'] + task_filters['Task Name'].tolist()
    tasks.remove('Tug')
    tasks.remove('Tug')
    a, b = tasks.index('Walk - Regular'), tasks.index('Walk - Fast')
    tasks[b], tasks[a] = tasks[a], tasks[b]

    xlabs = ['Cane' if x == 'Asymmetry - imagine you have a cane in the right hand' else x for x in tasks]
    xlabs = ['No shoe' if x == 'Asymmetry - no right shoe' else x for x in xlabs]
    xlabs = ['Hands on side' if x == 'Walk - both hands side' else x for x in xlabs]
    xlabs = [x[6:] if 'Walk - ' in x else x for x in xlabs]

    algs = data['Algorithm'].unique().tolist()
    algs.sort()

    means = []
    stds = []
    for i in range(len(algs)):
        means_i = []
        stds_i = []
        for j in range(len(tasks)):
            vals = data[(data['Algorithm'] == algs[i]) & (data['WalkingTask'] == tasks[j])][metric]
            means_i.append(vals.mean())
            stds_i.append(vals.std())
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
    input_file = join('C:\\Users\\zwaks\\Documents\\Data\\APDM June 2017\\Results\\param_opt',
                'Summary_search_fast4_test_alg_tpe_evals_2_folds_2.csv')
    rotate = True
    metric = 'MAPE'  # 'MAPE' or 'RMSE'
    show_plot = True
    save_name = 'alg_performance.png'
    create_alg_performance_plot(input_file, metric, save_name, rotate, show_plot)