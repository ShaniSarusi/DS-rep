from os.path import join, sep
import matplotlib.pyplot as plt
import pandas as pd
import Gait.Resources.config as c
import pickle
import numpy as np


def create_regression_performance_plot(data_file, save_name='alg_performance.png', show_plot=False,
                                       set_y_lim=True, y_min=0, y_max=30, inp2=None):

    # Set plot values
    vals = []
    means = []
    stds = []
    data = pd.read_csv(data_file)
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

    sample_ids = data['SampleId'].as_matrix()
    if inp2 is not None:
        with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
            sample = pickle.load(fp)
    for i in range(len(algs)):
        # if 'two_sta' in algs[i]:
        if 'one_sta' in algs[i]:
            continue
        if inp2 is not None:
            true_vals = sample.loc[np.intersect1d(data['SampleId'], sample_ids)]['CadenceApdmMean'].as_matrix()
            alg_vals = data.loc[data['SampleId'].isin(sample_ids)][algs[i].replace('sc_', 'cadence_apdm_')].as_matrix()
        else:
            true_vals = data.loc[data['SampleId'].isin(sample_ids)][true_label].as_matrix()
            alg_vals = data.loc[data['SampleId'].isin(sample_ids)][algs[i]].as_matrix()

        vals_i = 100 * (alg_vals - true_vals) / true_vals
        mean_val = vals_i.mean()
        std_val = vals_i.std()
        means.append(mean_val)
        stds.append(std_val)
        vals.append(vals_i)

    # Plotting


    # Set legend
    algs = [alg.replace('sc_', '') for alg in algs]
    algs = ['Left side only' if alg == 'lhs' else alg for alg in algs]
    algs = ['Right side only' if alg == 'rhs' else alg for alg in algs]
    algs = ['Fusion - Sum raw signal' if 'sum' in alg else alg for alg in algs]
    algs = ['Fusion - Diff raw signal' if 'diff' in alg else alg for alg in algs]
    algs = ['Fusion - Intersection of steps' if 'intersect' in alg else alg for alg in algs]
    algs = ['Fusion - Union two stages' if 'two_sta' in alg else alg for alg in algs]
    algs = ['Fusion - Union one stage' if 'one_sta' in alg else alg for alg in algs]
    legends = algs

    groups = means
    errors = stds
    colors = ['r', 'm', 'b', 'c', 'y', 'g', 'k']

    fig, ax = plt.subplots()
    x = [1, 1.5, 2.5, 3, 4, 4.5]
    box = plt.boxplot([vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]], 0, '', positions=x,
                labels=['Left', 'Right', 'Sum', 'Diff', 'Int', 'Union'], widths=0.4, whis=[5, 95], patch_artist=True)
    colors = ['cyan', 'cyan', 'lightgreen', 'lightgreen', 'pink', 'pink']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.setp(box['medians'], color='k')
    plt.yticks(fontsize=10)
    if inp2 is not None:
        ylab = 'Cadence\n(percent difference)'
    else:
        ylab = 'Step count\n(percent error)'
    plt.ylabel(ylab, fontsize=11)
    plt.xticks(fontsize=9)
    plt.tight_layout()
    ax = fig.gca()
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    fig = plt.gcf()
    fig.set_size_inches(4, 3)
    fig.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()


if __name__ == '__main__':
    dirpath = join('C:', sep, 'Users', 'zwaks', 'Desktop', 'GaitPaper')
    input_file = join(dirpath,'aa_param3small_10k_big_sc_1008_v1', 'gait_measures.csv')
    input_file = join(dirpath, 'aa_param3small_5000_195_sc_1008_v1', 'gait_measures.csv')
    show_plot = True
    save_name = join(dirpath, 'SC_all.png')
    inp2 = join(c.pickle_path, 'metadata_sample')

    #create_regression_performance_plot(input_file, save_name, show_plot, y_min=-30, y_max=30)
    create_regression_performance_plot(input_file, save_name, show_plot, y_min=-30, y_max=30, inp2=inp2)

