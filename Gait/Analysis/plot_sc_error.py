from os.path import join, sep
import matplotlib.pyplot as plt
import pandas as pd
import Gait.Resources.config as c
import pickle
import numpy as np


def plot_sc_error(data_file, save_name='sc_err.png', hide_splines=False, sc_label='apdm', whisk='5_95'):

    # Set plot values
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
        sample = pickle.load(fp)
    vals = []
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

    sample_ids = data['SampleId'].as_matrix()
    sample_ids = sample_ids[sample_ids != 66]  # Sample 66 is problematic, perhaps because of the apdm values
    for i in range(len(algs)):
        if 'two_sta' in algs[i]:
            continue
        if sc_label == 'apdm':
            true_vals = sample.loc[np.intersect1d(data['SampleId'], sample_ids)]['CadenceApdmMean'].as_matrix()
            alg_vals = data.loc[data['SampleId'].isin(sample_ids)][algs[i].replace('sc_', 'cadence_apdm_')].as_matrix()
        else:  # sc_manual
            true_vals = data.loc[data['SampleId'].isin(sample_ids)]['sc_manual'].as_matrix()
            alg_vals = data.loc[data['SampleId'].isin(sample_ids)][algs[i]].as_matrix()
        vals.append(100 * (alg_vals - true_vals) / true_vals)

    # Plotting
    fig, ax = plt.subplots()
    x = [1, 1.5, 2.5, 3, 4, 4.5]
    if whisk == '5_95':
        box = plt.boxplot([vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]], 0, '', positions=x,
                labels=['L', 'R', 'Sum', 'Diff', 'Int', 'Union'], widths=0.4, whis=[5, 95], patch_artist=True)
    else:
        box = plt.boxplot([vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]], 0, '+', positions=x,
                          labels=['Left', 'Right', 'Sum', 'Diff', 'Int', 'Union'], widths=0.4, patch_artist=True)
    colors = ['cyan', 'cyan', 'lightgreen', 'lightgreen', 'magenta', 'magenta']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.setp(box['medians'], color='k')
    # x-axis stuff
    plt.xticks(fontsize=9.5)
    plt.xlabel('No fusion      Low-level        High-level\n                   fusion             fusion')
    ax.xaxis.set_label_coords(0.52, -0.2)
    # y-axis stuff
    plt.yticks(np.arange(-15, 40, 5), fontsize=10)
    plt.ylim(-15,20)
    plt.ylabel('Step count error (%)', fontsize=10)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    # Hide the right and top spines
    if hide_splines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    fig = plt.gcf()
    fig.set_size_inches(3.75, 2.5)
    fig.tight_layout()
    plt.savefig(save_name, dpi=600)
    plt.show()

    # P-Values
    # from scipy.stats import ks_2samp
    # rhs = np.abs(vals[1])
    # sum = np.abs(vals[2])
    # uni = np.abs(vals[5])


if __name__ == '__main__':
    dirpath = join('C:', sep, 'Users', 'zwaks', 'Desktop', 'GaitPaper')
    input_file = join(dirpath, 'a_cad10k_param4small_oct22_final', 'gait_measures.csv')
    show_plot = True
    save_name = join(dirpath, 'sc_err.png')
    plot_sc_error(input_file, save_name)

