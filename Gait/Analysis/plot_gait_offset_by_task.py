import pickle
from os.path import join, sep
import matplotlib.pyplot as plt
import numpy as np
import Gait.Resources.config as c
import matplotlib.patches as mpatches
from Gait.Resources.gait_utils import initialize_offset_plots_data


def plot_gait_offset_by_task(input_file, save_name, max_dist_between_apdm_to_wrist_alg=0.5, hide_splines=False):
    heel_all, heel_lhs, heel_rhs, toe_all, toe_lhs, toe_rhs = \
        initialize_offset_plots_data(input_file, max_dist_between_apdm_to_wrist_alg)

    # Read task filters and metadata
    with open(join(c.pickle_path, 'task_filters'), 'rb') as fp:
        task_filters = pickle.load(fp)
        tasks = task_filters['Task Name'].tolist()
        tasks.remove('Tug')
        tasks.remove('Tug')
        a, b = tasks.index('Walk - Regular'), tasks.index('Walk - Fast')
        tasks[b], tasks[a] = tasks[a], tasks[b]
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
        sample = pickle.load(fp)
        sample['TaskName'] = sample['TaskName'].replace('Walk - Imagine you have a cane in the right hand',
                                                        'Asymmetry - Imagine you have a cane in the right hand')
        sample['TaskName'] = sample['TaskName'].replace('Walk - Without right shoe', 'Asymmetry - No right shoe')

    # Set plot values
    toe_l = []
    toe_r = []
    for i in range(len(tasks)):
        sample_ids = sample[sample['TaskName'] == tasks[i]]['SampleId'].as_matrix()
        x = toe_lhs['idx_fusion_high_level_union_one_stage'].iloc[sample_ids].as_matrix()
        x2 = [item/1000.0 for sublist in x for item in sublist]
        toe_l.append(x2)
        x = toe_rhs['idx_fusion_high_level_union_one_stage'].iloc[sample_ids].as_matrix()
        x2 = [item / 1000.0 for sublist in x for item in sublist]
        toe_r.append(x2)
    vals = []
    for i in range(len(tasks)):
        vals.append(toe_l[i])
        vals.append(toe_r[i])

    # PLOTTING ****************************************
    # set x axis locations
    x = []
    gap = 3
    inner_gap = 0.6
    num_algs = 2
    xtick_locs = []
    for i in range(len(tasks)):
        xtick_locs.append(gap * i + 0.5 * num_algs * inner_gap)
        for j in range(num_algs):
            x_tmp = i * gap + j * inner_gap
            x.append(x_tmp)

    # draw boxplot
    fig, ax = plt.subplots()
    box = plt.boxplot(vals, 0, '', positions=x, widths=0.4, whis=[5, 95], patch_artist=True)

    # Set x axis labels
    xlabs = ['Cane' if x == 'Asymmetry - Imagine you have a cane in the right hand' else x for x in tasks]
    xlabs = ['No shoe' if x == 'Asymmetry - No right shoe' else x for x in xlabs]
    xlabs = ['Hands on side' if x == 'Walk - Both hands side' else x for x in xlabs]
    xlabs = [x[6:] if 'Walk - ' in x else x for x in xlabs]
    xlabs = ['Right\nbag' if x == ' Right bag' else x for x in xlabs]
    xlabs = ['Hands\non side' if x == 'Hands on side' else x for x in xlabs]
    plt.xticks(xtick_locs, xlabs, fontsize=10)
    plt.xticks(fontsize=10)

    # Set y-axis
    plt.yticks(np.arange(-0.15, 0.2, 0.05), fontsize=10)
    plt.ylim(-0.15, 0.15)
    plt.ylabel('\u0394t after toe-off (s)', fontsize=11)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    # Set colors and legend
    colors = ['#1f77b4', '#ff7f0e'] * len(x)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.setp(box['medians'], color='k')
    # legend
    first = mpatches.Patch(facecolor=colors[0], label='Left leg toe off', edgecolor='black')
    second = mpatches.Patch(facecolor=colors[1], label='Right leg toe off', edgecolor='black')
    plt.legend(handles=[first, second], fontsize=9.5)

    if hide_splines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    fig = plt.gcf()
    fig.set_size_inches(6, 2.5)
    fig.tight_layout()
    plt.savefig(save_name, dpi=600)
    plt.show()

if __name__ == '__main__':
    dir_path = join('C:', sep, 'Users', 'zwaks', 'Desktop', 'GaitPaper')
    input_path = join(dir_path, 'a_cad10k_param4small_oct22_final', 'gait_measures.csv')
    save_path = join(dir_path, 'gait_phase_just_union2test_per_task.png')
    plot_gait_offset_by_task(input_path, save_path, hide_splines=True)
