from os.path import join, sep
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from Gait.Resources.gait_utils import initialize_offset_plots_data


def plot_gait_offset(input_file, save_name, max_dist_between_apdm_to_wrist_alg=0.5, hide_splines=False):
    heel_all, heel_lhs, heel_rhs, toe_all, toe_lhs, toe_rhs = \
        initialize_offset_plots_data(input_file, max_dist_between_apdm_to_wrist_alg)

    # alg to plot
    alg_name = 'idx_' + 'fusion_high_level_union_one_stage'
    vals = list()
    vals.append([item/1000.0 for sublist in heel_lhs[alg_name].tolist() for item in sublist])
    vals.append([item/1000.0 for sublist in heel_rhs[alg_name].tolist() for item in sublist])
    vals.append([item/1000.0 for sublist in toe_lhs[alg_name].tolist() for item in sublist])
    vals.append([item/1000.0 for sublist in toe_rhs[alg_name].tolist() for item in sublist])

    fig, ax = plt.subplots()
    box = plt.boxplot(vals, 0, '', positions=[1, 1.5, 2.5, 3], widths=0.4, whis=[5, 95], patch_artist=True)
    plt.xticks([1.25, 2.75], ['Heel strike', 'Toe-off'], fontsize=10)
    plt.yticks(np.arange(-0.1, 0.2, 0.05), fontsize=10)
    plt.ylabel('\u0394t after gait event (s)', fontsize=11)
    plt.xticks(fontsize=10)
    ax.xaxis.set_label_coords(0.46, -0.17)
    ax = fig.gca()
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    if hide_splines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Set colors and legend
    # colors = ['cyan', 'lightgreen'] * 2
    colors = ['#1f77b4', '#ff7f0e'] * 2
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.setp(box['medians'], color='k')
    first = mpatches.Patch(facecolor=colors[0], label='Left leg', edgecolor='black')
    second = mpatches.Patch(facecolor=colors[1], label='Right leg', edgecolor='black')
    plt.legend(handles=[first, second], fontsize=9.5)

    fig = plt.gcf()
    fig.set_size_inches(3, 2.5)
    fig.tight_layout()
    plt.savefig(save_name, dpi=600)
    plt.show()

if __name__ == '__main__':
    dir_path = join('C:', sep, 'Users', 'zwaks', 'Desktop', 'GaitPaper')
    input_path = join(dir_path, 'a_cad10k_param4small_oct22_final', 'gait_measures.csv')
    save_path = join(dir_path, 'gait_phase_just_union2test.png')
    plot_gait_offset(input_path, save_path, hide_splines=True)
