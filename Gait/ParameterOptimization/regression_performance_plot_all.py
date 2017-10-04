from os.path import join, dirname, sep
import matplotlib.pyplot as plt
import pandas as pd


def create_regression_performance_plot(data_file, save_name='alg_performance.png', show_plot=False,
                                       set_y_lim=True, y_min=0, y_max=30):

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
    for i in range(len(algs)):
        # if 'two_sta' in algs[i]:
        if 'one_sta' in algs[i]:
            continue
        alg_vals = data.loc[data['SampleId'].isin(sample_ids)][algs[i]].as_matrix()
        true_vals = data.loc[data['SampleId'].isin(sample_ids)][true_label].as_matrix()
        vals_i = 100 * (alg_vals - true_vals) / true_vals
        mean_val = vals_i.mean()
        std_val = vals_i.std()
        means.append(mean_val)
        stds.append(std_val)
        vals.append(vals_i)

    # Plotting
    fig, ax = plt.subplots()

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

    x = [1, 1.5, 2.5, 3, 4, 4.5]

    box = plt.boxplot([vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]], 0, '', positions=x,
                labels=['Left', 'Right', 'Sum', 'Diff', 'Inters.', 'Union'], widths=0.4, whis=[5, 95], patch_artist=True)
    colors = ['cyan', 'cyan', 'lightgreen', 'lightgreen', 'pink', 'pink']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.setp(box['medians'], color='k')
    plt.yticks(fontsize=10)
    plt.ylabel('Step count\n(percent error)', fontsize=11)
    plt.xticks(fontsize=8.5)
    plt.tight_layout()
    ax = fig.gca()
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    plt.show()

    x = [1, 1.4, 2.5, 2.9, 4, 4.4]
    plt.errorbar(x, means,
                 yerr=errors,
                 marker='o',
                 color='k',
                 ecolor='k',
                 markerfacecolor='g',
                 capsize=5,
                 linestyle='None')
    plt.xticks(x, ['Left', 'Right', 'Sum', 'Diff', 'Intersection', 'Union'], rotation='vertical', fontsize='16')
    plt.yticks(fontsize=14)
    plt.ylabel('Percent error', fontsize=18)
    plt.tight_layout()
    plt.show()

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
    ax.set_xticklabels(['All'], fontsize=12)
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
    ax.set_ylabel('Percent error', fontsize=18)
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
    input_file = join('C:', sep, 'Users', 'zwaks', 'Desktop', 'GaitPaper','aa_param1_500_sc_1002_v1', 'gait_measures.csv')

    show_plot = True
    save_name = 'pe_all_test.png'

    create_regression_performance_plot(input_file, save_name, show_plot, y_min=-30, y_max=30)

