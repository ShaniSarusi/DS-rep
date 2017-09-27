from os.path import join, dirname, sep
import Gait.Resources.config as c
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import pickle
from Gait.Pipeline.StepDetection import StepDetection


def compare_to_apdm(alg_data_file, algs, apdm_metrics, name_prefix="", show_plot=False):
    # Read data
    alg_res = pd.read_csv(alg_data_file, index_col='SampleId')
    idx = alg_res.index.tolist()
    with open(join(c.pickle_path, 'apdm_measures'), 'rb') as fp:
        apdm_measures_tmp = pickle.load(fp)
        apdm_measures = apdm_measures_tmp.iloc[[i for i in idx]]

    # Loop of plots for different metrics
    for metric in apdm_metrics:
        # Gather data
        if 'time_var' in metric:
            apdm_vals = (apdm_measures[metric + '_lhs'] + apdm_measures[metric + '_rhs']) / 2.0
        else:
            apdm_vals = apdm_measures[metric]
        idx_keep = pd.notnull(apdm_vals)
        apdm_vals = apdm_vals[idx_keep]
        metric_for_algs = metric
        if metric == 'apdm_toe_off_asymmetry_median':
            metric_for_algs = 'step_time_asymmetry_median'
        if 'time_var' in metric:
            alg_vals = []
            for j in range(len(algs)):
                side1 = alg_res[metric_for_algs + '_side1_' + algs[j]][idx_keep]
                side2 = alg_res[metric_for_algs + '_side2_' + algs[j]][idx_keep]
                avg = (side1 + side2)/2.0
                alg_vals.append(avg)
        else:
            alg_vals = [alg_res[metric_for_algs + '_' + algs[j]][idx_keep] for j in range(len(algs))]

        # Scatter plots
        if len(algs) > 6:
            a1 = 3
            b1 = 3
        elif len(algs) > 4:
            a1 = 3
            b1 = 2
        else:
            a1 = 2
            b1 = 2
        f, axarr = plt.subplots(a1, b1)
        corr = []
        for j in range(len(algs)):
            a = int(j/a1)
            b = j % b1

            idx_keep_j = pd.notnull(alg_vals[j])
            alg_vals_j = alg_vals[j][idx_keep_j]
            apdm_vals_j = apdm_vals[idx_keep_j]
            corr_j = pearsonr(apdm_vals_j, alg_vals_j)[0]
            corr.append(corr_j)

            axarr[a, b].scatter(apdm_vals_j, alg_vals_j)
            axarr[a, b].set_xlabel('APDM', fontsize=12)
            # Set y label
            y_label = 'alg_name'
            if 'lhs' in algs[j]: y_label='Lhs'
            if 'rhs' in algs[j]: y_label = 'Rhs'
            if 'intersect' in algs[j]: y_label = 'Intersect'
            if 'two_sta' in algs[j]: y_label = 'Union 2 stages'
            if 'one_sta' in algs[j]: y_label = 'Union 1 stage'
            if 'sum' in algs[j]: y_label = 'Sum'
            if 'diff' in algs[j]: y_label = 'Diff'
            axarr[a, b].set_ylabel(y_label, fontsize=12)

            val_max = max(max(alg_vals_j), max(apdm_vals_j))
            val_min = min(min(alg_vals_j), min(apdm_vals_j))
            val_range = val_max - val_min
            ax_min = val_min - 0.05 * val_range
            ax_max = val_max + 0.05 * val_range
            ax_range = ax_max - ax_min
            axarr[a, b].set_xlim(ax_min, ax_max)
            axarr[a, b].set_ylim(ax_min, ax_max)
            axarr[a, b].text(ax_min + 0.05 * ax_range, ax_min + 0.8 * ax_range, 'R= ' + str(round(corr_j, 3)))
        #plt.tight_layout()
        plt.suptitle(name_prefix + metric + ' comparison')
        plt.subplots_adjust(wspace=.4)

        # Save and show
        save_name = name_prefix + metric + '_comparison.png'
        save_path = join(dirname(alg_data_file), save_name)
        plt.savefig(save_path)
        if show_plot:
            plt.show()

if __name__ == '__main__':
    save_path = join('C:', sep, 'Users', 'zwaks', 'Desktop', 'GaitPaper','aa_param1_10k_sc_story')
    alg_data_path = join(save_path, 'gait_measures.csv')

    algorithms = ['lhs', 'rhs', 'fusion_high_level_intersect', 'fusion_high_level_union_two_stages',
                  'fusion_high_level_union_one_stage', 'fusion_low_level_sum', 'fusion_low_level_diff']
    metrics = ['cadence', 'stride_time_var', 'step_time_asymmetry_median', 'apdm_toe_off_asymmetry_median']

    compare_to_apdm(alg_data_path, algorithms, metrics, show_plot=True)
