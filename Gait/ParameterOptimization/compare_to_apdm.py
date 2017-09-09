import pickle
from os.path import join, dirname, sep
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import Gait.Resources.config as c
from Gait.Pipeline.StepDetection import StepDetection


def _read_apdm_measures(idx):
    # Read APDM measures
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
        sample = pickle.load(fp)
    with open(join(c.pickle_path, 'acc'), 'rb') as fp:
        acc = pickle.load(fp)
    with open(join(c.pickle_path, 'apdm_measures'), 'rb') as fp:
        apdm_measures = pickle.load(fp)
    with open(join(c.pickle_path, 'apdm_events'), 'rb') as fp:
        apdm_events = pickle.load(fp)
    sd = StepDetection(acc, sample, apdm_measures, apdm_events)
    sd.select_specific_samples(idx)
    apdm_measures = sd.apdm_measures
    del sd
    return apdm_measures


def compare_to_apdm(alg_data_file, algs, apdm_metrics, name_prefix="", show_plot=False):
    # Read data
    alg_res = pd.read_csv(alg_data_file, index_col='SampleId')
    idx = alg_res.index.tolist()
    apdm_measures = _read_apdm_measures(idx)

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
        if metric == 'toe_off_asymmetry_median':
            metric_for_algs = 'step_time_asymmetry2_median'
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
        if len(algs) > 4:
            f, axarr = plt.subplots(3, 3)
        else:
            f, axarr = plt.subplots(2, 2)
        corr = []
        for j in range(len(algs)):
            idx_keep_j = pd.notnull(alg_vals[j])
            alg_vals_j = alg_vals[j][idx_keep_j]
            apdm_vals_j = apdm_vals[idx_keep_j]
            corr_j = pearsonr(apdm_vals_j, alg_vals_j)[0]
            corr.append(corr_j)

            a = int(j/2)
            b = j % 2
            axarr[a, b].scatter(apdm_vals_j, alg_vals_j)
            axarr[a, b].set_xlabel('APDM', fontsize=12)
            axarr[a, b].set_ylabel(algs[j] + ' algorithm', fontsize=12)
            val_max = max(max(alg_vals_j), max(apdm_vals_j))
            val_min = min(min(alg_vals_j), min(apdm_vals_j))
            val_range = val_max - val_min
            ax_min = val_min - 0.05 * val_range
            ax_max = val_max + 0.05 * val_range
            ax_range = ax_max - ax_min
            axarr[a, b].set_xlim(ax_min, ax_max)
            axarr[a, b].set_ylim(ax_min, ax_max)
            axarr[a, b].text(ax_min + 0.05 * ax_range, ax_min + 0.9 * ax_range, 'R= ' + str(round(corr_j, 3)))
        plt.tight_layout()
        plt.suptitle(name_prefix + metric + ' comparison')

        # Save and show
        save_name = name_prefix + metric + '_comparison.png'
        save_path = join(dirname(alg_data_file), save_name)
        plt.savefig(save_path)
        if show_plot:
            plt.show()

if __name__ == '__main__':
    alg_file_name = 'gait_measures.csv'
    alg_file_name = 'gait_measures_all.csv'
    alg_data_path = join(c.results_path, 'param_opt', alg_file_name)

    save_path = 'C:\\Users\\zwaks\\Desktop\\apdm-june2017\\param7_machine1_5percent'
    alg_data_path = join(save_path, 'gait_measures_all.csv')
    alg_data_path = join(save_path, 'gait_measures_all.csv')


    algorithms = ['lhs', 'rhs', 'overlap', 'combined']
    algorithms = ['lhs', 'rhs', 'overlap', 'overlap_strong', 'combined']
    metrics = ['cadence', 'step_time_asymmetry', 'step_time_var', 'stride_time_var']
    metrics = ['cadence', 'step_time_asymmetry', 'stride_time_var', 'step_time_asymmetry2_median']
    metrics = ['cadence', 'step_time_asymmetry', 'stride_time_var', 'step_time_asymmetry2_median', 'toe_off_asymmetry_median']

    compare_to_apdm(alg_data_path, algorithms, metrics, show_plot=True)
