import pickle
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import Gait.Resources.config as c
from Gait.Pipeline.StepDetection import StepDetection


def compare_to_apdm(data_file, algs, apdm_metrics, show_plot=False):
    # Parameters
    alg_res = pd.read_csv(data_file, index_col='SampleId')

    # apdm_metrics = sd.apdm_measures.columns.tolist()
    # apdm_metrics = ['cadence', 'step_time_asymmetry', 'stride_time_var_lhs', 'stride_time_var_rhs', 'step_time_var_lhs', 'step_time_var_lhs']

    # Read APDM measures
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
        sample = pickle.load(fp)
    with open(join(c.pickle_path, 'acc'), 'rb') as fp:
        acc = pickle.load(fp)
    with open(join(c.pickle_path, 'apdm_measures'), 'rb') as fp:
        apdm_measures = pickle.load(fp)
    with open(join(c.pickle_path, 'apdm_events'), 'rb') as fp:
        apdm_events = pickle.load(fp)

    # Use only samples with step count
    id_nums = sample[sample['StepCount'].notnull()].index.tolist()

    # Preprocessing
    sd = StepDetection(acc, sample, apdm_measures, apdm_events)
    sd.select_specific_samples(alg_res.index)










    with open(join(c.pickle_path, 'sc_alg'), 'rb') as fp:
        sd = pickle.load(fp)
        # alg_res = sd.res
        # TODO may need to set indices to remove nulls...
        # # set indices
        # with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
        #     sample = pickle.load(fp)
        apdm_measures = sd.apdm_measures

    ####################################################################################################################
    # Loop of plots for different metrics
    for metric in apdm_metrics:
        # Gather data
        apdm_vals = apdm_measures[metric]
        idx_keep = pd.notnull(apdm_vals)
        apdm_vals = apdm_vals[idx_keep]
        alg_vals = [alg_res[metric + '_' + algs[j]][idx_keep] for j in range(len(algs))]
        corr = [pearsonr(apdm_vals, alg_vals[j])[0] for j in range(len(algs))]

        # Setup plot format
        f, axarr = plt.subplots(2, 2)
        for j in range(len(algs)):
            a = int(j/2)
            b = j % 2
            axarr[a, b].scatter(alg_vals[j], apdm_vals)
            axarr[a, b].set_xlabel('APDM', fontsize=12)
            axarr[a, b].set_ylabel(algs[j] + ' algorithm', fontsize=12)

            val_max = max(max(alg_vals[j]), max(apdm_vals))
            val_min = min(min(alg_vals[j]), min(apdm_vals))
            val_range = val_max - val_min
            ax_min = val_min - 0.05 * val_range
            ax_max = val_max + 0.05 * val_range
            ax_range = ax_max - ax_min
            axarr[a, b].set_xlim(ax_min, ax_max)
            axarr[a, b].set_ylim(ax_min, ax_max)
            axarr[a, b].text(ax_min + 0.05 * ax_range, ax_min + 0.9 * ax_range, 'R= ' + str(round(corr[j], 3)))
        plt.tight_layout()
        plt.suptitle(metric + ': APDM vs various algorithms')
        plt.show()

if __name__ == '__main__':
    alg_data_path = join(c.results_path, 'param_opt', 'gait_measures.csv')
    algorithms = ['lhs', 'rhs', 'overlap', 'combined']
    metrics = ['cadence', 'step_time_asymmetry']

    compare_to_apdm(alg_data_path, algorithms, metrics, show_plot=True)
