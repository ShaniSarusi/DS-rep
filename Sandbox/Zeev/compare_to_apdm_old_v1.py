import pickle
from os.path import join, dirname

import Gait_old.Resources.config as c
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from Sandbox.Zeev.Gait_old.Pipeline.gait_utils import set_filters


def compare_to_apdm(data_file, algs, apdm_metrics, show_plot=False):
    # Parameters
    f = set_filters(exp=2)
    alg_res = pd.read_csv(data_file, index_col='SampleId')

    # apdm_metrics = sd.apdm_measures.columns.tolist()
    # apdm_metrics = ['cadence', 'step_time_asymmetry', 'stride_time_var_lhs', 'stride_time_var_rhs', 'step_time_var_lhs', 'step_time_var_lhs']

    # Read APDM measures
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
    # for i in [0]:
    for metric in apdm_metrics:
        apdm_vals = apdm_measures[metric]
        idx_keep = pd.notnull(apdm_vals)
        apdm_vals = apdm_vals[idx_keep]
        alg_vals = [alg_res[metric + '_' + algs[j]][idx_keep] for j in range(len(algs))]

        # Correlation - plot 1
        corr = [pearsonr(apdm_vals, alg_vals[j])[0] for j in range(len(algs))]

        # Plot
        plt.suptitle(metric, fontsize=12)
        plt.subplot(121)
        y_pos = np.arange(len(corr))
        plt.bar(y_pos, corr, align='center', alpha=0.5)
        plt.xticks(y_pos, algs)
        plt.xlabel('Algorithm')
        plt.ylabel('Pearson correlation (R)')
        plt.title('Algorithm correlation to APDM')

        plt.subplot(122)
        vals = [apdm_vals] + alg_vals
        plt.boxplot(vals)
        y_pos = np.arange(len(vals)) + 1
        plt.xticks(y_pos, ['apdm'] + algs)
        plt.xlabel('Algorithm')
        plt.ylabel('Values')
        plt.title('Value distribution')

        # Save and show
        save_name = metric + '_comparison.png'
        save_path = join(dirname(data_file), save_name)
        plt.savefig(save_path)
        if show_plot: plt.show()

if __name__ == '__main__':
    alg_data_path = join(c.results_path, 'param_opt', 'gait_measures.csv')
    algorithms = ['lhs', 'rhs', 'overlap', 'combined']
    metrics = ['cadence', 'step_time_asymmetry']

    compare_to_apdm(alg_data_path, algorithms, metrics, show_plot=True)
