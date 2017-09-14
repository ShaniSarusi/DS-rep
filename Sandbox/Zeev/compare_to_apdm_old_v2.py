import pickle
from os.path import join, dirname

import Gait_old.Resources.config as c
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr


def compare_to_apdm(data_file, algs, apdm_metrics, show_plot=False):
    # Parameters
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
    for metric in apdm_metrics:
        # Gather data
        apdm_vals = apdm_measures[metric]
        idx_keep = pd.notnull(apdm_vals)
        apdm_vals = apdm_vals[idx_keep]
        alg_vals = [alg_res[metric + '_' + algs[j]][idx_keep] for j in range(len(algs))]
        corr = [pearsonr(apdm_vals, alg_vals[j])[0] for j in range(len(algs))]

        # Setup plot format
        fig = plt.figure(figsize=(10, 8))
        outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

        inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
        for j in range(len(algs)):
            ax = plt.Subplot(fig, inner[j])
            t = ax.text(0.05, 0.9, 'Pearson correlation ' + str(round(corr[j], 3)))
            t.set_ha('left')
            # b = ax.scatter(alg_vals[j], apdm_vals)

            ax.set_xlabel('APDM', fontsize=12)
            ax.set_ylabel(algs[j] + ' algorithm', fontsize=12)
            fig.add_subplot(ax)
            plt.tight_layout()


        # Boxplot
        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
        ax = plt.Subplot(fig, inner[0])
        vals = [apdm_vals] + alg_vals
        ax.boxplot(vals)
        ax.set_xticklabels(['apdm'] + algs, fontsize=14)
        ax.set_xlabel('Algorithm', fontsize=18)
        ax.set_ylabel('Values', fontsize=18)
        ax.set_title('Value distribution')
        fig.add_subplot(ax)


        # Save and show
        save_name = metric + '_comparison.png'
        save_path = join(dirname(data_file), save_name)
        if show_plot:
            plt.show()
        fig.savefig(save_path)

if __name__ == '__main__':
    alg_data_path = join(c.results_path, 'param_opt', 'gait_measures.csv')
    algorithms = ['lhs', 'rhs', 'overlap', 'combined']
    metrics = ['cadence', 'step_time_asymmetry']

    compare_to_apdm(alg_data_path, algorithms, metrics, show_plot=True)
