from os.path import join

import numpy as np
import pandas as pd

from Utils.DataHandling.data_processing import string_to_int_list


def sum_results_for_plotting_parameters(input_file, save_dir):
    data = pd.read_csv(input_file)
    params = ['Signal', 'Smoothing', 'Smoothing(window/filter frequency)', 'Peaks', 'Peak_param1', 'Peak_param2',
           'Overlap_Merge_Window', 'Overlap_adjacent_peaks_window', 'Mva_combined_window', 'Sine-min Hz', 'Sine-max Hz',
           'Sine-factor', 'Z']
    res = pd.DataFrame(index=params)
    algs = data['Algorithm'].unique().tolist()
    split_types = ['all', 'split']

    rows_all = np.array(data[data['WalkingTask'] == 'all'].index, dtype=int)
    rows_split = np.array(data[data['WalkingTask'] != 'all'].index, dtype=int)
    for alg in algs:
        alg_rows = np.array(data[data['Algorithm'] == alg].index, dtype=int)
        for split_type in split_types:
            split_rows = None
            if split_type == 'all':
                split_rows = rows_all
            if split_type == 'split':
                split_rows = rows_split
            rows = np.intersect1d(alg_rows, split_rows)
            col_name = split_type + '_' + alg

            res_vals = []
            for data_col in res.index:
                cell_val = data[data_col].iloc[rows].tolist()
                res_vals.append(cell_val)
            res[col_name] = res_vals

    # Calculate means and stds
    res_mean = pd.DataFrame(index=res.index, columns=res.columns)
    res_std = pd.DataFrame(index=res.index, columns=res.columns)
    for i in res.index:
        for col in res.columns:
            vals = string_to_int_list(res[col].loc[i])
            try:
                res_mean[col].loc[i] = np.mean(vals)
                res_std[col].loc[i] = np.std(vals)
            except:
                'do nothing'

    # Save results
    res.to_csv(join(save_dir, 'Summary_params_for_plots.csv'))
    res_mean.to_csv(join(save_dir, 'Summary_params_for_plots_mean.csv'))
    res_std.to_csv(join(save_dir, 'Summary_params_for_plots_std.csv'))

if __name__ == '__main__':
    save_path = 'C:\\Users\\zwaks\\Desktop\\apdm-june2017\\param6_2000ev_tpe'
    input_file = join(save_path, 'Summary_search_param6_alg_tpe_evals_2000_folds_5.csv')
    sum_results_for_plotting_parameters(input_file, save_path)
