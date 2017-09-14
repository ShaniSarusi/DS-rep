import ast
import pickle
import re
from os.path import join

import pandas as pd

from Sandbox.Zeev import Gait_old as c
from Utils.DataHandling.reading_and_writing_files import read_all_files_in_directory


def sum_results(save_dir, return_file_path=False):
    # Start code
    with open(join(c.pickle_path, 'task_filters'), 'rb') as fp:
        task_filters = pickle.load(fp)
    res = pd.DataFrame(columns=['Algorithm', 'WalkingTask', 'TaskNum', 'FoldNum', 'RMSE', 'MAPE', 'Signal', 'Smoothing',
                                'Smoothing(window/filter frequency)', 'Peaks', 'Peak_param1', 'Peak_param2',
                                'Overlap_Merge_Window', 'Overlap_adjacent_peaks_window', 'Mva_combined_window',
                                'Sine-min Hz', 'Sine-max Hz', 'Sine-factor', 'Z'])
    f = read_all_files_in_directory(dir_path=save_dir, file_type='csv')
    for i in range(len(f)):
        if '_walk' not in f[i]:
            continue
        if 'task' not in f[i]:
            continue
        alg = re.search('(.*)_walk', f[i]).group(1)
        training_data = re.search('task(.*)_', f[i]).group(1)
        if training_data == 'all':
            task_name = 'all'
        else:
            task_name = task_filters[task_filters['TaskId'] == int(training_data)]['Task Name'].iloc[0]

        f_i = pd.read_csv(join(save_dir, f[i]))
        for j in range(f_i.shape[0]):
            idx = res.shape[0]
            res.set_value(idx, 'Algorithm', alg)
            res.set_value(idx, 'WalkingTask', task_name)
            res.set_value(idx, 'TaskNum', training_data)
            res.set_value(idx, 'FoldNum', str(j+1))
            res.set_value(idx, 'RMSE', f_i['rmse'][j])
            if 'mape' in f_i.columns:
                res.set_value(idx, 'MAPE', f_i['mape'][j])

            # params
            p = ast.literal_eval(f_i['best'][j])
            # Signal
            if p['signal_to_use'] == 'norm':
                res.set_value(idx, 'Signal', p['signal_to_use'])
            elif not p['do_windows_if_vertical']:
                res.set_value(idx, 'Signal', 'Vertical-no windows')
            else:
                res.set_value(idx, 'Signal', 'Vertical: window- ' + str(p['vert_win']))
            # Smoothing
            if p['smoothing'] == 'mva':
                res.set_value(idx, 'Smoothing', 'mva')
                res.set_value(idx, 'Smoothing(window/filter frequency)', p['mva_win'])
            if p['smoothing'] == 'butter':
                res.set_value(idx, 'Smoothing', 'Butterfilter')
                res.set_value(idx, 'Smoothing(window/filter frequency)', p['butter_freq'])
            # Peaks
            # if p['peak_type'] == 'scipy':
            #     res.set_value(idx, 'Peaks', 'Scipy (' + str(p['p1_sc']) + ', ' + str(p['p2_sc']) + ')')
            # if p['peak_type'] == 'peak_utils':
            #     res.set_value(idx, 'Peaks', 'PeakUtils (' + str(p['p1_pu']) + ', ' + str(p['p2_pu']) + ')')
            res.set_value(idx, 'Peaks', p['peak_type'])
            if p['peak_type'] == 'scipy':
                res.set_value(idx, 'Peak_param1', p['p1_sc'])
                res.set_value(idx, 'Peak_param2', p['p2_sc'])
            elif p['peak_type'] == 'peak_utils':
                res.set_value(idx, 'Peak_param1', p['p1_pu'])
                res.set_value(idx, 'Peak_param2', p['p2_pu'])

            # Overlap algorithm
            if "overlap" in alg:
                res.set_value(idx, 'Overlap_Merge_Window', p['win_size_merge'])
                res.set_value(idx, 'Overlap_adjacent_peaks_window', p['win_size_remove_adjacent_peaks'])

            if "overlap_strong" in alg:
                res.set_value(idx, 'Z', p['z'])

            if "combined" in alg:
                res.set_value(idx, 'Mva_combined_window', p['mva_win_combined'])
                res.set_value(idx, 'Sine-min Hz', p['min_hz'])
                res.set_value(idx, 'Sine-max Hz', p['max_hz'])
                res.set_value(idx, 'Sine-factor', p['factor'])

    # Save
    file_name = 'Summary_search_' + c.search_space + '_alg_' + c.opt_alg + '_evals_' + str(c.max_evals) + '_folds_' + \
                str(c.n_folds) + '.csv'

    file_path = join(save_dir, file_name)
    res.to_csv(file_path, index=False)

    if return_file_path:
        return file_path

if __name__ == '__main__':
    save_path = join(c.results_path, 'param_opt')
    sum_results(save_path)
