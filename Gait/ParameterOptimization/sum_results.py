import pandas as pd
from Utils.DataHandling.reading_and_writing_files import read_all_files_in_directory
import re
import ast
from os.path import join
import pickle
import Gait.config as c

data_path = 'C:\\Users\\zwaks\\Desktop\\apdm-june2017\\small_search_space'
data_path = 'C:\\Users\\zwaks\\Documents\\Data\\APDM June 2017\\Results\\param_opt'


# Start code
with open(join(c.pickle_path, 'task_filters'), 'rb') as fp:
    task_filters = pickle.load(fp)
res = pd.DataFrame(columns=['Algorithm', 'WalkingTask', 'TaskNum', 'FoldNum', 'RMSE', 'MAPE', 'Signal', 'Smoothing',
                            'Smoothing(window/filter frequency)', 'Peaks', 'Peak_param1', 'Peak_param2',
                            'Overlap_Merge_Window', 'Overlap_adjacent_peaks_window', 'Mva_combined_window',
                            'Sine-min Hz', 'Sine-max Hz', 'Sine-factor'])
f = read_all_files_in_directory(dir_path=data_path, file_type='csv')
for i in range(len(f)):
    algo = re.search('(.*)_walk', f[i]).group(1)
    training_data = re.search('task(.*)_', f[i]).group(1)
    if training_data == 'all':
        task_name = 'all'
    else:
        task_name = task_filters[task_filters['TaskId'] == int(training_data)]['Task Name'].iloc[0]

    f_i = pd.read_csv(join(data_path, f[i]))
    for j in range(f_i.shape[0]):
        idx = res.shape[0]
        res.set_value(idx, 'Algorithm', algo)
        res.set_value(idx, 'WalkingTask', task_name)
        res.set_value(idx, 'TaskNum', training_data)
        res.set_value(idx, 'FoldNum', str(j+1))
        res.set_value(idx, 'RMSE', f_i['rmse'][j])
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
        if "overlap" in algo:
            res.set_value(idx, 'Overlap_Merge_Window', p['win_size_merge'])
            res.set_value(idx, 'Overlap_adjacent_peaks_window', p['win_size_remove_adjacent_peaks'])

        if "combined" in algo:
            res.set_value(idx, 'Mva_combined_window', p['mva_win_combined'])
            res.set_value(idx, 'Sine-min Hz', p['min_hz'])
            res.set_value(idx, 'Sine-max Hz', p['max_hz'])
            res.set_value(idx, 'Sine-factor', p['factor'])


# Save
res.to_csv(join(data_path, 'Summary.csv'), index=False)
