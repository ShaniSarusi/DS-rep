import ast
import pickle
import re
from os.path import join
import pandas as pd
import Gait.Resources.config as c
from Utils.DataHandling.reading_and_writing_files import read_all_files_in_directory


def write_best_params_to_csv(save_dir, return_file_path=False):
    # Start code
    with open(join(c.pickle_path, 'task_filters'), 'rb') as fp:
        task_filters = pickle.load(fp)
    res = pd.DataFrame(columns=['Algorithm', 'WalkingTask', 'TaskNum', 'FoldNum', 'RMSE', 'MAPE', 'Mva_win',
                                'Peak_min_thr', 'Peak_min_dist', 'Intersect_win', 'Union_min_dist', 'Union_min_thr'])
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
            res.set_value(idx, 'Mva_win', p['mva_win'])
            res.set_value(idx, 'Peak_min_thr', p['peak_min_thr'])
            res.set_value(idx, 'Peak_min_dist', p['peak_min_dist'])
            if "intersect_win" in p:
                res.set_value(idx, 'Intersect_win', p['intersect_win'])
            if "union_min_dist" in p:
                res.set_value(idx, 'Union_min_dist', p['union_min_dist'])
            if "union_min_thresh" in p:
                res.set_value(idx, 'Union_min_thr', p['union_min_thresh'])
            if "mva_win_combined" in p:
                res.set_value(idx, 'Mva_win_combined', p['mva_win_combined'])

    # Save
    file_name = 'Summary_search_' + c.search_space + '_alg_' + c.opt_alg + '_evals_' + str(c.max_evals) + '_folds_' + \
                str(c.n_folds) + '.csv'

    file_path = join(save_dir, file_name)
    res.to_csv(file_path, index=False)

    if return_file_path:
        return file_path

if __name__ == '__main__':
    save_path = join(c.results_path, 'param_opt')
    write_best_params_to_csv(save_path)
