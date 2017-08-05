import pickle
from os.path import join

import hyperopt as hp
import numpy as np
import pandas as pd
from hyperopt import fmin, Trials, tpe, space_eval

import Gait.Resources.config as c
from Gait.ParameterOptimization.evaluate_test_set import evaluate_on_test_set
from Gait.ParameterOptimization.objective_functions import all_algorithms
from Gait.ParameterOptimization.sum_results import sum_results
from Gait.ParameterOptimization.alg_performance_plot import create_alg_performance_plot
from Gait.ParameterOptimization.compare_to_apdm import compare_to_apdm
from Utils.Preprocessing.other_utils import split_data

if c.search_space == 'fast':
    import Gait.Resources.param_search_space_fast as param_search_space
elif c.search_space == 'small':
    import Gait.Resources.param_search_space_small as param_search_space
elif c.search_space == 'full':
    import Gait.Resources.param_search_space_full as param_search_space
if c.search_space == 'fast2':
    import Gait.Resources.param_search_space_fast2 as param_search_space
if c.search_space == 'fast3':
    import Gait.Resources.param_search_space_fast3 as param_search_space


##########################################################################################################
# Running parameters
space_all = list()
space_all.append(param_search_space.space_single_side_lhs)
space_all.append(param_search_space.space_overlap)
space_all.append(param_search_space.space_combined)
space_all.append(param_search_space.space_single_side_rhs)

objective_function_all = list()
objective_function_all.append('step_detection_single_side_lhs')
objective_function_all.append('step_detection_two_sides_overlap')
objective_function_all.append('step_detection_two_sides_combined_signal')
objective_function_all.append('step_detection_single_side_rhs')

n_folds = c.n_folds
max_evals = c.max_evals
alg = c.alg
metric = c.metric_to_optimize
save_dir = join(c.results_path, 'param_opt')
do_verbose = c.do_verbose

##########################################################################################################
# Set data splits
path_sample = join(c.pickle_path, 'metadata_sample')
with open(path_sample, 'rb') as fp:
    sample = pickle.load(fp)

ids = sample[sample['StepCount'].notnull()].index.tolist()  # use only samples with step count
filt = dict()
filt['notnull'] = sample[sample['StepCount'].notnull()].index.tolist()

task_ids = []
if c.data_type == 'split':
    walk_tasks = [1, 2, 3, 4, 5, 6, 7, 10]
    for k in walk_tasks:
        task_i = np.intersect1d(filt['notnull'], sample[sample['TaskId'] == k]['SampleId'].as_matrix())
        task_ids.append(task_i)
elif c.data_type == 'all':
    walk_tasks = ['all']
    task_i = filt['notnull']
    task_ids.append(task_i)
elif c.data_type == 'both':
    walk_tasks = [1, 2, 3, 4, 5, 6, 7, 10, 'all']
    for k in walk_tasks:
        if k == 'all':
            task_i = filt['notnull']
        else:
            task_i = np.intersect1d(filt['notnull'], sample[sample['TaskId'] == k]['SampleId'].as_matrix())
        task_ids.append(task_i)

train_all = []
test_all = []
for k in range(len(task_ids)):
    train_i, test_i = split_data(task_ids[k], n_folds=n_folds)
    train_all.append(train_i)
    test_all.append(test_i)


##########################################################################################################
# The parameter optimization code
df_gait_measures_by_alg = []
if c.data_type == 'both':
    df_gait_measures_by_alg_split = []
    df_gait_measures_by_alg_all = []
else:
    df_gait_measures_by_alg = []
for i in range(len(objective_function_all)):
    objective_function = objective_function_all[i]
    space = space_all[i]

    objective = all_algorithms[objective_function]
    if alg == 'tpe':
        algorithm = tpe.suggest
    elif alg == 'random':
        algorithm = hp.rand.suggest
    else:
        algorithm = tpe.suggest

    df_gait_measures_by_task = []
    for j in range(len(walk_tasks)):
        best = []
        root_mean_squared_error = []
        mape = []
        train = train_all[j]
        test = test_all[j]
        results = []
        for k in range(n_folds):
            print('************************************************************************')
            print('\rOptimizing Walk Task ' + str(walk_tasks[j]) + ': algorithm- ' + objective_function + '.   Using '
                  + alg + " search.   Running fold " + str(k + 1) + ' of ' + str(n_folds) + '. Max evals: ' +
                  str(c.max_evals))
            print('************************************************************************')

            # Optimize parameters
            space['sample_ids'] = train[k]
            space['metric'] = metric
            space['verbose'] = do_verbose
            trials = Trials()
            res = fmin(objective, space, algo=algorithm, max_evals=max_evals, trials=trials)
            results.append(res)

            # show progress
            with open(join(save_dir, objective_function + '_walk_task' + str(walk_tasks[j]) + '_fold_' + str(k+1)),
                      'wb') as fp:
                results2 = [results, train_all, test_all]
                pickle.dump(results2, fp)

        ##########################################################################################################
        # Evaluate results on cross validation test sets
        print('*****************************************************************************************')
        print('Finished running folds. Below are the best params and test set RMSE from each fold')
        print('*****************************************************************************************')
        for k in range(n_folds):
            print('Fold ' + str(k+1))
            space['verbose'] = True
            root_mean_squared_error_k, mape_k, best_params_k = evaluate_on_test_set(space, results[k], test[k],
                                                                                    objective, k, n_folds)
            root_mean_squared_error.append(root_mean_squared_error_k)
            mape.append(mape_k)
            best.append(best_params_k)

        ##########################################################################################################
        # Save gait specific metrics
        df_list = []
        for k in range(n_folds):
            params = space_eval(space, results[k])
            params['sample_ids'] = test[k]
            params['metric'] = 'get_res'
            params['verbose'] = False
            df = objective(params)
            df_list.append(df)
        df_for_task_j = pd.concat(df_list)
        df_gait_measures_by_task.append(df_for_task_j)

        ##########################################################################################################
        # Save results
        results = dict()
        results['best'] = best
        results['rmse'] = root_mean_squared_error
        results['mape'] = mape
        results['train'] = train
        results['test'] = test
        r = pd.DataFrame(results)
        r.to_csv(join(save_dir, objective_function + '_walk_task' + str(walk_tasks[j]) + '_all.csv'))

        with open(join(save_dir, objective_function + '_walk_task' + str(walk_tasks[j]) + '_all'), 'wb') as fp:
            to_save = [results, train_all, test_all]
            pickle.dump(to_save, fp)

        ##########################################################################################################
        # Print results
        print('Best params are:')
        print(best)
        print('RMSE results for all folds are:')
        print(root_mean_squared_error)

    if c.data_type == 'both':
        df_for_alg_i_split = pd.concat(df_gait_measures_by_task[0:-1])
        df_for_alg_i_all = pd.concat([df_gait_measures_by_task[-1]])
        df_gait_measures_by_alg_split.append(df_for_alg_i_split)
        df_gait_measures_by_alg_all.append(df_for_alg_i_all)
    else:
        df_for_alg_i = pd.concat(df_gait_measures_by_task)
        df_gait_measures_by_alg.append(df_for_alg_i)

##########################################################################################################
# Summarize and save all optimization results
print('***Summarizing and saving results***')
file_name = sum_results(save_dir, return_file_path=True)

# Create performance metric plots
data_file = join(save_dir, file_name)
metric = 'MAPE'  # 'MAPE' or 'RMSE'
create_alg_performance_plot(data_file, metric, save_name='alg_performance.png', rotate=True, show_plot=False)


def _gait_measure_analysis(df, dir, p_save_name, p_algs, p_metrics, prefix=""):
    res_gait = df[0]
    for i in range(1, len(df)):
        right = df[i]
        cols_left = res_gait.columns.tolist()
        cols_right = right.columns.tolist()
        cols_shared = list(set(cols_left).intersection(cols_right))
        right = right.drop(cols_shared, axis=1)
        res_gait = res_gait.join(right, how='outer')
    res_gait = res_gait.sort_index()
    gait_measure_path = join(dir, p_save_name)
    res_gait.to_csv(gait_measure_path)

    # Plot gait metric comparisons to APDM
    compare_to_apdm(gait_measure_path, p_algs, p_metrics, name_prefix=prefix)

# Summarize and save gait measure results
algs = ['lhs', 'rhs', 'overlap', 'combined']
metrics = ['cadence', 'step_time_asymmetry']
if c.data_type == 'both':
    save_name = 'gait_measures_split.csv'
    _gait_measure_analysis(df_gait_measures_by_alg_split, save_dir, save_name, algs, metrics, prefix='split')
    save_name = 'gait_measures_all.csv'
    _gait_measure_analysis(df_gait_measures_by_alg_all, save_dir, save_name, algs, metrics, prefix='all')
else:
    save_name = 'gait_measures.csv'
    _gait_measure_analysis(df_gait_measures_by_alg, save_dir, save_name, algs, metrics)
