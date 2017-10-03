import pickle
from multiprocessing import Pool
from os.path import join
import hyperopt as hp
import numpy as np
import pandas as pd
from hyperopt import fmin, Trials, tpe, space_eval
import Gait.Resources.config as c
from Gait.Resources.gait_utils import evaluate_on_test_set, create_gait_measure_csvs, create_sd_class_for_obj_functions, par_fmin
import Gait.ParameterOptimization.objective_functions as o
from Gait.ParameterOptimization.compare_to_apdm import compare_to_apdm
from Gait.ParameterOptimization.regression_performance_plot import create_regression_performance_plot
from Gait.ParameterOptimization.write_best_params_to_csv import write_best_params_to_csv
from Utils.Preprocessing.other_utils import split_data

# Set search spaces
if c.search_space == 'param1':
    import Gait.ParameterOptimization.ParamSearchSpace.param_space_1 as param_search_space
if c.search_space == 'param1small':
    import Gait.ParameterOptimization.ParamSearchSpace.param_space_1small as param_search_space
if c.search_space == 'param3small':
    import Gait.ParameterOptimization.ParamSearchSpace.param_space_3small as param_search_space

# Set algorithms
algs = c.algs
search_spaces = list()
objective_functions = list()
if 'lhs' in algs:
    search_spaces.append(param_search_space.space_single_side)
    objective_functions.append(o.objective_step_detection_single_side_lhs)
if 'fusion_high_level_intersect' in algs:
    search_spaces.append(param_search_space.space_fusion_high_level_intersect)
    objective_functions.append(o.step_detection_fusion_high_level_intersect)
if 'fusion_high_level_union_two_stages' in algs:
    search_spaces.append(param_search_space.space_fusion_high_level_union_two_stages)
    objective_functions.append(o.step_detection_fusion_high_level_union_two_stages)
if 'fusion_high_level_union_one_stage' in algs:
    search_spaces.append(param_search_space.space_fusion_high_level_union_one_stage)
    objective_functions.append(o.step_detection_fusion_high_level_union_one_stage)
if 'fusion_low_level_sum' in algs:
    search_spaces.append(param_search_space.space_fusion_low_level)
    objective_functions.append(o.step_detection_fusion_low_level_sum)
if 'fusion_low_level_diff' in algs:
    search_spaces.append(param_search_space.space_fusion_low_level)
    objective_functions.append(o.step_detection_fusion_low_level_diff)
if 'rhs' in algs:
    search_spaces.append(param_search_space.space_single_side)
    objective_functions.append(o.objective_step_detection_single_side_rhs)


# Set running parameters
n_folds = c.n_folds
max_evals = c.max_evals
alg = c.opt_alg
metric = c.metric_to_optimize
save_dir = join(c.results_path, 'param_opt')
do_verbose = c.do_verbose

##########################################################################################################
# Select samples for training
path_sample = join(c.pickle_path, 'metadata_sample')
with open(path_sample, 'rb') as fp:
    sample = pickle.load(fp)

# Only samples with step count
ids_notnull = np.array(sample[sample['StepCount'].notnull()].index, dtype=int)

# Remove People: JeremyAtia, EfratWasserman, and AvishaiWeingarten first trial with sternum sensor on back
ids_jeremy_atia = np.array(sample[sample['Person'] == 'JeremyAtia'].index, dtype=int)
ids_efrat_wasserman = np.array(sample[sample['Person'] == 'EfratWasserman'].index, dtype=int)
ids_avishai_weingarten = np.array(sample[sample['Person'] == 'AvishaiWeingarten'].index, dtype=int)
ids_sternum_sensor_incorrect = np.array(sample[sample['Comments'] == 'Sternum was on back'].index, dtype=int)
ids_avishai_sternum = np.intersect1d(ids_avishai_weingarten, ids_sternum_sensor_incorrect)
ids_people_to_remove = np.sort(np.hstack((ids_jeremy_atia, ids_efrat_wasserman, ids_avishai_sternum)))

# Remove outlier step counts as compared to APDM cadence
ids = np.setdiff1d(ids_notnull, ids_people_to_remove)
if c.outlier_percent_to_remove > 0:
    z = sample['CadenceDifference']
    ids = np.array(z[z < z.iloc[ids].quantile(1 - c.outlier_percent_to_remove/100.0)].index, dtype=int)

# Can also remove other samples labeled as ‘bad’ (n=~20), but excluding Chen Adamati sternum/back which was in fact good

##########################################################################################################
# Set data splits
task_ids = []
if c.tasks_to_optimize == 'split':
    walk_tasks = [1, 2, 3, 4, 5, 6, 7, 10]
    for k in walk_tasks:
        task_i = np.intersect1d(ids, sample[sample['TaskId'] == k]['SampleId'].as_matrix())
        task_ids.append(task_i)
elif c.tasks_to_optimize == 'all':
    walk_tasks = ['all']
    task_i = ids
    task_ids.append(task_i)
elif c.tasks_to_optimize == 'both_split_and_all':
    walk_tasks = [1, 2, 3, 4, 5, 6, 7, 10, 'all']
    for k in walk_tasks:
        if k == 'all':
            task_i = ids
        else:
            task_i = np.intersect1d(ids, sample[sample['TaskId'] == k]['SampleId'].as_matrix())
        task_ids.append(task_i)

train_all = []
test_all = []
for i in range(len(task_ids)):
    train_i, test_i = split_data(task_ids[i], n_folds=n_folds)
    train_all.append(train_i)
    test_all.append(test_i)


##########################################################################################################
# The parameter optimization code
df_gait_measures_by_alg = []
df_gait_measures_by_alg_split = []
df_gait_measures_by_alg_all = []

# Loop over the different algorithms (objective functions) to optimize
for i in range(len(objective_functions)):
    space = search_spaces[i]
    objective = objective_functions[i]
    obj_func_name = objective_functions[i].__name__.replace('objective_','')
    if alg == 'tpe':
        opt_algorithm = tpe.suggest
    elif alg == 'random':
        opt_algorithm = hp.rand.suggest
    else:
        opt_algorithm = tpe.suggest

    # Loop over the different training sets (walking tasks) on which to perform optimization
    df_gait_measures_by_task = []
    for j in range(len(walk_tasks)):
        best = []
        root_mean_squared_error = []
        mape = []
        train = train_all[j]
        test = test_all[j]
        results = []

        # Optimize each fold
        space['metric'] = metric
        space['verbose'] = do_verbose
        space['max_dist_from_apdm'] = c.max_dist_from_apdm_for_comparing_events

        if c.do_multi_core:
            # The parallel function. It is defined each time so that it uses various parameters from the outer
            # scope: objective, space, opt_algorithm, and max_evals

            # The parallel code
            # pool = Pool(processes=cpu_count())
            l = []
            for k in range(n_folds):
                l.append((space, train[k], objective, opt_algorithm, max_evals, k))
            pool = Pool(processes=n_folds)
            # results = pool.starmap(par_fmin, [('oo', 0), ('you', 1), ('yann', 2), ('bb',3), ('aa', 4)])
            results = pool.starmap(par_fmin, l)
            pool.close()
            pool.join()
        else:
            for k in range(n_folds):
                print('************************************************************************')
                print('\rOptimizing ' + c.metric_to_optimize + '. Optimizing Walk Task ' + str(walk_tasks[j]) + ': algorithm- ' + obj_func_name +
                      '    Search space: ' + c.search_space + '   Search type: ' + alg + '   Fold ' +
                      str(k + 1) + ' of ' + str(n_folds) + '. Max evals: ' + str(c.max_evals))
                print('************************************************************************')

                # Optimize parameters
                space['sample_ids'] = train[k]
                trials = Trials()
                res = fmin(objective, space, algo=opt_algorithm, max_evals=max_evals, trials=trials)
                results.append(res)

                # show progress
                with open(join(save_dir, obj_func_name + '_walk_task' + str(walk_tasks[j]) + '_fold_' +
                          str(k+1)), 'wb') as fp:
                    results2 = [results, train_all, test_all]
                    pickle.dump(results2, fp)

        ################################################################################################################
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

        ################################################################################################################
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

        ################################################################################################################
        # Save results
        save_results = dict()
        save_results['best'] = best
        save_results['rmse'] = root_mean_squared_error
        save_results['mape'] = mape
        save_results['train'] = train
        save_results['test'] = test
        r = pd.DataFrame(save_results)
        r.to_csv(join(save_dir, obj_func_name + '_walk_task' + str(walk_tasks[j]) + '_all.csv'))

        with open(join(save_dir, obj_func_name + '_walk_task' + str(walk_tasks[j]) + '_all'), 'wb') as fp:
            to_save = [save_results, train_all, test_all]
            pickle.dump(to_save, fp)

        ################################################################################################################
        # Print results
        print('**************************')
        print('Best params are:')
        for d in best:
            if 'verbose' in d:
                del d['verbose']
            if 'sample_ids' in d:
                del d['sample_ids']
            if 'metric' in d:
                del d['metric']
            if 'max_dist_from_apdm' in d:
                del d['max_dist_from_apdm']
        for key in best[0].keys():
            vals = [x[key] for x in best]
            print('\t' + key + ': ' + ', '.join(str(x) for x in vals))
        rmse_to_print = [(round(x, 1)) for x in root_mean_squared_error]
        print('RMSE results for all folds are: ' + ', '.join(str(x) for x in rmse_to_print))
        print('\n')

    if c.tasks_to_optimize == 'both_split_and_all':
        df_for_alg_i_split = pd.concat(df_gait_measures_by_task[0:-1])
        df_for_alg_i_all = pd.concat([df_gait_measures_by_task[-1]])
        df_gait_measures_by_alg_split.append(df_for_alg_i_split)
        df_gait_measures_by_alg_all.append(df_for_alg_i_all)
    else:
        df_for_alg_i = pd.concat(df_gait_measures_by_task)
        df_gait_measures_by_alg.append(df_for_alg_i)

########################################################################################################################
# Summarize best parameters
print('***Writing best params from all folds and all algs to csv***')
file_name = write_best_params_to_csv(save_dir, return_file_path=True)

# Summarize and save gait measure results
print('***Summarizing all results for each individual sample***')
data_file = join(save_dir, 'gait_measures.csv')
create_gait_measure_csvs(df_gait_measures_by_alg, data_file)

# Data analysis plots
metrics = ['cadence', 'step_time_asymmetry_median', 'stride_time_var', 'apdm_toe_off_asymmetry_median']
compare_to_apdm(data_file, algs, metrics, name_prefix="")

create_regression_performance_plot(data_file, 'MAPE', save_name='alg_performance_mape.png', rotate=True,
                                   show_plot=False)
create_regression_performance_plot(data_file, 'RMSE', save_name='alg_performance_rmse.png', rotate=True,
                                   show_plot=False)
create_regression_performance_plot(data_file, 'PE', save_name='alg_performance_pe.png', rotate=True,
                                   show_plot=False, y_min=-20)
