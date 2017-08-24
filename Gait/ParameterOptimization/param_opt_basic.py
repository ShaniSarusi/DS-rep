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
from Gait.ParameterOptimization.sum_results_for_plotting_parameters import sum_results_for_plotting_parameters
from Gait.ParameterOptimization.regression_performance_plot import create_regression_performance_plot
from Gait.Pipeline.gait_utils import gait_measure_analysis
from Utils.Preprocessing.other_utils import split_data
from multiprocessing import Pool, cpu_count


# Set search spaces
if c.search_space == 'fast':
    import Gait.Resources.param_search_space_fast as param_search_space
if c.search_space == 'small':
    import Gait.Resources.param_search_space_small as param_search_space
if c.search_space == 'full':
    import Gait.Resources.param_search_space_full as param_search_space
if c.search_space == 'fast2':
    import Gait.Resources.param_search_space_fast2 as param_search_space
if c.search_space == 'fast3':
    import Gait.Resources.param_search_space_fast3 as param_search_space
if c.search_space == 'fast4':
    import Gait.Resources.param_search_space_fast4 as param_search_space
if c.search_space == 'fast5':
    import Gait.Resources.param_search_space_fast5 as param_search_space
if c.search_space == 'param6':
    import Gait.Resources.param_space_6 as param_search_space
if c.search_space == 'param7':
    import Gait.Resources.param_space_7 as param_search_space
if c.search_space == 'param8':
    import Gait.Resources.param_space_8 as param_search_space

# Set algorithms
algs = c.algs
space_all = list()
objective_function_all = list()
if 'lhs' in algs:
    space_all.append(param_search_space.space_single_side_lhs)
    objective_function_all.append('step_detection_single_side_lhs')
if 'overlap' in algs:
    objective_function_all.append('step_detection_two_sides_overlap')
    space_all.append(param_search_space.space_overlap)
if 'overlap_strong' in algs:
    objective_function_all.append('step_detection_two_sides_overlap_strong')
    space_all.append(param_search_space.space_overlap_strong)
if 'combined' in algs:
    objective_function_all.append('step_detection_two_sides_combined_signal')
    space_all.append(param_search_space.space_combined)
if 'rhs' in algs:
    objective_function_all.append('step_detection_single_side_rhs')
    space_all.append(param_search_space.space_single_side_rhs)

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
if c.data_type == 'split':
    walk_tasks = [1, 2, 3, 4, 5, 6, 7, 10]
    for k in walk_tasks:
        task_i = np.intersect1d(ids, sample[sample['TaskId'] == k]['SampleId'].as_matrix())
        task_ids.append(task_i)
elif c.data_type == 'all':
    walk_tasks = ['all']
    task_i = ids
    task_ids.append(task_i)
elif c.data_type == 'both':
    walk_tasks = [1, 2, 3, 4, 5, 6, 7, 10, 'all']
    for k in walk_tasks:
        if k == 'all':
            task_i = ids
        else:
            task_i = np.intersect1d(ids, sample[sample['TaskId'] == k]['SampleId'].as_matrix())
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
# The two lists below are relevant only for c.data_type == 'both':
df_gait_measures_by_alg_split = []
df_gait_measures_by_alg_all = []

# Loop over the different algorithms (objective functions) to optimize
for i in range(len(objective_function_all)):
    objective_function = objective_function_all[i]
    space = space_all[i]

    objective = all_algorithms[objective_function]
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
        if c.do_multi_core:
            # The parallel function. It is defined each time so that it uses various parameters from the outer
            # scope: objective, space, opt_algorithm, and max_evals
            def par_fmin(k):
                print('************************************************************************')
                print('\rOptimizing Walk Task ' + str(
                    walk_tasks[j]) + ': algorithm- ' + objective_function + '.   Using '
                      + alg + " search.   Running fold " + str(k + 1) + ' of ' + str(n_folds) + '. Max evals: ' +
                      str(c.max_evals))
                print('************************************************************************')
                space['sample_ids'] = train[k]
                par_results = fmin(objective, space, algo=opt_algorithm, max_evals=max_evals, trials=Trials())
                return par_results

                # show progress
                # with open(
                #         join(save_dir, objective_function + '_walk_task' + str(walk_tasks[j]) + '_fold_' + str(k + 1)),
                #         'wb') as fp:
                #     results2 = [results, train_all, test_all]
                #     pickle.dump(results2, fp)

            # The parallel code
            # pool = Pool(processes=cpu_count())
            pool = Pool(processes=n_folds)
            results = pool.map(par_fmin, range(n_folds))
            pool.close()
            pool.join()
        else:
            for k in range(n_folds):
                print('************************************************************************')
                print('\rOptimizing Walk Task ' + str(walk_tasks[j]) + ': algorithm- ' + objective_function + '.   Using '
                      + alg + " search.   Running fold " + str(k + 1) + ' of ' + str(n_folds) + '. Max evals: ' +
                      str(c.max_evals))
                print('************************************************************************')

                # Optimize parameters
                space['sample_ids'] = train[k]
                trials = Trials()
                res = fmin(objective, space, algo=opt_algorithm, max_evals=max_evals, trials=trials)
                results.append(res)

                # show progress
                with open(join(save_dir, objective_function + '_walk_task' + str(walk_tasks[j]) + '_fold_' + str(k+1)),
                          'wb') as fp:
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
        r.to_csv(join(save_dir, objective_function + '_walk_task' + str(walk_tasks[j]) + '_all.csv'))

        with open(join(save_dir, objective_function + '_walk_task' + str(walk_tasks[j]) + '_all'), 'wb') as fp:
            to_save = [save_results, train_all, test_all]
            pickle.dump(to_save, fp)

        ################################################################################################################
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

########################################################################################################################
# Summarize and save all optimization results
print('***Summarizing and saving results***')
file_name = sum_results(save_dir, return_file_path=True)
sum_results_for_plotting_parameters(file_name, save_dir)

# Create performance metric plots
data_file = join(save_dir, file_name)
create_regression_performance_plot(data_file, 'MAPE', save_name='alg_performance_mape.png', rotate=True, show_plot=False)
create_regression_performance_plot(data_file, 'RMSE', save_name='alg_performance_rmse.png', rotate=True, show_plot=False)
create_regression_performance_plot(data_file, 'PE', save_name='alg_performance_rmse.png', rotate=True, show_plot=False)

# Summarize and save gait measure results
metrics = ['cadence', 'step_time_asymmetry', 'step_time_asymmetry2_median', 'stride_time_var',
           'toe_off_asymmetry_median']
if c.data_type == 'both':
    save_name1 = 'gait_measures_split.csv'
    gait_measure_analysis(df_gait_measures_by_alg_split, save_dir, save_name1, algs, metrics, prefix='split')
    save_name2 = 'gait_measures_all.csv'
    gait_measure_analysis(df_gait_measures_by_alg_all, save_dir, save_name2, algs, metrics, prefix='all')

    data_file = join(save_dir, save_name1)
    create_regression_performance_plot(data_file, 'MAPE', save_name='alg_performance_mape_split.png', rotate=True,
                                       show_plot=False)
    create_regression_performance_plot(data_file, 'RMSE', save_name='alg_performance_rmse_split.png', rotate=True,
                                       show_plot=False)
    create_regression_performance_plot(data_file, 'PE', save_name='alg_performance_rmse_split.png', rotate=True,
                                       show_plot=False)

    data_file = join(save_dir, save_name2)
    create_regression_performance_plot(data_file, 'MAPE', save_name='alg_performance_mape_all.png', rotate=True,
                                       show_plot=False)
    create_regression_performance_plot(data_file, 'RMSE', save_name='alg_performance_rmse_all.png', rotate=True,
                                       show_plot=False)
    create_regression_performance_plot(data_file, 'PE', save_name='alg_performance_rmse_all.png', rotate=True,
                                       show_plot=False)

else:
    save_name = 'gait_measures.csv'
    gait_measure_analysis(df_gait_measures_by_alg, save_dir, save_name, algs, metrics)
    data_file = join(save_dir, save_name)
    create_regression_performance_plot(data_file, 'MAPE', save_name='alg_performance_mape2.png', rotate=True,
                                       show_plot=False)
    create_regression_performance_plot(data_file, 'RMSE', save_name='alg_performance_rmse2.png', rotate=True,
                                       show_plot=False)
    create_regression_performance_plot(data_file, 'PE', save_name='alg_performance_rmse2.png', rotate=True,
                                       show_plot=False)