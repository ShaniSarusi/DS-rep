import pickle
from os.path import join
import numpy as np
import hyperopt as hp
from hyperopt import fmin, Trials, tpe
import pandas as pd

import Gait.config as c
from Gait.Pipeline.StepDetection import StepDetection
from Utils.Preprocessing.other_utils import split_data
from Gait.ParameterOptimization.objective_functions import all_algorithms
from Gait.ParameterOptimization.evaluate_test_set_function import evaluate_on_test_set

if c.search_space == 'full':
    import Gait.ParameterOptimization.param_search_space as param_search_space
elif c.search_space == 'small':
    import Gait.ParameterOptimization.param_search_space2 as param_search_space


##########################################################################################################
# Running parameters
space_all = list()
space_all.append(param_search_space.space_single_side_lhs)
space_all.append(param_search_space.space_overlap)
space_all.append(param_search_space.space_combined)
space_all.append(param_search_space.space_single_side_rhs)

objective_function_all = list()
objective_function_all.append('step_detection_single_side')
objective_function_all.append('step_detection_two_sides_overlap')
objective_function_all.append('step_detection_two_sides_combined_signal')

n_folds = c.n_folds
max_evals = c.max_evals
alg = 'tpe'  # Can be 'tpe' or 'random'

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
    for i in walk_tasks:
        task_i = np.intersect1d(filt['notnull'], sample[sample['TaskId'] == i]['SampleId'].as_matrix())
        task_ids.append(task_i)
elif c.data_type == 'all':
    walk_tasks = ['all']
    task_i = filt['notnull']
    task_ids.append(task_i)

train_all = []
test_all = []
for i in range(len(task_ids)):
    train_i, test_i = split_data(np.arange(len(task_ids[i])), n_folds=n_folds)
    train_all.append(train_i)
    test_all.append(test_i)


##########################################################################################################
# The parameter optimization code
for k in range(len(objective_function_all)):
    objective_function = objective_function_all[k]
    space = space_all[k]

    objective = all_algorithms[objective_function]
    if alg == 'tpe':
        algorithm = tpe.suggest
    elif alg == 'random':
        algorithm = hp.rand.suggest

    for j in range(len(walk_tasks)):
        best = []
        root_mean_squared_error = []
        train = train_all[j]
        test = test_all[j]
        results = []
        for i in range(n_folds):
            print('************************************************************************')
            print('\rOptimizing Walk Task ' + str(walk_tasks[j]) + ': algorithm- ' + objective_function + '.   Using ' + alg + " search.   Running fold " +
                  str(i + 1) + ' of ' + str(n_folds) + '.')
            print('************************************************************************')

            # Optimize parameters
            space['sample_ids'] = train[i]
            trials = Trials()
            res = fmin(objective, space, algo=algorithm, max_evals=max_evals, trials=trials)
            results.append(res)

            # show progress
            with open(join(c.results_path, 'param_opt', objective_function + '_walk_task' + str(walk_tasks[j]) + '_fold_' + str(i+1)), 'wb') as fp:
                results2 = [results, train_all, test_all]
                pickle.dump(results2, fp)


        ##########################################################################################################
        # Evaluate results on test set and print out
        for i in range(n_folds):
            root_mean_squared_error_i, best_params_i = evaluate_on_test_set(space, results[i], test[i], objective, i, n_folds)
            root_mean_squared_error.append(root_mean_squared_error_i)
            best.append(best_params_i)


        ##########################################################################################################
        # Save results
        results = dict()
        results['best'] = best
        results['rmse'] = root_mean_squared_error
        results['train'] = train
        results['test'] = test
        r = pd.DataFrame(results)
        r.to_csv(join(c.results_path, 'param_opt', objective_function + '_walk_task' + str(walk_tasks[j]) + '_all.csv'))

        with open(join(c.results_path, 'param_opt', objective_function + '_walk_task' + str(walk_tasks[j]) + '_all'), 'wb') as fp:
            results2 = [results, train_all, test_all]
            pickle.dump(results2, fp)


        ##########################################################################################################
        # Print results
        print(best)
        print(root_mean_squared_error)
