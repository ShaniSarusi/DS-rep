import pickle
from copy import copy, deepcopy
from os.path import join
import numpy as np
from hyperopt import fmin, Trials, tpe, rand
from hyperopt import space_eval
import pprint

import Gait.config as c
from Gait.Pipeline.StepDetection import StepDetection
from Utils.Preprocessing.other_utils import split_data
import Gait.ParameterOptimization.config_param_search as conf_params
from Gait.ParameterOptimization.objective_functions import all_algorithms

##########################################################################################################
# Running parameters
space = conf_params.space_single_side
objective_function = 'single_side_lhs'
n_folds = 4
max_evals = 3
alg = 'random'  # 'tpe'


##########################################################################################################
# Load input data to algorithms
with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
    sample = pickle.load(fp)
with open(join(c.pickle_path, 'acc'), 'rb') as fp:
    acc = pickle.load(fp)
ids = sample[sample['StepCount'].notnull()].index.tolist()  # use only samples with step count
sc = StepDetection(acc, sample)
sc.select_specific_samples(ids)


##########################################################################################################
# The parameter optimization code
objective = all_algorithms[objective_function]
train, test = split_data(np.arange(len(ids)), n_folds=n_folds)
best = []
rmse = []
for i in range(n_folds):
    print('************************************************************************')
    print('\rOptimizing: ' + objective_function + '.   Using ' + alg + " search algorithm.   Running fold " +
          str(i + 1) + ' of ' + str(n_folds) + '.')
    print('************************************************************************\n')

    # optimize params
    sc_train = deepcopy(sc)
    sc_train.select_specific_samples(train[i])
    space['sc'] = sc_train
    trials = Trials()
    res = None
    if alg == 'tpe':
        res = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    elif alg == 'random':
        res = fmin(objective, space, algo=rand.suggest, max_evals=max_evals, trials=trials)

    # Evaluate result on test set
    best_params_i = space_eval(space, res)
    sc_test = copy(sc)
    sc_test.select_specific_samples(test[i])
    best_params_i['sc'] = sc_test
    rmse_i = objective(best_params_i)

    # Save cross validation fold results
    rmse.append(rmse_i)
    del best_params_i['sc']
    best.append(best_params_i)

    # Print cross validation fold results
    print("\nRMSE of fold " + str(i + 1) + ' from ' + str(n_folds) + ' is ' + str(
        round(rmse_i, 1)) + ". The param values are:")
    pprint.pprint(best_params_i)
    print()


##########################################################################################################
# Save results
results = dict()
results['best'] = best
results['rmse'] = rmse
results['train'] = train
results['test'] = test
with open(join(c.pickle_path, 'hypopt3'), 'wb') as fp:
    pickle.dump(results, fp)


##########################################################################################################
# Print results
print(best)
print(rmse)
