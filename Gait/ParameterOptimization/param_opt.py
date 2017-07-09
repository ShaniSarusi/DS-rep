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
do_spark = False
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
sd = StepDetection(acc, sample)
sd.select_specific_samples(ids)


##########################################################################################################
# The parameter optimization code
objective = all_algorithms[objective_function]
train, test = split_data(np.arange(len(ids)), n_folds=n_folds)
best = []
root_mean_squared_error = []
if do_spark:
    # Prepare data to be split on workers
    rdd_data = []
    for i in range(n_folds):
        data_i = []
        trials = Trials()
        sd_train = deepcopy(sd)
        sd_train.select_specific_samples(train[i])
        space['sc'] = sd_train
        data_i.append(objective)
        data_i.append(space)
        if alg == 'tpe':
            data_i.append(tpe.suggest)
        elif alg == 'random':
            data_i.append(rand.suggest)
        data_i.append(max_evals)
        data_i.append(trials)
        rdd_data.append(data_i)

    from pyspark.sql import SparkSession
    from pyspark.sql.types import *
    from pyspark.sql.functions import *

    spark = SparkSession \
        .builder \
        .appName("wiki pagecount") \
        .config("spark.sql.shuffle.partitions", "3") \
        .getOrCreate()

    sc = spark.sparkContext
    # Run parallel code
    rdd = sc.parallelize(rdd_data)
    res_rdd = rdd.map(lambda x: fmin(x[0], x[1], algo=x[2], max_evals=x[3], trials=x[4]))
    spark_results = res_rdd.collect()

    # Evaluate results on test set and print out
    for i in range(n_folds):
        # Define params for test set
        best_params_i = space_eval(space, spark_results[i])
        sc_test = copy(sd)
        sc_test.select_specific_samples(test[i])
        best_params_i['sc'] = sc_test

        # Run test once set using params
        root_mean_squared_error_i = objective(best_params_i)

        # Store params and root mean square error (RMSE)
        root_mean_squared_error.append(root_mean_squared_error_i)
        del best_params_i['sc']
        best.append(best_params_i)

        # Print cross validation fold results
        print("\nRMSE of fold " + str(i + 1) + ' from ' + str(n_folds) + ' is ' + str(
            round(root_mean_squared_error_i, 1)) + ". The param values are:")
        pprint.pprint(best_params_i)
        print()
else:
    for i in range(n_folds):
        print('************************************************************************')
        print('\rOptimizing: ' + objective_function + '.   Using ' + alg + " search algorithm.   Running fold " +
              str(i + 1) + ' of ' + str(n_folds) + '.')
        print('************************************************************************\n')

        # Optimize params
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
        root_mean_squared_error_i = objective(best_params_i)

        # Save cross validation fold results
        root_mean_squared_error.append(root_mean_squared_error_i)
        del best_params_i['sc']
        best.append(best_params_i)

        # Print cross validation fold results
        print("\nRMSE of fold " + str(i + 1) + ' from ' + str(n_folds) + ' is ' + str(
            round(root_mean_squared_error_i, 1)) + ". The param values are:")
        pprint.pprint(best_params_i)
        print()


##########################################################################################################
# Save results
results = dict()
results['best'] = best
results['rmse'] = root_mean_squared_error
results['train'] = train
results['test'] = test
with open(join(c.pickle_path, 'hypopt3'), 'wb') as fp:
    pickle.dump(results, fp)


##########################################################################################################
# Print results
print(best)
print(root_mean_squared_error)
