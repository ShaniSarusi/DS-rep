import pickle
from copy import deepcopy
from os.path import join
import numpy as np
import hyperopt as hp
from hyperopt import fmin, Trials, tpe

import Gait.config as c
from Gait.Pipeline.StepDetection import StepDetection
from Utils.Preprocessing.other_utils import split_data
import Gait.ParameterOptimization.config_param_search as conf_params
from Gait.ParameterOptimization.objective_functions import all_algorithms
from Gait.ParameterOptimization.evaluate_test_set_function import evaluate_on_test_set

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

##########################################################################################################
# Running parameters
space = conf_params.space_single_side
objective_function = 'single_side_lhs'
do_spark = True
n_folds = 4
max_evals = 3
alg = 'random'  # Can be 'tpe' or 'random'

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
if alg == 'tpe':
    algorithm = tpe.suggest
elif alg == 'random':
    algorithm = hp.rand.suggest

if do_spark:
    # Prepare data to be split on workers
    rdd_data = []
    for i in range(n_folds):
        data_i = []
        trials = Trials()
        sd_train = deepcopy(sd)
        sd_train.select_specific_samples(train[i])
        space['sd'] = sd_train
        data_i.append(objective)
        data_i.append(space)
        data_i.append(algorithm)
        data_i.append(max_evals)
        data_i.append(trials)
        rdd_data.append(data_i)

    spark = SparkSession \
        .builder \
        .appName("Gait parameter optimization") \
        .config("spark.sql.shuffle.partitions", "3") \
        .getOrCreate()

    sc = spark.sparkContext
    # Run parallel code
    rdd = sc.parallelize(rdd_data)
    res_rdd = rdd.map(lambda x: fmin(x[0], x[1], algo=x[2], max_evals=x[3], trials=x[4]))
    results = res_rdd.collect()

else:
    results = []
    for i in range(n_folds):
        print('************************************************************************')
        print('\rOptimizing: ' + objective_function + '.   Using ' + alg + " search algorithm.   Running fold " +
              str(i + 1) + ' of ' + str(n_folds) + '.')
        print('************************************************************************')

        # Optimize params
        sd_train = deepcopy(sd)
        sd_train.select_specific_samples(train[i])
        space['sd'] = sd_train
        trials = Trials()
        res = fmin(objective, space, algo=algorithm, max_evals=max_evals, trials=trials)
        results.append(res)


##########################################################################################################
# Evaluate results on test set and print out
for i in range(n_folds):
    root_mean_squared_error_i, best_params_i = evaluate_on_test_set(space, results[i], sd, test[i], i, n_folds)
    root_mean_squared_error.append(root_mean_squared_error_i)
    best.append(best_params_i)


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
