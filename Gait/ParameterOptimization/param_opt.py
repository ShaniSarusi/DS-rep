import pickle
from os.path import join
import numpy as np
import hyperopt as hp
from hyperopt import fmin, Trials, tpe

import Gait.config as c
from Utils.Connections.connections import load_pickle_file_from_s3, save_pickle_file_to_s3
from Gait.Pipeline.StepDetection import StepDetection
from Utils.Preprocessing.other_utils import split_data
import Gait.ParameterOptimization.param_search_space as param_search_space
from Gait.ParameterOptimization.objective_functions import all_algorithms
from Gait.ParameterOptimization.evaluate_test_set_function import evaluate_on_test_set

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

##########################################################################################################
# Running parameters
space = param_search_space.space_single_side
# space = param_search_space.space_overlap
# space = param_search_space.space_combined
objective_function = 'step_detection_single_side'
do_spark = False
n_folds = 4
max_evals = 3
alg = 'random'  # Can be 'tpe' or 'random'

##########################################################################################################
# Set data splits
path_sample = join(c.pickle_path, 'metadata_sample')
if c.run_on_cloud:
    sample = load_pickle_file_from_s3(c.aws_region_name, c.s3_bucket, path_sample)
else:
    with open(path_sample, 'rb') as fp:
        sample = pickle.load(fp)

ids = sample[sample['StepCount'].notnull()].index.tolist()  # use only samples with step count
train, test = split_data(np.arange(len(ids)), n_folds=n_folds)

##########################################################################################################
# The parameter optimization code
objective = all_algorithms[objective_function]
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
        space['sample_ids'] = train[i]
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
    if c.run_on_cloud:
        sc.addPyFile('DS.zip')

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

        # Optimize parameters
        space['sample_ids'] = train[i]
        trials = Trials()
        res = fmin(objective, space, algo=algorithm, max_evals=max_evals, trials=trials)
        results.append(res)


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

if c.run_on_cloud:
    file_name = 'hypopt_cloud_1'
    with open(file_name, 'wb') as fp:
        pickle.dump(results, fp)
    save_pickle_file_to_s3(c.aws_region_name, c.s3_bucket, file_name)

else:
    with open(join(c.pickle_path, 'hypopt_cloud_1'), 'wb') as fp:
        pickle.dump(results, fp)


##########################################################################################################
# Print results
print(best)
print(root_mean_squared_error)
