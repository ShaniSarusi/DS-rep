from hyperopt import space_eval
import pprint
from copy import copy


# Evaluate result on test set
def evaluate_on_test_set(p_space, p_res, test_set, objective, fold_i=None, folds=None, verbose=False):
    params = space_eval(p_space, p_res)
    params['sample_ids'] = test_set
    root_mean_squared_error = objective(params)

    # Print cross validation fold results
    if verbose:
        print("\nRMSE of fold " + str(fold_i + 1) + ' from ' + str(folds) + ' is ' + str(
            round(root_mean_squared_error, 1)) + ". The param values are:")
        pprint.pprint(params)
        print()

    return root_mean_squared_error, params


# Evaluate result on test set
def evaluate_on_test_set2(p_space, p_res, p_sd, test_set, objective, fold_i=None, folds=None, verbose=False):
    params = space_eval(p_space, p_res)
    sd_test = copy(p_sd)
    sd_test.select_specific_samples(test_set)
    params['sd'] = sd_test
    root_mean_squared_error = objective(params)
    del params['sd']

    # Print cross validation fold results
    if verbose:
        print("\nRMSE of fold " + str(fold_i + 1) + ' from ' + str(folds) + ' is ' + str(
            round(root_mean_squared_error, 1)) + ". The param values are:")
        pprint.pprint(params)
        print()

    return root_mean_squared_error, params
