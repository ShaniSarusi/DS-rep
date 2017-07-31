from hyperopt import space_eval
import pprint


# Evaluate result on test set
def evaluate_on_test_set(p_space, p_res, test_set, objective, fold_i=None, folds=None, verbose=False):
    params = space_eval(p_space, p_res)
    params['sample_ids'] = test_set
    params['metric'] = 'both'
    root_mean_squared_error, mape = objective(params)

    # Print cross validation fold results
    if verbose:
        print("\nRMSE of fold " + str(fold_i + 1) + ' from ' + str(folds) + ' is ' + str(
            round(root_mean_squared_error, 1)) + ". The param values are:")
        pprint.pprint(params)
        print()

    return root_mean_squared_error, mape, params
