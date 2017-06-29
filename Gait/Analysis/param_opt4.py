from os.path import join
import pickle
from hyperopt import space_eval
from hyperopt import fmin, Trials, tpe, rand
from Gait.Pipeline.StepDetection import StepDetection
import Gait.config as c
from Gait.Analysis.config_param_search import space_single_side
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from copy import deepcopy
from Utils.Preprocessing.other_utils import split_data

##########################################################################################################
with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
    sample = pickle.load(fp)
with open(join(c.pickle_path, 'acc'), 'rb') as fp:
    acc = pickle.load(fp)
ids = sample[sample['StepCount'].notnull()].index.tolist()  # use only samples with step count
sc = StepDetection(acc, sample)
sc.select_specific_samples(ids)


def objective(p):
    s = p['sc']
    s.select_signal(p['signal'])
    if p['smoothing'] == 'mva':
        s.mva(win_size=p['mva_win'])
    elif p['smoothing'] == 'butter':
        s.bf(p_type='lowpass', order=5, freq=p['butter_freq_single_side'])
        s.mean_normalization()

    if p['peaktype'] == 'scipy':
        s.step_detect_single_side_wpd_method(side='lhs', peak_type=p['peaktype'], p1=p['p1_sc'], p2=p['p1_sc'] +
                                                                                                    p['p2_sc'])
    elif p['peaktype'] == 'peak_utils':
        s.step_detect_single_side_wpd_method(side='lhs', peak_type=p['peaktype'], p1=p['p1_pu'], p2=p['p2_pu'])
    if p['remove_weak_signals']:
        s.remove_weak_signals(p['weak_signal_thresh'])
    for j in range(s.res.shape[0]):
        s.res.set_value(s.res.index[j], 'sc3_lhs', len(s.res.iloc[j]['idx3_lhs']))
    res_rmse = sqrt(mean_squared_error(s.res['sc_true'], s.res['sc3_lhs']))
    return res_rmse

##########################################################################################################
space = space_single_side

# The Trials object will store details of each iteration
trials = Trials()

# Run the hyperparameter search using the tpe algorithm
n_folds = 5
train, test = split_data(np.arange(len(ids)), n_folds=n_folds)
best = []
rmse = []
max_evals = 1000
alg = 'random'  # 'tpe'
for i in range(n_folds):
    print("\rRunning fold " + str(i + 1) + ' from ' + str(n_folds))
    # optimize params
    sc_train = deepcopy(sc)
    sc_train.select_specific_samples(train[i])
    space['sc'] = sc_train
    if 'alg' == 'tpe':
        res = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    elif 'alg' == 'random':
        res = fmin(objective, space, algo=rand.suggest, max_evals=max_evals, trials=trials)
    best_params_i = space_eval(space, res)
    best.append(best_params_i)

    # test result
    sc_test = deepcopy(sc)
    sc_test.select_specific_samples(test[i])
    best_params_i['sc'] = sc_test
    rmse_i = objective(best_params_i)
    print("\rRMSE of fold " + str(i + 1) + ' from ' + str(n_folds) + ' is ' + str(rmse_i))
    rmse.append(rmse_i)

##########################################################################################################
# Results

print(best)
print(rmse)
# save params
results = dict()
results['best'] = best
results['rmse'] = rmse

with open(join(c.pickle_path, 'hypopt2'), 'wb') as fp:
    pickle.dump(results, fp)
