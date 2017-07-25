# functions (i.e. algorithms) to optimize
from math import sqrt
from sklearn.metrics import mean_squared_error
from copy import copy
from os.path import join
import Gait.config as c
from Gait.Pipeline.StepDetection import StepDetection
import pickle


def objective_step_detection_single_side(p):
    # Load input data to algorithms
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
        sample = pickle.load(fp)
    with open(join(c.pickle_path, 'acc'), 'rb') as fp:
        acc = pickle.load(fp)
    s = StepDetection(acc, sample)

    # Set sample ids for dataset
    ids = sample[sample['StepCount'].notnull()].index.tolist()
    s.select_specific_samples(ids)
    s.select_specific_samples(p['sample_ids'])


    side = p['side']
    signal_to_use = p['signal_to_use']
    if signal_to_use == 'vertical' and p['do_windows_if_vertical']:
        vert_win = p['vert_win']
    else:
        vert_win = None
    smoothing = p['smoothing']
    mva_win = p['mva_win']
    butter_freq = p['butter_freq']
    peak_type = p['peak_type']
    if p['peak_type'] == 'scipy':
        peak_param1 = p['p1_sc']
        peak_param2 = p['p2_sc']
    elif p['peak_type'] == 'peak_utils':
        peak_param1 = p['p1_pu']
        peak_param2 = p['p2_pu']
    if p['remove_weak_signals']:
        weak_signal_thresh = p['weak_signal_thresh']
    else:
        weak_signal_thresh = None

    s.step_detection_single_side(side=side, signal_to_use=signal_to_use, smoothing=smoothing, mva_win=mva_win,
                                         vert_win=vert_win, butter_freq=butter_freq, peak_type=peak_type,
                                         peak_param1=peak_param1, peak_param2=peak_param2,
                                         weak_signal_thresh=weak_signal_thresh, verbose=True)

    # ********** Calculate RMSE
    s.add_gait_metrics(verbose=False)
    res_rmse = sqrt(mean_squared_error(s.res['sc_manual'], s.res['sc_' + side]))
    print('\tResult: RMSE is ' + str(round(res_rmse,2)))
    return res_rmse


# Store all algorithms in a dictionary
all_algorithms = dict()
all_algorithms['step_detection_single_side'] = objective_step_detection_single_side
