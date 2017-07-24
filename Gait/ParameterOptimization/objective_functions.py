# functions (i.e. algorithms) to optimize
from math import sqrt
from sklearn.metrics import mean_squared_error
from copy import copy


def objective_step_detection_single_side(p):
    s = copy(p['sd'])
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
    return res_rmse


# TODO erase the function below once the function above works
def objective_single_side_lhs(p):
    # set data
    s = p['sd']

    # choose norm, vertical, or vertical with windows
    s.select_signal(p['signal'])

    # choose smoothing  TODO add an option for no smoothing
    if p['smoothing'] == 'mva':
        s.mva(win_size=p['mva_win'])
    elif p['smoothing'] == 'butter':
        s.bf(p_type='lowpass', order=5, freq=p['butter_freq_single_side'])

    # do mean normalization
    s.mean_normalization()

    # do peak finding
    if p['peak_type'] == 'scipy':
        s.step_detect_single_side_wpd_method(side='lhs', peak_type=p['peak_type'], p1=p['p1_sc'], p2=p['p1_sc'] + p['p2_sc'], verbose=False)
    elif p['peak_type'] == 'peak_utils':
        s.step_detect_single_side_wpd_method(side='lhs', peak_type=p['peak_type'], p1=p['p1_pu'], p2=p['p2_pu'], verbose=False)

    # remove weak signals
    if p['remove_weak_signals']:
        s.remove_weak_signals(p['weak_signal_thresh'])

    # ********** Calculate RMSE
    for j in range(s.res.shape[0]):
        s.res.set_value(s.res.index[j], 'sc3_lhs', len(s.res.iloc[j]['idx3_lhs']))
    res_rmse = sqrt(mean_squared_error(s.res['sc_true'], s.res['sc3_lhs']))
    return res_rmse


# Store all algorithms in a dictionary
all_algorithms = dict()
all_algorithms['single_side_lhs'] = objective_single_side_lhs
all_algorithms['step_detection_single_side'] = objective_step_detection_single_side
