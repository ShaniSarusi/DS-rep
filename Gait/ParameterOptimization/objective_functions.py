# functions (i.e. algorithms) to optimize
from math import sqrt
from sklearn.metrics import mean_squared_error


def objective_single_side_lhs(p):
    # set data
    s = p['sc']

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
