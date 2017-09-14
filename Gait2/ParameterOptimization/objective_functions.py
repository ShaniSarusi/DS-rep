# functions (i.e. algorithms) to optimize
from math import sqrt
from sklearn.metrics import mean_squared_error
from Gait.Pipeline.StepDetection import StepDetection
from Gait.Pipeline.gait_utils import create_sd_class_for_obj_functions
from Utils.BasicStatistics.statistics_functions import mean_absolute_percentage_error


def objective_step_detection_single_side(p):
    s = create_sd_class_for_obj_functions()

    # Set sample ids for dataset
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
    peak_param1 = None
    peak_param2 = None
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
    metric = p['metric']
    if 'verbose' not in p:
        verbose = True
    else:
        verbose = p['verbose']

    max_dist_from_apdm = p['max_dist_from_apdm']

    s.step_detection_single_side(side=side, signal_to_use=signal_to_use, smoothing=smoothing, mva_win=mva_win,
                                         vert_win=vert_win, butter_freq=butter_freq, peak_type=peak_type,
                                         peak_param1=peak_param1, peak_param2=peak_param2,
                                         weak_signal_thresh=weak_signal_thresh, verbose=verbose)

    # ********** Calculate RMSE and/or MAPE
    s.add_gait_metrics(verbose=False, max_dist_from_apdm=max_dist_from_apdm)
    rmse = sqrt(mean_squared_error(s.res['sc_manual'], s.res['sc_' + side]))
    mape = mean_absolute_percentage_error(s.res['sc_manual'], s.res['sc_' + side], handle_zeros=True)

    nanval = 0.5
    asym_rmse = sqrt(mean_squared_error(s.apdm_measures['toe_off_asymmetry_median'],
                                        s.res['step_time_asymmetry2_median_' + side].fillna(nanval)))

    if metric == 'both':
        if verbose: print('\tResult: RMSE is ' + str(round(rmse, 2)))
        return rmse, mape
    elif metric == 'sc_mape':
        if verbose: print('\tResult: Mean Absolute Percentage Error is ' + str(round(mape, 2)))
        return mape
    elif metric == 'sc_rmse' :
        if verbose: print('\tResult: RMSE ' + str(round(rmse, 2)))
        return rmse
    elif metric == 'get_res':
        return s.res
    elif metric == 'asym_rmse':
        if verbose: print('\tResult: Toe off asymmetry RMSE is ' + str(round(asym_rmse, 2)))
        return asym_rmse
    else:  # rmse
        if verbose: print('\tResult: RMSE ' + str(round(rmse, 2)))
        return rmse


def objective_step_detection_two_sides_overlap(p):
    # Load input data to algorithms
    s = create_sd_class_for_obj_functions()

    # Set sample ids for dataset
    s.select_specific_samples(p['sample_ids'])

    signal_to_use = p['signal_to_use']
    if signal_to_use == 'vertical' and p['do_windows_if_vertical']:
        vert_win = p['vert_win']
    else:
        vert_win = None
    smoothing = p['smoothing']
    mva_win = p['mva_win']
    butter_freq = p['butter_freq']
    peak_type = p['peak_type']
    peak_param1 = None
    peak_param2 = None
    if p['peak_type'] == 'scipy':
        peak_param1 = p['p1_sc']
        peak_param2 = p['p2_sc']
    elif p['peak_type'] == 'peak_utils':
        peak_param1 = p['p1_pu']
        peak_param2 = p['p2_pu']
    win_size_merge = p['win_size_merge']
    win_size_remove_adjacent_peaks = p['win_size_remove_adjacent_peaks']
    metric = p['metric']
    if 'verbose' not in p:
        verbose = True
    else:
        verbose = p['verbose']
    max_dist_from_apdm = p['max_dist_from_apdm']

    s.step_detection_two_sides_overlap(signal_to_use=signal_to_use, smoothing=smoothing, mva_win=mva_win,
                                       vert_win=vert_win, butter_freq=butter_freq, peak_type=peak_type,
                                       peak_param1=peak_param1, peak_param2=peak_param2, win_size_merge=win_size_merge,
                                       win_size_remove_adjacent_peaks=win_size_remove_adjacent_peaks, verbose=verbose)

    # ********** Calculate RMSE and/or MAPE
    s.add_gait_metrics(verbose=False, max_dist_from_apdm=max_dist_from_apdm)
    rmse = sqrt(mean_squared_error(s.res['sc_manual'], s.res['sc_overlap']))
    mape = mean_absolute_percentage_error(s.res['sc_manual'], s.res['sc_overlap'], handle_zeros=True)

    nanval = 0.5
    asym_rmse = sqrt(mean_squared_error(s.apdm_measures['toe_off_asymmetry_median'],
                                        s.res['step_time_asymmetry2_median_overlap'].fillna(nanval)))

    if metric == 'both':
        if verbose: print('\tResult: RMSE is ' + str(round(rmse, 2)))
        return rmse, mape
    elif metric == 'sc_mape':
        if verbose: print('\tResult: Mean Absolute Percentage Error is ' + str(round(mape, 2)))
        return mape
    elif metric == 'sc_rmse':
        if verbose: print('\tResult: RMSE ' + str(round(rmse, 2)))
        return rmse
    elif metric == 'get_res':
        return s.res
    elif metric == 'asym_rmse':
        if verbose: print('\tResult: Toe off asymmetry RMSE is ' + str(round(asym_rmse, 2)))
        return asym_rmse
    else:  # rmse
        if verbose: print('\tResult: RMSE ' + str(round(rmse, 2)))
        return rmse


def objective_step_detection_two_sides_overlap_strong(p):
    # Load input data to algorithms
    s = create_sd_class_for_obj_functions()

    # Set sample ids for dataset
    s.select_specific_samples(p['sample_ids'])

    signal_to_use = p['signal_to_use']
    if signal_to_use == 'vertical' and p['do_windows_if_vertical']:
        vert_win = p['vert_win']
    else:
        vert_win = None
    smoothing = p['smoothing']
    mva_win = p['mva_win']
    butter_freq = p['butter_freq']
    peak_type = p['peak_type']
    peak_param1 = None
    peak_param2 = None
    if p['peak_type'] == 'scipy':
        peak_param1 = p['p1_sc']
        peak_param2 = p['p2_sc']
    elif p['peak_type'] == 'peak_utils':
        peak_param1 = p['p1_pu']
        peak_param2 = p['p2_pu']
    win_size_merge = p['win_size_merge']
    win_size_remove_adjacent_peaks = p['win_size_remove_adjacent_peaks']
    metric = p['metric']
    z = p['z']
    if 'verbose' not in p:
        verbose = True
    else:
        verbose = p['verbose']
    max_dist_from_apdm = p['max_dist_from_apdm']

    s.step_detection_two_sides_overlap_opt_strong(signal_to_use=signal_to_use, smoothing=smoothing, mva_win=mva_win,
                                       vert_win=vert_win, butter_freq=butter_freq, peak_type=peak_type,
                                       peak_param1=peak_param1, peak_param2=peak_param2, win_size_merge=win_size_merge,
                                       win_size_remove_adjacent_peaks=win_size_remove_adjacent_peaks, z=z,
                                       verbose=verbose)

    # ********** Calculate RMSE and/or MAPE
    s.add_gait_metrics(verbose=False, max_dist_from_apdm=max_dist_from_apdm)
    rmse = sqrt(mean_squared_error(s.res['sc_manual'], s.res['sc_overlap_strong']))
    mape = mean_absolute_percentage_error(s.res['sc_manual'], s.res['sc_overlap_strong'], handle_zeros=True)

    nanval = 0.5
    asym_rmse = sqrt(mean_squared_error(s.apdm_measures['toe_off_asymmetry_median'],
                                        s.res['step_time_asymmetry2_median_overlap_strong'].fillna(nanval)))

    if metric == 'both':
        if verbose: print('\tResult: RMSE is ' + str(round(rmse, 2)))
        return rmse, mape
    elif metric == 'sc_mape':
        if verbose: print('\tResult: Mean Absolute Percentage Error is ' + str(round(mape, 2)))
        return mape
    elif metric == 'sc_rmse':
        if verbose: print('\tResult: RMSE ' + str(round(rmse, 2)))
        return rmse
    elif metric == 'get_res':
        return s.res
    elif metric == 'asym_rmse':
        if verbose: print('\tResult: Toe off asymmetry RMSE is ' + str(round(asym_rmse, 2)))
        return asym_rmse
    else:  # rmse
        if verbose: print('\tResult: RMSE ' + str(round(rmse, 2)))
        return rmse


def objective_step_detection_two_sides_combined_signal(p):
    # Load input data to algorithms
    s = create_sd_class_for_obj_functions()

    # Set sample ids for dataset
    s.select_specific_samples(p['sample_ids'])

    signal_to_use = p['signal_to_use']
    if signal_to_use == 'vertical' and p['do_windows_if_vertical']:
        vert_win = p['vert_win']
    else:
        vert_win = None
    smoothing = p['smoothing']
    mva_win = p['mva_win']
    butter_freq = p['butter_freq']
    mva_win_combined = p['mva_win_combined']
    min_hz = p['min_hz']
    max_hz = p['max_hz']
    factor = p['factor']
    peak_type = p['peak_type']
    peak_param1 = None
    peak_param2 = None
    if p['peak_type'] == 'scipy':
        peak_param1 = p['p1_sc']
        peak_param2 = p['p2_sc']
    elif p['peak_type'] == 'peak_utils':
        peak_param1 = p['p1_pu']
        peak_param2 = p['p2_pu']
    metric = p['metric']
    if 'verbose' not in p:
        verbose = True
    else:
        verbose = p['verbose']
    max_dist_from_apdm = p['max_dist_from_apdm']

    s.step_detection_two_sides_combined_signal(signal_to_use=signal_to_use, smoothing=smoothing, mva_win=mva_win,
                                               vert_win=vert_win, butter_freq=butter_freq,
                                               mva_win_combined=mva_win_combined, min_hz=min_hz, max_hz=max_hz,
                                               factor=factor, peak_type=peak_type, peak_param1=peak_param1,
                                               peak_param2=peak_param2, verbose=verbose)

    # ********** Calculate RMSE and/or MAPE
    s.add_gait_metrics(verbose=False, max_dist_from_apdm=max_dist_from_apdm)
    rmse = sqrt(mean_squared_error(s.res['sc_manual'], s.res['sc_combined']))
    mape = mean_absolute_percentage_error(s.res['sc_manual'], s.res['sc_combined'], handle_zeros=True)

    nanval = 0.5
    asym_rmse = sqrt(mean_squared_error(s.apdm_measures['toe_off_asymmetry_median'],
                                        s.res['step_time_asymmetry2_median_combined'].fillna(nanval)))

    if metric == 'both':
        if verbose: print('\tResult: RMSE is ' + str(round(rmse, 2)))
        return rmse, mape
    elif metric == 'sc_mape':
        if verbose: print('\tResult: Mean Absolute Percentage Error is ' + str(round(mape, 2)))
        return mape
    elif metric == 'sc_rmse':
        if verbose: print('\tResult: RMSE ' + str(round(rmse, 2)))
        return rmse
    elif metric == 'get_res':
        return s.res
    elif metric == 'asym_rmse':
        if verbose: print('\tResult: Toe off asymmetry RMSE is ' + str(round(asym_rmse, 2)))
        return asym_rmse
    else:  # rmse
        if verbose: print('\tResult: RMSE ' + str(round(rmse, 2)))
        return rmse


# Store all algorithms in a dictionary
all_algorithms = dict()
all_algorithms['step_detection_single_side'] = objective_step_detection_single_side
all_algorithms['step_detection_single_side_lhs'] = objective_step_detection_single_side
all_algorithms['step_detection_single_side_rhs'] = objective_step_detection_single_side
all_algorithms['step_detection_two_sides_overlap'] = objective_step_detection_two_sides_overlap
all_algorithms['step_detection_two_sides_overlap_strong'] = objective_step_detection_two_sides_overlap_strong
all_algorithms['step_detection_two_sides_combined_signal'] = objective_step_detection_two_sides_combined_signal
