# functions (i.e. algorithms) to optimize
from math import sqrt
from sklearn.metrics import mean_squared_error
from os.path import join
import Gait.config as c
from Gait.Pipeline.StepDetection import StepDetection
import pickle
from Utils.Connections.connections import load_pickle_file_from_s3
from Utils.BasicStatistics.statistics_functions import mean_absolute_percentage_error


def objective_step_detection_single_side(p):
    # Load input data to algorithms
    path_sample = join(c.pickle_path, 'metadata_sample')
    path_acc = join(c.pickle_path, 'acc')
    if c.run_on_cloud:
        sample = load_pickle_file_from_s3(c.aws_region_name, c.s3_bucket, path_sample)
        acc = load_pickle_file_from_s3(c.aws_region_name, c.s3_bucket, path_acc)
    else:
        with open(path_sample, 'rb') as fp:
            sample = pickle.load(fp)
        with open(path_acc, 'rb') as fp:
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

    s.step_detection_single_side(side=side, signal_to_use=signal_to_use, smoothing=smoothing, mva_win=mva_win,
                                         vert_win=vert_win, butter_freq=butter_freq, peak_type=peak_type,
                                         peak_param1=peak_param1, peak_param2=peak_param2,
                                         weak_signal_thresh=weak_signal_thresh, verbose=True)

    # ********** Calculate RMSE and/or MAPE
    s.add_gait_metrics(verbose=False)
    if metric == 'both':
        rmse = mean_absolute_percentage_error(s.res['sc_manual'], s.res['sc_' + side], handle_zeros=True)
        mape = mean_absolute_percentage_error(s.res['sc_manual'], s.res['sc_' + side], handle_zeros=True)
        print('\tResult: RMSE is ' + str(round(rmse, 2)))
        return rmse, mape
    elif metric == 'mape':
        res = mean_absolute_percentage_error(s.res['sc_manual'], s.res['sc_' + side], handle_zeros=True)
        metric_name = 'Mean Absolute Percentage Error'
    else:  # rmse
        res = sqrt(mean_squared_error(s.res['sc_manual'], s.res['sc_' + side]))
        metric_name = 'RMSE'
    print('\tResult: ' + metric_name + ' is ' + str(round(res, 2)))
    return res


def objective_step_detection_two_sides_overlap(p):
    # Load input data to algorithms
    path_sample = join(c.pickle_path, 'metadata_sample')
    path_acc = join(c.pickle_path, 'acc')
    if c.run_on_cloud:
        sample = load_pickle_file_from_s3(c.aws_region_name, c.s3_bucket, path_sample)
        acc = load_pickle_file_from_s3(c.aws_region_name, c.s3_bucket, path_acc)
    else:
        with open(path_sample, 'rb') as fp:
            sample = pickle.load(fp)
        with open(path_acc, 'rb') as fp:
            acc = pickle.load(fp)
    s = StepDetection(acc, sample)

    # Set sample ids for dataset
    ids = sample[sample['StepCount'].notnull()].index.tolist()
    s.select_specific_samples(ids)
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

    s.step_detection_two_sides_overlap(signal_to_use=signal_to_use, smoothing=smoothing, mva_win=mva_win,
                                       vert_win=vert_win, butter_freq=butter_freq, peak_type=peak_type,
                                       peak_param1=peak_param1, peak_param2=peak_param2, win_size_merge=win_size_merge,
                                       win_size_remove_adjacent_peaks=win_size_remove_adjacent_peaks, verbose=True)

    # ********** Calculate RMSE and/or MAPE
    s.add_gait_metrics(verbose=False)
    if metric == 'both':
        rmse = mean_absolute_percentage_error(s.res['sc_manual'], s.res['sc_overlap'], handle_zeros=True)
        mape = mean_absolute_percentage_error(s.res['sc_manual'], s.res['sc_overlap'], handle_zeros=True)
        print('\tResult: RMSE is ' + str(round(rmse, 2)))
        return rmse, mape
    elif metric == 'mape':
        res = mean_absolute_percentage_error(s.res['sc_manual'], s.res['sc_overlap'], handle_zeros=True)
        metric_name = 'Mean Absolute Percentage Error'
    else:  # rmse
        res = sqrt(mean_squared_error(s.res['sc_manual'], s.res['sc_overlap']))
        metric_name = 'RMSE'
    print('\tResult: ' + metric_name + ' is ' + str(round(res, 2)))
    return res


def objective_step_detection_two_sides_combined_signal(p):
    # Load input data to algorithms
    path_sample = join(c.pickle_path, 'metadata_sample')
    path_acc = join(c.pickle_path, 'acc')
    if c.run_on_cloud:
        sample = load_pickle_file_from_s3(c.aws_region_name, c.s3_bucket, path_sample)
        acc = load_pickle_file_from_s3(c.aws_region_name, c.s3_bucket, path_acc)
    else:
        with open(path_sample, 'rb') as fp:
            sample = pickle.load(fp)
        with open(path_acc, 'rb') as fp:
            acc = pickle.load(fp)
    s = StepDetection(acc, sample)

    # Set sample ids for dataset
    ids = sample[sample['StepCount'].notnull()].index.tolist()
    s.select_specific_samples(ids)
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

    s.step_detection_two_sides_combined_signal(signal_to_use=signal_to_use, smoothing=smoothing, mva_win=mva_win,
                                               vert_win=vert_win, butter_freq=butter_freq,
                                               mva_win_combined=mva_win_combined, min_hz=min_hz, max_hz=max_hz,
                                               factor=factor, peak_type=peak_type, peak_param1=peak_param1,
                                               peak_param2=peak_param2, verbose=True)

    # ********** Calculate RMSE and/or MAPE
    s.add_gait_metrics(verbose=False)
    if metric == 'both':
        rmse = mean_absolute_percentage_error(s.res['sc_manual'], s.res['sc_combined'], handle_zeros=True)
        mape = mean_absolute_percentage_error(s.res['sc_manual'], s.res['sc_combined'], handle_zeros=True)
        print('\tResult: RMSE is ' + str(round(rmse, 2)))
        return rmse, mape
    elif metric == 'mape':
        res = mean_absolute_percentage_error(s.res['sc_manual'], s.res['sc_combined'], handle_zeros=True)
        metric_name = 'Mean Absolute Percentage Error'
    else:  # rmse
        res = sqrt(mean_squared_error(s.res['sc_manual'], s.res['sc_combined']))
        metric_name = 'RMSE'
    print('\tResult: ' + metric_name + ' is ' + str(round(res, 2)))
    return res


# Store all algorithms in a dictionary
all_algorithms = dict()
all_algorithms['step_detection_single_side'] = objective_step_detection_single_side
all_algorithms['step_detection_single_side_lhs'] = objective_step_detection_single_side
all_algorithms['step_detection_single_side_rhs'] = objective_step_detection_single_side
all_algorithms['step_detection_two_sides_overlap'] = objective_step_detection_two_sides_overlap
all_algorithms['step_detection_two_sides_combined_signal'] = objective_step_detection_two_sides_combined_signal
