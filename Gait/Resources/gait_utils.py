import copy
import pickle
from os.path import join
import Gait.Resources.config as c
import numpy as np
import pandas as pd
from Gait.Pipeline.StepDetection import StepDetection
from Utils.Connections.connections import load_pickle_file_from_s3
from hyperopt import space_eval
import pprint
from math import sqrt
from sklearn.metrics import mean_squared_error
from Utils.BasicStatistics.statistics_functions import mean_absolute_percentage_error
from hyperopt import fmin, Trials


def par_fmin(space, train, objective, opt_algorithm, max_evals, k_iter):
    print('************************************************************************')
    # print('\rOptimizing ' + c.metric_to_optimize + '. Optimizing Walk Task ' + str(
    #     walk_tasks[j]) + ': algorithm- ' + obj_func_name +
    #       '    Search space: ' + c.search_space + '   Search type: ' + alg + '   Fold ' +
    #       str(k_iter + 1) + ' of ' + str(n_folds) + '. Max evals: ' + str(c.max_evals))
    # print('************************************************************************')
    space['sample_ids'] = train[k_iter]
    s = create_sd_class_for_obj_functions()
    s.normalize_norm()
    s.select_specific_samples(space['sample_ids'])
    space['s'] = s
    par_results = fmin(objective, space, algo=opt_algorithm, max_evals=max_evals, trials=Trials())
    return par_results


def create_sd_class_for_obj_functions():
    # Load input data to algorithms
    path_sample = join(c.pickle_path, 'metadata_sample')
    path_acc = join(c.pickle_path, 'acc')
    path_apdm_events = join(c.pickle_path, 'apdm_events')
    path_apdm_measures = join(c.pickle_path, 'apdm_measures')
    if c.run_on_cloud:
        sample = load_pickle_file_from_s3(c.aws_region_name, c.s3_bucket, path_sample)
        acc = load_pickle_file_from_s3(c.aws_region_name, c.s3_bucket, path_acc)
        apdm_measures = load_pickle_file_from_s3(c.aws_region_name, c.s3_bucket, path_apdm_measures)
        apdm_events = load_pickle_file_from_s3(c.aws_region_name, c.s3_bucket, path_apdm_events)
    else:
        with open(path_sample, 'rb') as fp:
            sample = pickle.load(fp)
        with open(path_acc, 'rb') as fp:
            acc = pickle.load(fp)
        with open(path_apdm_measures, 'rb') as fp:
            apdm_measures = pickle.load(fp)
        with open(path_apdm_events, 'rb') as fp:
            apdm_events = pickle.load(fp)
    sd = StepDetection(acc, sample, apdm_measures, apdm_events)
    return sd


def split_by_person():
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
        sample = pickle.load(fp)
    # split sample[perons]
    names = pd.unique(sample['Person'])
    filt = dict()
    for name in names:
        filt[name] = sample[sample['Person'] == name]['SampleId'].as_matrix()
    return filt


def create_gait_measure_csvs(df, p_gait_measure_path):
    res_gait = df[0]
    for i in range(1, len(df)):
        right = df[i]
        cols_left = res_gait.columns.tolist()
        cols_right = right.columns.tolist()
        cols_shared = list(set(cols_left).intersection(cols_right))
        right = right.drop(cols_shared, axis=1)
        res_gait = res_gait.join(right, how='outer')
    res_gait = res_gait.sort_index()
    res_gait.to_csv(p_gait_measure_path)


def set_filters(exp=2):
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
        sample = pickle.load(fp)
    filt = dict()
    # Shared
    filt['notnull'] = sample[sample['StepCount'].notnull()].index.tolist()
    if exp == 1:
        reg = sample[sample['WalkType'] == 'Regular']['SampleId'].as_matrix()
        fast = sample[sample['PaceInstructions'] == 'H']['SampleId'].as_matrix()
        med = sample[sample['PaceInstructions'] == 'M']['SampleId'].as_matrix()
        slow = sample[sample['PaceInstructions'] == 'L']['SampleId'].as_matrix()

        filt['reg'] = np.intersect1d(filt['notnull'], reg)
        filt['pd_sim'] = np.setdiff1d(filt['notnull'], reg)
        filt['reg_fast'] = np.intersect1d(filt['reg'], fast)
        filt['reg_med'] = np.intersect1d(filt['reg'], med)
        filt['reg_slow'] = np.intersect1d(filt['reg'], slow)
    if exp == 2:
        # Type
        filt['type_walk'] = sample[sample['Type'] == 'Walk']['SampleId'].as_matrix()
        filt['type_tug'] = sample[sample['Type'] == 'Tug']['SampleId'].as_matrix()
        # Arm status
        filt['arm_status_free'] = sample[sample['ArmStatus'] == 'Free']['SampleId'].as_matrix()
        filt['arm_status_one_fixed'] = sample[sample['ArmStatus'] == 'One fixed']['SampleId'].as_matrix()
        filt['arm_status_fixed'] = sample[sample['ArmStatus'] == 'Fixed']['SampleId'].as_matrix()
        filt['arm_status_tug'] = sample[sample['ArmStatus'] == 'Tug']['SampleId'].as_matrix()
        # Arm swing degree
        filt['arm_swing_high'] = sample[sample['ArmSwing'] == 'High']['SampleId'].as_matrix()
        filt['arm_swing_medium'] = sample[sample['ArmSwing'] == 'Medium']['SampleId'].as_matrix()
        filt['arm_swing_low'] = sample[sample['ArmSwing'] == 'Low']['SampleId'].as_matrix()
        filt['arm_swing_asymmetric'] = sample[sample['ArmSwing'] == 'Asymmetric']['SampleId'].as_matrix()
        filt['arm_swing_tug'] = sample[sample['ArmSwing'] == 'Tug']['SampleId'].as_matrix()
        # Step Asymmetry
        filt['step_asymmetry_none'] = sample[sample['StepAsymmetry'] == 'None']['SampleId'].as_matrix()
        filt['step_asymmetry_high'] = sample[sample['StepAsymmetry'] == 'High']['SampleId'].as_matrix()
        filt['step_asymmetry_medium'] = sample[sample['StepAsymmetry'] == 'Medium']['SampleId'].as_matrix()
        filt['step_asymmetry_low'] = sample[sample['StepAsymmetry'] == 'Low']['SampleId'].as_matrix()
        filt['step_asymmetry_tug'] = sample[sample['StepAsymmetry'] == 'Tug']['SampleId'].as_matrix()
        # Quality
        filt['quality_good'] = sample[sample['Quality'] == 'Good']['SampleId'].as_matrix()
        filt['quality_bad'] = sample[sample['Quality'] == 'Bad']['SampleId'].as_matrix()
        # Task ID
        num_tasks = 10
        for i in range(num_tasks):
            name = 'task_' + str(i + 1)
            filt[name] = sample[sample['TaskId'] == (i + 1)]['SampleId'].as_matrix()
    return filt


def truncate(data, front, back):
    out = []
    for i in range(len(data)):
        n = len(data[i]['rhs'])
        f = int(n * front / 100.0)
        b = int(n * (1.0 - back / 100.0))
        tmp = dict()
        tmp['rhs'] = data[i]['rhs'].iloc[range(f, b)]
        tmp['lhs'] = data[i]['lhs'].iloc[range(f, b)]
        out.append(copy.deepcopy(tmp))
    return out


def combine_both_single(b, l, r, win_size=30):
    single = np.unique([l, r])
    for i in range(len(single)):
        if np.min(np.abs(b-single[i])) > win_size:
            b.append(single[i])
    return b


def calculate_time_duration_of_samples():
    with open(join(c.pickle_path, 'acc'), 'rb') as fp:
        acc = pickle.load(fp)
    res = []
    for i in range(len(acc)):
        val = (acc[i]['lhs']['ts'].iloc[-1] - acc[i]['lhs']['ts'].iloc[0]).total_seconds()
        res.append(val)
    return res


def calc_asymmetry(side1, side2):
    m = np.mean([side1, side2])
    if m <= 0:
        return np.nan
    asymmetry = np.abs(side1 - side2) / m
    return asymmetry


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


def get_obj_function_results(s, alg_name, metric, verbose=True):
    if metric == 'get_res':
        return s.res
    elif metric == 'both':
        rmse = sqrt(mean_squared_error(s.res['sc_manual'], s.res['sc_' + alg_name]))
        mape = mean_absolute_percentage_error(s.res['sc_manual'], s.res['sc_' + alg_name], handle_zeros=True)
        if verbose: print('\tResult: RMSE is ' + str(round(rmse, 2)))
        return rmse, mape
    elif metric == 'sc_mape':
        mape = mean_absolute_percentage_error(s.res['sc_manual'], s.res['sc_' + alg_name], handle_zeros=True)
        if verbose: print('\tResult: Mean Absolute Percentage Error is ' + str(round(mape, 2)))
        return mape
    elif metric == 'sc_rmse':
        rmse = sqrt(mean_squared_error(s.res['sc_manual'], s.res['sc_' + alg_name]))
        if verbose: print('\tResult: RMSE ' + str(round(rmse, 2)))
        return rmse
    elif metric == 'asym_rmse':
        nanval = 0.5
        asym_rmse = sqrt(mean_squared_error(s.apdm_measures['apdm_toe_off_asymmetry_median'],
                                            s.res['step_time_asymmetry_median_' + alg_name].fillna(nanval)))
        if verbose: print('\tResult: Toe off asymmetry RMSE is ' + str(round(asym_rmse, 2)))
        return asym_rmse
    else:  # rmse
        rmse = sqrt(mean_squared_error(s.res['sc_manual'], s.res['sc_' + alg_name]))
        if verbose: print('\tResult: RMSE ' + str(round(rmse, 2)))
        return rmse
