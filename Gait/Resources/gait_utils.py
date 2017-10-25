import copy
import pickle
import re
import numpy as np
import pandas as pd
import pprint
import ast
from math import sqrt
from sklearn.metrics import mean_squared_error
from os.path import join
import matplotlib.pyplot as plt
from hyperopt import space_eval
# Imports from within package
from Gait.Pipeline.StepDetection import StepDetection
import Gait.Resources.config as c
# Imports from outside package
from Utils.BasicStatistics.statistics_functions import mean_absolute_percentage_error
from Utils.DataHandling.reading_and_writing_files import read_all_files_in_directory
from Utils.Connections.connections import load_pickle_file_from_s3



def create_sd_class_for_obj_functions(force_local=False):
    # Load input data to algorithms
    path_sample = join(c.pickle_path, 'metadata_sample')
    path_acc = join(c.pickle_path, 'acc')
    path_apdm_events = join(c.pickle_path, 'apdm_events')
    path_apdm_measures = join(c.pickle_path, 'apdm_measures')
    if c.run_on_cloud and not force_local:
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


def evaluate_on_test_set(p_space, p_res, test_set, objective, fold_i=None, folds=None, verbose=False, do_cadence=False):
    params = space_eval(p_space, p_res)
    params['sample_ids'] = test_set
    if do_cadence:
        params['metric'] = 'apdm_cad_rmse'
        root_mean_squared_error = objective(params)
        mape = 1
    else:
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
    elif metric == 'apdm_cad_rmse':
        nanval = 50
        apdm_cad_rmse = sqrt(mean_squared_error(s.apdm_measures['cadence'],
                                            s.res['cadence_apdm_' + alg_name].fillna(nanval)))
        if verbose: print('\tResult: APDM cadence RMSE is ' + str(round(apdm_cad_rmse, 2)))
        return apdm_cad_rmse
    else:  # rmse
        rmse = sqrt(mean_squared_error(s.res['sc_manual'], s.res['sc_' + alg_name]))
        if verbose: print('\tResult: RMSE ' + str(round(rmse, 2)))
        return rmse


def calc_sc_for_excluded_ids(file_path, excluded_ids, alg_name='high_level_union_one_s'):
    params = pd.read_csv(file_path)
    n_folds = len(params)

    values = []
    for i in range(n_folds):
        p = ast.literal_eval(params['best'][i])
        s = create_sd_class_for_obj_functions(force_local=True)
        s.normalize_norm()
        s.select_specific_samples(excluded_ids)
        if alg_name in 'fusion_high_level_union_one_stage':
            s.step_detection_fusion_high_level(signal_to_use='norm', vert_win=None, use_single_max_min_for_all_samples=True,
                smoothing='mva', mva_win=p['mva_win'], peak_min_thr=p['peak_min_thr'], peak_min_dist=p['peak_min_dist'],
                fusion_type='union_one_stage', union_min_dist=p['union_min_dist'], verbose=False, do_normalization=False)
        else:
            print('alg is not yet implemented')
            return
        s.add_gait_metrics()
        values_fold_i = s.res['cadence_apdm_fusion_high_level_union_one_stage'].as_matrix()
        values.append(values_fold_i)

    res = np.array(values)
    res = np.mean(res, axis=0)
    return res.tolist()


def bp_me(vals, labels=['Left', 'Right', 'Sum', 'Diff', 'Int', 'Union'], save_name=None, ylabel=None):
    fig, ax = plt.subplots()
    x = [1, 1.5, 2.5, 3, 4, 4.5]
    box = plt.boxplot([vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]], 0, '', positions=x,
                labels=labels, widths=0.4, whis=[5, 95], patch_artist=True)
    colors = ['cyan', 'cyan', 'lightgreen', 'lightgreen', 'pink', 'pink']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.setp(box['medians'], color='k')
    plt.yticks(fontsize=10)
    if ylabel is None:
        plt.ylabel('Step count\n(percent error)', fontsize=11)
    else:
        plt.ylabel(ylabel, fontsize=11)
    plt.xticks(fontsize=9)
    plt.tight_layout()
    ax = fig.gca()
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    fig = plt.gcf()
    fig.set_size_inches(3.5, 2.75)
    fig.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()


def write_best_params_to_csv(save_dir, return_file_path=False):
    # Start code
    with open(join(c.pickle_path, 'task_filters'), 'rb') as fp:
        task_filters = pickle.load(fp)
    res = pd.DataFrame(columns=['Algorithm', 'WalkingTask', 'TaskNum', 'FoldNum', 'RMSE', 'MAPE', 'Mva_win',
                                'Peak_min_thr', 'Peak_min_dist', 'Intersect_win', 'Union_min_dist', 'Union_min_thr'])
    f = read_all_files_in_directory(dir_path=save_dir, file_type='csv')
    for i in range(len(f)):
        if '_walk' not in f[i]:
            continue
        if 'task' not in f[i]:
            continue
        alg = re.search('(.*)_walk', f[i]).group(1)
        training_data = re.search('task(.*)_', f[i]).group(1)
        if training_data == 'all':
            task_name = 'all'
        else:
            task_name = task_filters[task_filters['TaskId'] == int(training_data)]['Task Name'].iloc[0]

        f_i = pd.read_csv(join(save_dir, f[i]))
        for j in range(f_i.shape[0]):
            idx = res.shape[0]
            res.set_value(idx, 'Algorithm', alg)
            res.set_value(idx, 'WalkingTask', task_name)
            res.set_value(idx, 'TaskNum', training_data)
            res.set_value(idx, 'FoldNum', str(j+1))
            res.set_value(idx, 'RMSE', f_i['rmse'][j])
            if 'mape' in f_i.columns:
                res.set_value(idx, 'MAPE', f_i['mape'][j])

            # params
            p = ast.literal_eval(f_i['best'][j])
            res.set_value(idx, 'Mva_win', p['mva_win'])
            res.set_value(idx, 'Peak_min_thr', p['peak_min_thr'])
            res.set_value(idx, 'Peak_min_dist', p['peak_min_dist'])
            if "intersect_win" in p:
                res.set_value(idx, 'Intersect_win', p['intersect_win'])
            if "union_min_dist" in p:
                res.set_value(idx, 'Union_min_dist', p['union_min_dist'])
            if "union_min_thresh" in p:
                res.set_value(idx, 'Union_min_thr', p['union_min_thresh'])
            if "mva_win_combined" in p:
                res.set_value(idx, 'Mva_win_combined', p['mva_win_combined'])

    # Save
    file_name = 'Summary_search_' + c.search_space + '_alg_' + c.opt_alg + '_evals_' + str(c.max_evals) + '_folds_' + \
                str(c.n_folds) + '.csv'

    file_path = join(save_dir, file_name)
    res.to_csv(file_path, index=False)

    if return_file_path:
        return file_path