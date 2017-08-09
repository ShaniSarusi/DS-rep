import copy
import pickle
from os.path import join

import numpy as np
import pandas as pd

import Gait.Resources.config as c
from Gait.ParameterOptimization.compare_to_apdm import compare_to_apdm

def split_by_person():
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
        sample = pickle.load(fp)
    # split sample[perons]
    names = pd.unique(sample['Person'])
    filt = dict()
    for name in names:
        filt[name] = sample[sample['Person'] == name]['SampleId'].as_matrix()
    return filt


def gait_measure_analysis(df, p_dir, p_save_name, p_algs, p_metrics, prefix=""):
    res_gait = df[0]
    for i in range(1, len(df)):
        right = df[i]
        cols_left = res_gait.columns.tolist()
        cols_right = right.columns.tolist()
        cols_shared = list(set(cols_left).intersection(cols_right))
        right = right.drop(cols_shared, axis=1)
        res_gait = res_gait.join(right, how='outer')
    res_gait = res_gait.sort_index()
    gait_measure_path = join(p_dir, p_save_name)
    res_gait.to_csv(gait_measure_path)

    # Plot gait metric comparisons to APDM
    compare_to_apdm(gait_measure_path, p_algs, p_metrics, name_prefix=prefix)


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


# def typefilter(data, filt_data, string):
#     filt = (filt_data == string).as_matrix()
#     a = [i for (i, v) in zip(data, filt) if v]
#     b = [i for (i, v) in zip(data, ~filt) if v]
#     return a, b
