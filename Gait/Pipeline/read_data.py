import pickle
import pandas as pd
import h5py
from os import listdir
import numpy as np
import Gait.config as c
from os.path import join, isdir
from Utils.DataHandling.data_processing import make_df
from Utils.DataHandling.reading_and_writing_files import read_all_files_in_directory, pickle_excel_file

# params and global variables
sides = [{"name": 'lhs', "sensor": "/Sensors/" + str(c.lhs_wrist_sensor) + "/"},
         {"name": 'rhs', "sensor": "/Sensors/" + str(c.rhs_wrist_sensor) + "/"}]


def pickle_metadata():
    # Make metadata data frame
    pickle_excel_file(input_path=join(c.common_path, 'SampleData.xlsx'), output_name='metadata_sample',
                      output_path=c.pickle_path)
    pickle_excel_file(input_path=join(c.common_path, 'SubjectData.xlsx'), output_name='metadata_subject',
                      output_path=c.pickle_path)
    pickle_excel_file(input_path=join(c.common_path, 'TaskFilters.xlsx'), output_name='task_filters',
                      output_path=c.pickle_path)

    sample = pd.read_excel(join(c.common_path, 'SampleData.xlsx'))
    subject = pd.read_excel(join(c.common_path, 'SubjectData.xlsx'))
    with open(join(c.pickle_path, 'metadata_sample'), 'wb') as fp:
        pickle.dump(sample, fp)
    with open(join(c.pickle_path, 'metadata_subject'), 'wb') as fp:
        pickle.dump(subject, fp)


def read_apdm_result_files():
    subjects = [f for f in listdir(c.input_path) if isdir(join(c.input_path, f))]
    subjects = sorted(subjects)

    apdm_files = []
    for i in range(len(subjects)):
        subject_path = join(c.input_path, subjects[i])
        files = read_all_files_in_directory(subject_path, prefix='2017', do_sort=True)
        for j in range(len(files)):
            apdm_files.append(join(subject_path, files[j]))
    return apdm_files


def extract_apdm_results(p_apdm_files, input_files, p_len_raw_data):
    num_files = len(p_apdm_files)
    if num_files == 0:
        return
    assert num_files == p_len_raw_data, "Number of apdm files is not equal to length of raw data input"

    # Set measures DataFrame
    cols_measures = ['cadence', 'asymmetry_step_time', 'cv_stride_time_lhs', 'cv_stride_time_rhs', 'cv_step_time_lhs',
                     'cv_step_time_lhs']
    measures = pd.DataFrame(index=range(num_files), columns=cols_measures)
    for i in range(num_files):
        if "_TUG_" in p_apdm_files[i]:
            continue
        measures_i = pd.read_csv(p_apdm_files[i], skiprows=np.arange(9))
        measures_i = measures_i.set_index(measures_i['Measure'])

        cadence = float(measures_i.loc['Gait - Lower Limb - Cadence L (steps/min)']['Mean'] +
                        measures_i.loc['Gait - Lower Limb - Cadence R (steps/min)']['Mean']) / 2.0

        step_l_m = measures_i.loc['Gait - Lower Limb - Step Duration L (s)']['Mean']
        step_l_std = measures_i.loc['Gait - Lower Limb - Step Duration L (s)']['StDev']
        step_r_m = measures_i.loc['Gait - Lower Limb - Step Duration R (s)']['Mean']
        step_r_std = measures_i.loc['Gait - Lower Limb - Step Duration R (s)']['StDev']

        stride_l_m = measures_i.loc['Gait - Lower Limb - Gait Cycle Duration L (s)']['Mean']
        stride_l_std = measures_i.loc['Gait - Lower Limb - Gait Cycle Duration L (s)']['StDev']
        stride_r_m = measures_i.loc['Gait - Lower Limb - Gait Cycle Duration R (s)']['Mean']
        stride_r_std = measures_i.loc['Gait - Lower Limb - Gait Cycle Duration R (s)']['StDev']

        asymmetry_step_time = 100.0*(np.abs(step_l_m - step_r_m) / np.mean([step_l_m, step_r_m]))
        cv_stride_time_lhs = round(stride_l_std / stride_l_m, 3)
        cv_stride_time_rhs = round(stride_r_std / stride_r_m, 3)
        cv_step_time_lhs = round(step_l_std / step_l_m, 3)
        cv_step_time_rhs = round(step_r_std / step_r_m, 3)

        row = [cadence, asymmetry_step_time, cv_stride_time_lhs, cv_stride_time_rhs, cv_step_time_lhs, cv_step_time_rhs]
        measures.iloc[i] = row

    # Set events DataFrame
    # get time
    p_time = []
    for i in range(len(input_files)):
        print('In file ' + str(i + 1) + ' of ' + str(len(input_files)))
        f = h5py.File(join(c.common_path, input_files[i]), 'r')
        time_i = {}
        for side in sides:
            time_i[side['name']] = make_df(f[side['sensor'] + 'Time'], ['value'])
        p_time.append(time_i)

    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
        sample = pickle.load(fp)
    f_events = pd.read_csv(p_apdm_files[0], skiprows=np.arange(66), header=None)
    f_events = f_events.drop(f_events.index[len(f_events) - 1])  # drop last column
    cols_events = f_events.iloc[:, 0]
    events = pd.DataFrame(index=range(num_files), columns=cols_events)
    first_event_idx = 5
    for i in range(num_files):
        if "_TUG_" in p_apdm_files[i]:
            continue
        events_i = pd.read_csv(p_apdm_files[i], skiprows=np.arange(66), header=None)
        row = events_i.iloc[:, first_event_idx:].values
        # Truncate
        trunc = (p_time[i]['rhs'].iloc[sample.iloc[i]['CropStartIndex']] - p_time[i]['rhs'].iloc[0])/1e6
        row = row - trunc[0]
        # Store in DataFrame
        row = row.tolist()
        events.iloc[i] = row[:-1]

    return measures, events


def read_input_files_names():
    # Get all input file paths, sorted by chronologically by subject, then by test
    subjects = [f for f in listdir(c.input_path) if isdir(join(c.input_path, f))]
    subjects = sorted(subjects)

    input_files = []
    for i in range(len(subjects)):
        subject_path = join(c.input_path, subjects[i])
        if isdir(join(subject_path, 'rawData')):
            subject_path = join(subject_path, 'rawData')
        h5py_files = read_all_files_in_directory(subject_path, "h5", do_sort=True)

        for j in range(len(h5py_files)):
            input_files.append(join(subject_path, h5py_files[j]))
    return input_files


def extract_sensor_data(input_files):
    p_acc = []; p_bar = []; p_gyr = []; p_mag = []; p_temp = []; p_time = []
    for i in range(len(input_files)):
        print('In file ' + str(i + 1) + ' of ' + str(len(input_files)))  # TODO - consider using logger
        f = h5py.File(join(c.common_path, input_files[i]), 'r')

        acc_i = {}; bar_i = {}; gyr_i = {}; mag_i = {}; temp_i = {}; time_i = {}
        for side in sides:
            acc_i[side['name']] = make_df(f[side['sensor'] + 'Accelerometer'], ['x', 'y', 'z'])
            bar_i[side['name']] = make_df(f[side['sensor'] + 'Barometer'], ['value'])
            gyr_i[side['name']] = make_df(f[side['sensor'] + 'Gyroscope'], ['x', 'y', 'z'])
            mag_i[side['name']] = make_df(f[side['sensor'] + 'Magnetometer'], ['x', 'y', 'z'])
            temp_i[side['name']] = make_df(f[side['sensor'] + 'Temperature'], ['value'])
            time_i[side['name']] = make_df(f[side['sensor'] + 'Time'], ['value'])

        # append current file data
        p_acc.append(acc_i); p_bar.append(bar_i); p_gyr.append(gyr_i); p_mag.append(mag_i); p_temp.append(temp_i)
        p_time.append(time_i)
    return p_acc, p_bar, p_gyr, p_mag, p_temp, p_time


def add_norm(data):
    for i in range(len(data)):
        for side in sides:
            data[i][side['name']]['n'] = (data[i][side['name']]['x'] ** 2 + data[i][side['name']]['y'] ** 2 +
                                          data[i][side['name']]['z'] ** 2) ** 0.5
    return data


def fix_data_types(p_acc, p_bar, p_gyr, p_mag, p_temp, p_time):
    for i in range(len(p_acc)):
        for side in sides:
            p_acc[i][side['name']] = p_acc[i][side['name']].astype(dtype=float)
            p_bar[i][side['name']] = p_bar[i][side['name']].astype(dtype=float)
            p_gyr[i][side['name']] = p_gyr[i][side['name']].astype(dtype=float)
            p_mag[i][side['name']] = p_mag[i][side['name']].astype(dtype=float)
            p_temp[i][side['name']] = p_temp[i][side['name']].astype(dtype=float)
            p_time[i][side['name']]['value'] = pd.to_datetime(p_time[i][side['name']]['value'], unit='us')
    return p_acc, p_bar, p_gyr, p_mag, p_temp, p_time


def add_ts_to_sensor_data(p_acc, p_bar, p_gyr, p_mag, p_temp, p_time):
    # add_ts (everything is synced in apdm)
    for i in range(len(p_acc)):
        for side in sides:
            p_acc[i][side['name']]['ts'] = p_time[i][side['name']]['value']
            p_bar[i][side['name']]['ts'] = p_time[i][side['name']]['value']
            p_gyr[i][side['name']]['ts'] = p_time[i][side['name']]['value']
            p_mag[i][side['name']]['ts'] = p_time[i][side['name']]['value']
            p_temp[i][side['name']]['ts'] = p_time[i][side['name']]['value']
    return p_acc, p_bar, p_gyr, p_mag, p_temp


def metadata_truncate_start_label(p_time, distance=8):
    # read data
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp: sample = pickle.load(fp)
    for i in sample['SampleId']:
        val = (p_time[i]['rhs'].iloc[-1] - p_time[i]['rhs'].iloc[sample.iloc[i]['CropStartIndex']])[0].total_seconds()
        sample.set_value(i, 'DurationWithCrop', val)
        sample.set_value(i, 'CadenceWithCrop', sample.iloc[i]['StepCount']*60/sample.iloc[i]['DurationWithCrop'])
        if pd.notnull(sample['Speed'][i]):
            sample.set_value(i, 'SpeedWithCrop', distance/sample.iloc[i]['DurationWithCrop'])
    with open(join(c.pickle_path, 'metadata_sample'), 'wb') as fp: pickle.dump(sample, fp)


def truncate_start_sensor_data(p_acc, p_bar, p_gyr, p_mag, p_temp, p_time):
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp: sample = pickle.load(fp)
    for i in sample['SampleId']:
        for side in sides:
            idx = sample['CropStartIndex'].iloc[i]
            p_acc[i][side['name']] = p_acc[i][side['name']].truncate(idx).reset_index()
            p_bar[i][side['name']] = p_bar[i][side['name']].truncate(idx).reset_index()
            p_gyr[i][side['name']] = p_gyr[i][side['name']].truncate(idx).reset_index()
            p_mag[i][side['name']] = p_mag[i][side['name']].truncate(idx).reset_index()
            p_temp[i][side['name']] = p_temp[i][side['name']].truncate(idx).reset_index()
            p_time[i][side['name']] = p_time[i][side['name']].truncate(idx).reset_index()
    return p_acc, p_bar, p_gyr, p_mag, p_temp, p_time


def pickle_sensor_data(p_acc, p_bar, p_gyr, p_mag, p_temp, p_time):
    with open(join(c.pickle_path, 'acc'), 'wb') as fp: pickle.dump(p_acc, fp)
    with open(join(c.pickle_path, 'bar'), 'wb') as fp: pickle.dump(p_bar, fp)
    with open(join(c.pickle_path, 'gyr'), 'wb') as fp: pickle.dump(p_gyr, fp)
    with open(join(c.pickle_path, 'mag'), 'wb') as fp: pickle.dump(p_mag, fp)
    with open(join(c.pickle_path, 'temp'), 'wb') as fp: pickle.dump(p_temp, fp)
    with open(join(c.pickle_path, 'time'), 'wb') as fp: pickle.dump(p_time, fp)


def load_sensor_data():
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp: p_sample = pickle.load(fp)
    with open(join(c.pickle_path, 'time'), 'rb') as fp: p_time = pickle.load(fp)
    with open(join(c.pickle_path, 'acc'), 'rb') as fp: p_acc = pickle.load(fp)
    with open(join(c.pickle_path, 'bar'), 'rb') as fp: p_bar = pickle.load(fp)
    with open(join(c.pickle_path, 'gyr'), 'rb') as fp: p_gyr = pickle.load(fp)
    with open(join(c.pickle_path, 'mag'), 'rb') as fp: p_mag = pickle.load(fp)
    with open(join(c.pickle_path, 'temp'), 'rb') as fp: p_temp = pickle.load(fp)
    return p_acc, p_bar, p_gyr, p_mag, p_temp, p_time, p_sample


def create_result_matrix():
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp: sample = pickle.load(fp)
    with open(join(c.pickle_path, 'features_steps'), 'rb') as fp: step_features = pickle.load(fp)
    # with open(join(common_input, 'features_armswing'), 'rb') as fp: arm_swing_features = pickle.load(fp)

    # connect DataFrame
    step_features = step_features.drop('step_durations', axis=1)

    # df_results = pd.concat([step_features, arm_swing_features], axis=1)
    df_results = step_features
    df_sample_and_results = pd.concat([sample, df_results], axis=1)

    # save csvs
    df_sample_and_results.to_csv(join(c.results_path, 'Results.csv'))


if __name__ == '__main__':
    # Store metadata
    pickle_metadata()

    # Read file names
    raw_data_input_file_names = read_input_files_names()
    apdm_files = read_apdm_result_files()

    # Read and process raw signal data
    acc, bar, gyr, mag, temp, time = extract_sensor_data(raw_data_input_file_names)
    acc = add_norm(acc)
    gyr = add_norm(gyr)
    mag = add_norm(mag)
    acc, bar, gyr, mag, temp, time = fix_data_types(acc, bar, gyr, mag, temp, time)
    acc, bar, gyr, mag, temp = add_ts_to_sensor_data(acc, bar, gyr, mag, temp, time)

    # Truncate signal data
    metadata_truncate_start_label(time)
    acc, bar, gyr, mag, temp, time = truncate_start_sensor_data(acc, bar, gyr, mag, temp, time)
    pickle_sensor_data(acc, bar, gyr, mag, temp, time)

    # Read, process, and store apdm data
    apdm_measures, apdm_events = extract_apdm_results(apdm_files, p_len_raw_data=len(acc))
    with open(join(c.pickle_path, 'apdm_measures'), 'wb') as fp: pickle.dump(apdm_measures, fp)
    with open(join(c.pickle_path, 'apdm_events'), 'wb') as fp: pickle.dump(apdm_events, fp)
