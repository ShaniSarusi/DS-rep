import pickle
import pandas as pd
import h5py
from os import listdir
import Gait.config as c
from os.path import isfile, join, isdir, split
from Utils.DataHandling.data_processing import make_df
from Utils.DataHandling import reading_and_writing_files as rd

# params and global variables
sides = [{"name": 'lhs', "sensor": "/Sensors/" + str(c.lhs_wrist_sensor) + "/"},
         {"name": 'rhs', "sensor": "/Sensors/" + str(c.rhs_wrist_sensor) + "/"}]


def pickle_metadata():
    # Make metadata data frame
    rd.pickle_excel_file(input_path=join(c.common_path, 'SampleData.xlsx'), output_name='metadata_sample',
                         output_path=c.pickle_path)
    rd.pickle_excel_file(input_path=join(c.common_path, 'SubjectData.xlsx'), output_name='metadata_subject',
                         output_path=c.pickle_path)
    rd.pickle_excel_file(input_path=join(c.common_path, 'TaskFilters.xlsx'), output_name='task_filters',
                         output_path=c.pickle_path)

    sample = pd.read_excel(join(c.common_path, 'SampleData.xlsx'))
    subject = pd.read_excel(join(c.common_path, 'SubjectData.xlsx'))
    with open(join(c.pickle_path, 'metadata_sample'), 'wb') as fp:
        pickle.dump(sample, fp)
    with open(join(c.pickle_path, 'metadata_subject'), 'wb') as fp:
        pickle.dump(subject, fp)


def read_input_files_names():
    # Get all input file paths, sorted by chronologically by subject, then by test
    input_files = []
    subjects = [f for f in listdir(c.input_path) if isdir(join(c.input_path, f))]
    subjects = sorted(subjects)
    for i in range(len(subjects)):
        subject_path = join(c.input_path, subjects[i])
        # csv_files = rd.all_files_in_directory(subject_path, "csv")
        if isdir(join(subject_path, 'rawData')):
            subject_path = join(subject_path, 'rawData')
        h5py_files = rd.all_files_in_directory(subject_path, "h5", do_sort=True)

        for j in range(len(h5py_files)):
            input_files.append(join(subject_path, h5py_files[j]))
    return input_files


def extract_sensor_data(input_files):
    p_acc = []; p_bar = []; p_gyr = []; p_mag = []; p_temp = []; p_time = []
    for i in range(len(input_files)):
        print('In file ' + str(i + 1) + ' of ' + str(len(input_files)))  # TODO - use logger
        # print('\r', 'In file ' + str(i + 1) + ' of ' + str(len(input_files)), end=" ")  # TODO - use logger
        f = h5py.File(join(c.common_path, input_files[i]), 'r')

        # read sensor data and convert to pandas format
        sensors = {"name":"Accelerometer",
                  "columns": ['x', 'y', 'z'],
                   "side": {"right":{}, "left": {}}}

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
    # add_ts (everything is synced in apdm
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


if __name__ == '__main__':
    pickle_metadata()
    acc, bar, gyr, mag, temp, time = extract_sensor_data(read_input_files_names())
    acc = add_norm(acc)
    gyr = add_norm(gyr)
    mag = add_norm(mag)
    acc, bar, gyr, mag, temp, time = fix_data_types(acc, bar, gyr, mag, temp, time)
    acc, bar, gyr, mag, temp = add_ts_to_sensor_data(acc, bar, gyr, mag, temp, time)

    metadata_truncate_start_label(time)
    acc, bar, gyr, mag, temp, time = truncate_start_sensor_data(acc, bar, gyr, mag, temp, time)
    pickle_sensor_data(acc, bar, gyr, mag, temp, time)
