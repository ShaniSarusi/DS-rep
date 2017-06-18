import pickle
from os.path import join

import Gait.Utils.preprocessing as pre
import pandas as pd

import Gait.Utils.algorithms as alg
import Gait.config as config


def extract_generic_features():
    # load metadata and sensor data
    with open(join(config.pickle_path, 'metadata_sample'), 'rb') as fp: sample = pickle.load(fp)
    with open(join(config.pickle_path, 'acc'), 'rb') as fp: acc = pickle.load(fp)
    with open(join(config.pickle_path, 'gyr'), 'rb') as fp: gyr = pickle.load(fp)

    # pre-processing - truncate
    fr_pct = 10
    bk_pct = 10
    acc = pre.truncate(acc, fr_pct, bk_pct)
    gyr = pre.truncate(gyr, fr_pct, bk_pct)

    acc = pre.butter_filter_lowpass(acc, order=10, sampling_rate=128, freq=15)
    gyr = pre.butter_filter_lowpass(gyr, order=10, sampling_rate=128, freq=15)

    # calculate features
    ft = pd.DataFrame(index=range(sample.shape[0]))
    # single side accelerometer  # TODO which features need absolute value? Where does gravity need to be removed?
    ft = alg.add_feature(ft, sensor=acc, sensor_name='acc', axes=['x', 'y', 'z', 'n'], sides=['lhs', 'rhs'], what='mean')
    ft = alg.add_feature(ft, sensor=acc, sensor_name='acc', axes=['x', 'y', 'z', 'n'], sides=['lhs', 'rhs'], what='median')
    ft = alg.add_feature(ft, sensor=acc, sensor_name='acc', axes=['x', 'y', 'z', 'n'], sides=['lhs', 'rhs'], what='std')
    # both sides accelerometer
    ft = alg.add_feature(ft, sensor=acc, sensor_name='acc', axes=['x', 'y', 'z', 'n'], sides=['both'], what='mean_diff')
    ft = alg.add_feature(ft, sensor=acc, sensor_name='acc', axes=['x', 'y', 'z', 'n'], sides=['both'], what='median_diff')
    ft = alg.add_feature(ft, sensor=acc, sensor_name='acc', axes=['x', 'y', 'z', 'n'], sides=['both'], what='std_diff')
    ft = alg.add_feature(ft, sensor=acc, sensor_name='acc', axes=['x', 'y', 'z', 'n'], sides=['both'], what='cross_corr')

    # single side gyro
    ft = alg.add_feature(ft, sensor=gyr, sensor_name='gyr', axes=['x', 'y', 'z', 'n'], sides=['lhs', 'rhs'], what='mean')
    ft = alg.add_feature(ft, sensor=gyr, sensor_name='gyr', axes=['x', 'y', 'z', 'n'], sides=['lhs', 'rhs'], what='median')
    ft = alg.add_feature(ft, sensor=gyr, sensor_name='gyr', axes=['x', 'y', 'z', 'n'], sides=['lhs', 'rhs'], what='std')
    # both sides gyro
    ft = alg.add_feature(ft, sensor=gyr, sensor_name='gyr', axes=['x', 'y', 'z', 'n'], sides=['both'], what='mean_diff')
    ft = alg.add_feature(ft, sensor=gyr, sensor_name='gyr', axes=['x', 'y', 'z', 'n'], sides=['both'], what='median_diff')
    ft = alg.add_feature(ft, sensor=gyr, sensor_name='gyr', axes=['x', 'y', 'z', 'n'], sides=['both'], what='std_diff')
    ft = alg.add_feature(ft, sensor=gyr, sensor_name='gyr', axes=['x', 'y', 'z', 'n'], sides=['both'], what='cross_corr')

    # save features
    with open(join(config.pickle_path, 'features_generic'), 'wb') as fp:
        pickle.dump(ft, fp)

if __name__ == '__main__':
    extract_generic_features()


