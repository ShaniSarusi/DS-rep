import numpy as np
import pandas as pd
from os.path import join


def read_data_windows(data_path, read_also_home_data=False, sample_freq=50, window_size=5):
    samples_in_window = sample_freq * window_size

    lab_x = pd.read_csv(join(data_path, 'lab_x.csv'), header=None)
    lab_x = lab_x.as_matrix()
    lab_x = lab_x[1:, range(1, samples_in_window+1)]

    lab_y = pd.read_csv(join(data_path, 'lab_y.csv'), header=None)
    lab_y = lab_y.as_matrix()
    lab_y = lab_y[1:, range(1, samples_in_window+1)]

    lab_z = pd.read_csv(join(data_path, 'lab_z.csv'), header=None)
    lab_z = lab_z.as_matrix()
    lab_z = lab_z[1:, range(1, samples_in_window+1)]

    if read_also_home_data is True:
        home_x = pd.read_csv(join(data_path, 'home_x.csv'), header=None)
        home_x = home_x.as_matrix()
        home_x = np.array(home_x, dtype='object')

        home_y = pd.read_csv(join(data_path, 'home_y.csv'), header=None)
        home_y = home_y.as_matrix()
        home_y = np.array(home_y, dtype='object')

        home_z = pd.read_csv(join(data_path, 'home_z.csv'), header=None)
        home_z = home_z.as_matrix()
        home_z = np.array(home_z, dtype='object')

        return home_x, home_y, home_z, lab_x, lab_y, lab_z

    else:
        return lab_x, lab_y, lab_z


def read_tag_data(data_path):
    tag_data = pd.read_csv(join(data_path, 'metadata.csv'), header='infer')
    return tag_data
