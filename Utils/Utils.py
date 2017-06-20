import pandas as pd
import numpy as np
from functools import reduce


# read export tool data
def read_export_tool_csv(csv_path):
    df = pd.read_csv(csv_path, skiprows=[0], header=None)
    df.user = df.iloc[1][0]
    df.columns = ['a', 'b', 'ts', 'c', 'x', 'y', 'z']
    df.drop(['a', 'b', 'c'], axis=1, inplace=True)
    df['x'] = df['x'].str.replace('{x=', '')
    df['y'] = df['y'].str.replace('y=', '')
    df['z'] = df['z'].str.replace('z=', '')
    df['z'] = df['z'].str.replace('}', '')
    #
    # set types
    df['x'] = df['x'].astype('float')
    df['y'] = df['y'].astype('float')
    df['z'] = df['z'].astype('float')
    df['ts'] = pd.to_datetime(df.loc[:, 'ts'])
    #
    return df


def make_df(data, col_names):
    df = pd.DataFrame(index=range(data.shape[0]), columns=col_names)
    for i in range(data.shape[0]):
        df.iloc[i] = data[i]
    return df


def pd_to_np(arr):
    if isinstance(arr, pd.Series):
        return np.asarray(arr.astype(float))
    else:
        return arr


def multi_intersect(input_tuple):
    return reduce(np.intersect1d, input_tuple)


def mean_and_std(values, round_by=2):
    values = pd_to_np(values)
    # return str(round(np.mean(values), round_by)) + u" \u00B1 " + str(round(np.std(values), round_by))
    return str(round(np.mean(values), round_by)) + ' +- ' + str(round(np.std(values), round_by))


def cv(values, round_by=3):
    values = pd_to_np(values)
    coefficient_of_var = np.std(values) / np.mean(values)
    return str(round(coefficient_of_var, round_by))
