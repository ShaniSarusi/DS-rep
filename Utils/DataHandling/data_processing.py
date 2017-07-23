import numpy as np
import pandas as pd
from functools import reduce


def pd_to_np(arr):
    """
    Converts an input (either Pandas Series or numpy array) into a numpy array.

    :param arr: Pandas series or numpy array
    :return: numpy array
    """
    if isinstance(arr, pd.Series):
        return np.asarray(arr.astype(float))
    else:
        return arr


def multi_intersect(input_tuple):
    return reduce(np.intersect1d, input_tuple)


def make_df(data, col_names):
    df = pd.DataFrame(index=range(data.shape[0]), columns=col_names)
    for i in range(data.shape[0]):
        df.iloc[i] = data[i]
    return df


def chunk_it(seq, n, shuffle=False):
    """
    Accepts a array and returns a list of the array broken into chunks.

    :param seq: The input array
    :param n: Number of chunks to break the array into
    :param shuffle: boolean (default false). If true, the sequence is randomly shuffled before breaking into chunks.
    :return: List of n equal size arrays
    """

    if shuffle:
        np.random.shuffle(seq)
    avg = len(seq) / float(n)
    # print(avg)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out
