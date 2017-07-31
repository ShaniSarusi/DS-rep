"""
This module contains functions that assist with various data processing and handling needs. For example, spitting a
list into equal size chunks.
"""

import numpy as np
import pandas as pd
from functools import reduce


def pd_to_np(arr):
    """
    Convert the input (either Pandas Series or numpy array) into a numpy array.

    Input:
    arr (Pandas series or numpy array): The input to be converted if necessary

    Output:
    out1 (numpy array): The output converted into a numpy array
    """
    if isinstance(arr, pd.Series):
        return np.asarray(arr.astype(float))
    else:
        return arr


def multi_intersect(input_tuple):
    """
    Return the intersection of the all the content within the input tuple

    Input:
    input_tuple (A tuple of lists or arrays): The input tuple

    Output:
    out1 (A single list or numpy array): The intersection of all the values in the tuple
    """
    return reduce(np.intersect1d, input_tuple)


# TODO see if the below function can be replaced in the code by something more efficient. Essentially a single line pandas line
def make_df(data, col_names):
    """
    Take the input data and column names and create a dataframe. This function is likely unnecessary and inefficient

    :param data (list of lists, maybe matrix...): Data likely in list format
    :param col_names (list of strings): column names in the dataframe
    :return: A Pandas DataFrame
    """
    df = pd.DataFrame(index=range(data.shape[0]), columns=col_names)
    for i in range(data.shape[0]):
        df.iloc[i] = data[i]
    return df


def chunk_it(seq, n, shuffle=False):
    """
    Accept an array and return a list of the array broken into chunks.

    Input:
    seq (array or list): The input array
    n (int): Number of chunks to break the array into
    shuffle (boolean): default false. If true, the sequence is randomly shuffled before breaking into chunks.

    Output:
    out1 (list of arrays): List of n equal size arrays
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
