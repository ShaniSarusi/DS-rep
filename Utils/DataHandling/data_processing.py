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


def string_to_int_list(input_string, replace_dot_zero=True):
    """
    Convert the input (a string) into a list.

    Input:
        str (String): The string input. It should generally be a list in string format, such as '[0, 33, 22, 8]'

    Output:
        z (list): The input string converted into a list, with newlines, '.0', and brackets handled
    """

    # Check if input is string. If not return the input
    if not isinstance(input_string, str):
        return input_string

    # Convert string to list
    z = input_string.replace('\n', '')
    z = z.replace('[', '')
    z = z.replace(']', '')
    z = z.replace(' ', ',')
    if replace_dot_zero:
        z = z.replace('.0', '')
    z = z.split(sep=',')
    z = list(filter(None, z))
    z = [int(i) for i in z]
    return z


def string_to_float_list(input_string):
    """
    Convert the input (a string) into a float list.

    Input:
        str (String): The string input. It should generally be a list in string format, such as '[0.2, 33.1, 22, 8.55]'

    Output:
        z (list): The input string converted into a list, with newlines and brackets handled
    """

    # Check if input is string. If not return the input
    if not isinstance(input_string, str):
        return input_string

    # Convert string to list
    z = input_string.replace('\n', '')
    z = z.replace('[', '')
    z = z.replace(']', '')
    z = z.replace(' ', ',')
    z = z.split(sep=',')
    z = list(filter(None, z))
    z = [float(i) for i in z]
    return z


def multi_intersect(input_tuple):
    """
    Return the intersection of the all the content within the input tuple

    Input:
        input_tuple (A tuple of lists or arrays): The input tuple

    Output:
        out1 (A single list or numpy array): The intersection of all the values in the tuple
    """
    return reduce(np.intersect1d, input_tuple)


def make_df_from_hdf5_dataset(hdf5_dataset, col_names):
    """
    Take the input data and column names and create a dataframe.

    Input:
        hdf5_dataset (hdf5 dataset): Dataset read from hdf5 file in hdf5 format
        col_names (list of strings): column names in the dataframe

    Output:
        out1 (Pandas DataFrame): Returns the input data and column names as a dataframe
    """

    data = [hdf5_dataset[i] for i in range(hdf5_dataset.shape[0])]
    return pd.DataFrame(data, columns=col_names)


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
