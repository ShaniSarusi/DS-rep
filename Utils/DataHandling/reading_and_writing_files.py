"""
This module contains various functions for reading data from files and directories
"""

from os.path import isfile, join, isdir, split
from os import listdir
import pandas as pd
import pickle


def read_all_files_in_directory(dir_path, file_type=None, prefix=None, do_sort=False):
    """
    Take a directory and return a list of all files in the directory. Optionally all files of a specific type.

    Input:
        dir_path (string): The path of the directory
        file_type (string): If specified (default None), only file names of this type will be read. Example: 'csv'
        prefix (string): If specified (default None), only file names that contain this prefix will be read.
        do_sort (boolean): If true (default False), the input files will be sorted.

    Output:
        out1 (list of strings): Each string is a file in the input directory
    """
    if isdir(dir_path) is False:
        return
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    if file_type is not None:
        files = [f for f in files if f.endswith("."+file_type)]
    if prefix is not None:
        files = [f for f in files if f.startswith(prefix)]
    if do_sort:
        files = sorted(files)
    return files


def pickle_excel_file(input_path, output_name=None, output_path=None):
    if isfile(input_path) is False:
        return
    file = pd.read_excel(input_path)
    head, tail = split(input_path)
    if output_path is None:
        output_path = head
    if output_name is None:
        output_name = tail
    with open(join(output_path, output_name), 'wb') as fp:
        pickle.dump(file, fp)


def read_export_tool_csv(csv_path):
    df = pd.read_csv(csv_path, skiprows=[0], header=None)
    df.user = df.iloc[1][0]
    df.columns = ['a', 'b', 'ts', 'c', 'x', 'y', 'z']
    df.drop(['a', 'b', 'c'], axis=1, inplace=True)
    df['x'] = df['x'].str.replace('{x=', '')
    df['y'] = df['y'].str.replace('y=', '')
    df['z'] = df['z'].str.replace('z=', '')
    df['z'] = df['z'].str.replace('}', '')

    # set types
    df['x'] = df['x'].astype('float')
    df['y'] = df['y'].astype('float')
    df['z'] = df['z'].astype('float')
    df['ts'] = pd.to_datetime(df.loc[:, 'ts'])

    return df
