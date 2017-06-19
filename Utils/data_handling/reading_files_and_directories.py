# useful functions for reading data from files and directories
from os.path import isfile, join, isdir, split
from os import listdir
import pandas as pd
import pickle


def all_files_in_directory(dir_path, file_type=None, do_sort=False):
    if isdir(dir_path) is False:
        return
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    if file_type is not None:
        files = [f for f in files if f.endswith("."+file_type)]
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