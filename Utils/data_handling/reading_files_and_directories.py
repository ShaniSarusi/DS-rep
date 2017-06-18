# useful functions for reading data from files and directories
from os.path import isfile, join, isdir
from os import listdir


def all_files_in_directory(dir_path, file_type=None, do_sort=False):
    if isdir(dir_path) is False:
        return
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    if file_type is not None:
        files = [f for f in files if f.endswith("."+file_type)]
    if do_sort:
        files = sorted(files)
    return files
