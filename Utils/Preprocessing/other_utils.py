from Utils.DataHandling.data_processing import chunk_it
from Utils.Preprocessing.projections import project_gravity
import numpy as np


def normalize_signal(sig):
    ''' Normalize a signal: '''
    y = (sig - np.min(sig))/(np.max(sig)-np.min(sig))
    return y


def normalize_max_min(signals, use_single_max_min_for_all_samples=False):
    if use_single_max_min_for_all_samples:
        sig_min = np.min([x.min() for x in signals])
        sig_max = np.max([x.max() for x in signals])
        sig_range = sig_max - sig_min
        signals = [(x - sig_min) / sig_range for x in signals]
    else:
        signals = [normalize_signal(x) for x in signals]
    return signals


def split_data(samples, n_folds):
    idx_folds = chunk_it(samples, n_folds, shuffle=True)
    train = []
    test = []
    for i in range(n_folds):
        test.append(sorted([x for x in idx_folds[i]]))
        tr_idx = np.setdiff1d(samples, idx_folds[i])
        train.append(sorted([x for x in tr_idx]))
    return train, test


def reduce_dim_3_to_1(data, signal_to_use, vert_win, verbose=True):
    if verbose: print("\tStep: Selecting " + signal_to_use + " signal")
    if signal_to_use == 'norm':
        res = [data[i]['n'] for i in range(len(data))]
    if signal_to_use == 'vertical':
        if verbose and vert_win is not None: print("\tStep: Vertical projection window size is: " + str(vert_win))
        res = [project_gravity(data[i]['x'], data[i]['y'], data[i]['z'], num_samples_per_interval=vert_win,
                                return_only_vertical=True) for i in range(len(data))]
    return res
