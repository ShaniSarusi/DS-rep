from Utils.DataHandling.data_processing import chunk_it
import numpy as np


def normalize_signal(sig):
    ''' Normalize a signal: '''
    y = (sig - np.min(sig))/(np.max(sig)-np.min(sig))
    return y


def normalize_max_min(signals, use_all_samples_for_max_min=False):
    if use_all_samples_for_max_min:
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
