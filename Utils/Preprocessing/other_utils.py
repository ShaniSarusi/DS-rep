from Utils.DataHandling.data_processing import chunk_it
import numpy as np


def normalize_signal(sig):
    ''' Normalize a signal: '''
    y = (sig - np.min(sig))/(np.max(sig)-np.min(sig))
    return y


def split_data(samples, n_folds):
    idx_folds = chunk_it(samples, n_folds, shuffle=True)
    train = []
    test = []
    for i in range(n_folds):
        test.append(sorted([x for x in idx_folds[i]]))
        tr_idx = np.setdiff1d(samples, idx_folds[i])
        train.append(sorted([x for x in tr_idx]))
    return train, test
