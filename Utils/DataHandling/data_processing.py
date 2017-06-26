import numpy as np
import pandas as pd
from functools import reduce


def pd_to_np(arr):
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


def chunk_it(seq, num, shuffle=False):
    if shuffle:
        np.random.shuffle(seq)
    avg = len(seq) / float(num)
    # print(avg)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out
