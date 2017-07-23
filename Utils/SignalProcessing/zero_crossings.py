import numpy as np

def zero_crossings(input_signal):
    # returns indices of zero crossings, and area between each consecutive zero crossings

    # Find zero crossings
    crossings_idx = np.where(np.diff(np.signbit(input_signal)))[0] + 1
    if len(crossings_idx) == 0:
        return None, None

    # Remove crossing at point zero if it exists. Also, no crossing should exist at the last point if it is zero.
    if crossings_idx[0] == 1:
        crossings_idx = np.delete(crossings_idx, 0)

    idx = np.unique(np.append(0, crossings_idx, len(input_signal)))
    area = []
    for i in range(len(idx) - 1):
        st = idx[i]
        en = idx[i+1]
        area.append(np.sum(input_signal[st:en]))
    return crossings_idx, np.asarray(area)