import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


class TwoAcc:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        self.ft = pd.DataFrame()
        self.time_offset = None
        self.shift_by = None

    def add_norm(self, ptype='both'):
        if (ptype == 'left') or (ptype == 'both'):
            self.lhs['n'] = (self.lhs['x'] ** 2 + self.lhs['y'] ** 2 + self.lhs['z'] ** 2) ** 0.5
        if (ptype == 'right') or (ptype == 'both'):
            self.rhs['n'] = (self.rhs['x'] ** 2 + self.rhs['y'] ** 2 + self.rhs['z'] ** 2) ** 0.5

    # Synchronize signal
    def calc_offset(self, lhs_shake_range, rhs_shake_range):
        a = self.lhs.loc[lhs_shake_range, 'n'].as_matrix()
        b = self.rhs.loc[rhs_shake_range, 'n'].as_matrix()
        offset = np.argmax(signal.correlate(a, b))
        self.shift_by = a.shape[0] - offset
        self.time_offset = self.rhs.loc[rhs_shake_range[0] + self.shift_by, 'ts'] - self.lhs.loc[lhs_shake_range[0],
                                                                                                 'ts']
        # The link below says you need to subtract 1 from shift_by, but based on my plotting it looks like you don't
        # need to
        # link: http://stackoverflow.com/questions/4688715/find-time-shift-between-two-similar-waveforms

    def plot_shift(self, lhs_range, show_shift=True):
        a = self.lhs.loc[lhs_range, 'n'].as_matrix()
        b_idx = self.get_closest_index(self.lhs, value=self.lhs.loc[lhs_range[0], 'ts'])
        rhs_range = range(b_idx, b_idx + len(lhs_range))
        b = self.rhs.loc[rhs_range, 'n'].as_matrix()

        plt.plot(a)
        if show_shift:
            plt.plot(b)
        else:
            plt.plot(b[range(b_idx + self.shift_by, b_idx + len(lhs_range) + self.shift_by)])

    @staticmethod
    def get_closest_index(data, value):
        # below
        tmp = (data <= value)
        ind = None
        if np.any(tmp): #there is at least one True
            ind = tmp[tmp == True].index[-1]
        else:
            tmp = (data > value)
            ind = tmp[tmp == True].index[0]
        return ind
