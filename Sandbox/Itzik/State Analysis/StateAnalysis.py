import pandas as pd
import numpy as np
from scipy import optimize
import random
import math as m
from datetime import timedelta


class AutomaticStateAnalysis:
    def __init__(self, data, std_thresh=0.001, static_win_len=None, static_win_duration=None):
        self.data = data
        self.std_thresh = std_thresh
        self.static_win_len = static_win_len
        self.static_win_duration = static_win_duration
        self.windows = None
        self.static_windows = None
        self.xyz_data = None
        self.calib_matrix = None

    def normalize(self, pfactor=1):
        if pfactor == 1:
            return
        self.data['x'] = self.data['x'] / pfactor
        self.data['y'] = self.data['y'] / pfactor
        self.data['z'] = self.data['z'] / pfactor

    def calc_windows(self, verbose=False):
        # WINDOWS ARE IN TERMS OF 1-MINUTE RESOLUTION
        # find first and last window start times, and number of windows
        tmp = self.data['ts'].iloc[0]
        first_win = tmp + timedelta(minutes=1) - timedelta(seconds=tmp.second, microseconds=tmp.microsecond)
        tmp = self.data['ts'].iloc[-1]
        last_win = tmp - timedelta(minutes=1, seconds=tmp.second, microseconds=tmp.microsecond)
        diff = last_win - first_win
        tot_win = int(round(diff.total_seconds())/60)

        # get num samples per window, start and end indices
        n_win = 0
        st = np.empty(tot_win)
        en = np.empty(tot_win)
        n_samples = np.empty(tot_win)
        i = 0
        while (i < self.data.shape[0]) & (n_win < tot_win):
            t = self.data['ts'].iloc[i]
            win_start = first_win + n_win * timedelta(minutes=1)
            # check if data point is not in new window
            if t < win_start:
                i += 1
                continue

            # check if the new window has data points or is empty. If empty, continue until non empty window
            win_end = []
            for n_win in range(n_win, tot_win):
                win_end = first_win + (n_win+1)*timedelta(minutes=1)
                if t >= win_end:  # no data points
                    n_samples[n_win] = 0
                else:
                    break

            # get start and end indices
            st[n_win] = i
            for i in range(i, self.data.shape[0]-1):
                next_t = self.data['ts'].iloc[i+1]
                if next_t >= win_end:
                    en[n_win] = i
                    n_samples[n_win] = en[n_win] - st[n_win] + 1
                    break
            if verbose:
                print('Windows Finished: ' + str(n_win*100/tot_win) + ' percent')
            n_win += 1
            i += 1
        print('Finished windows')
        st = st.astype(int)
        en = en.astype(int)
        n_samples = n_samples.astype(int)

        # set window data frame
        cols = ['SampleId', 'TSstart', 'Duration', 'NumSamples', 'Xstd', 'Ystd', 'Zstd', 'Nstd', 'Xmean', 'Ymean', 'Zmean', 'Nmean']
        self.windows = pd.DataFrame(index=range(tot_win), columns=cols)
        self.windows['SampleId'] = range(1, 1+ tot_win)
        self.windows['Duration'] = 1
        self.windows['NumSamples'] = n_samples

        for i in range(tot_win):
            self.windows.loc[i, 'TSstart'] = first_win + i*timedelta(minutes=1)
            if self.windows.loc[i, 'NumSamples'] == 0:
                continue
            ran = range(st[i], en[i] + 1)
            norms = (self.data['x'][ran]**2 + self.data['y'][ran]**2 + self.data['z'][ran]**2)**0.5
            self.windows.loc[i, 'Xstd'] = self.data['x'].iloc[ran].std()
            self.windows.loc[i, 'Ystd'] = self.data['y'].iloc[ran].std()
            self.windows.loc[i, 'Zstd'] = self.data['z'].iloc[ran].std()
            self.windows.loc[i, 'Nstd'] = norms.std()
            self.windows.loc[i, 'Xmean'] = self.data['x'].iloc[ran].mean()
            self.windows.loc[i, 'Ymean'] = self.data['y'].iloc[ran].mean()
            self.windows.loc[i, 'Zmean'] = self.data['z'].iloc[ran].mean()
            self.windows.loc[i, 'Nmean'] = norms.mean()

    def find_window_stds_by_duration(self, min_samples, search_intervals=500):
        # check if there is data
        last_data_index = self.data.shape[0]-1
        total_dur = self.data.loc[last_data_index]['ts'] - self.data.loc[0]['ts']
        if total_dur < self.static_win_duration:
            return

        self.windows = pd.DataFrame(columns=['Df_data_index', 'TSstart', 'Duration', 'NumSamples', 'Xstd', 'Ystd', 'Zstd', 'Xmean', 'Ymean', 'Zmean'])
        idx_first = 0
        n_win = 0
        while idx_first < last_data_index:
            ts_first = self.data.loc[idx_first]['ts']
            # get last index (by iterating over large chunks at a time
            max_time = ts_first + self.static_win_duration
            idx_last = None
            tmp_first = idx_first
            while idx_last is None:
                tmp_max_last = tmp_first + search_intervals
                if tmp_max_last > last_data_index:
                    tmp_max_last = last_data_index

                tmp = (self.data.loc[tmp_first:tmp_max_last]['ts'] >= max_time)
                if np.any(tmp):  # there is at least one True
                    idx_last = tmp[tmp == True].index[0] - 1
                elif tmp_max_last == last_data_index:
                    idx_last = last_data_index
                else:
                    tmp_first = tmp_max_last
            # ts_last = self.data.loc[idx_last]['ts']

            # verify minimum number of samples in interval
            num_samples = (idx_last - idx_first) + 1
            if num_samples < min_samples:
                idx_first = idx_last + 1
                continue

            # get std and mean values
            x = self.data.loc[range(idx_first, idx_last)]['x'].std()
            y = self.data.loc[range(idx_first, idx_last)]['y'].std()
            z = self.data.loc[range(idx_first, idx_last)]['z'].std()
            xm = self.data.loc[range(idx_first, idx_last)]['x'].mean()
            ym = self.data.loc[range(idx_first, idx_last)]['y'].mean()
            zm = self.data.loc[range(idx_first, idx_last)]['z'].mean()

            self.windows.loc[n_win] = [idx_first, ts_first, self.static_win_duration, num_samples, x, y, z, xm, ym, zm]
            n_win += 1
            idx_first = idx_last + 1

    def find_window_stds_by_n_samples(self):
        #check if there is enough data
        n = np.floor(self.data.shape[0]/self.static_win_len).astype(int)
        if n == 0:
            return

        # code below may be optimized
        self.windows = pd.DataFrame(columns=['Df_data_index', 'TSstart', 'Duration', 'NumSamples', 'Xstd', 'Ystd', 'Zstd'])
        for i in range(n):
            idx_first = i*self.static_win_len
            idx_last = idx_first + self.static_win_len

            ts_first = self.data.loc[idx_first]['ts']
            x = self.data.loc[range(idx_first, idx_last)]['x'].std()
            y = self.data.loc[range(idx_first, idx_last)]['y'].std()
            z = self.data.loc[range(idx_first, idx_last)]['z'].std()
            duration = ts_first - self.data.loc[idx_last]['ts']
            self.windows.loc[i] = [idx_first, ts_first, duration, self.static_win_len, x, y, z]

        # set NaN values for window means
        self.windows['Xmean'] = np.NAN
        self.windows['Ymean'] = np.NAN
        self.windows['Zmean'] = np.NAN

    def find_static_windows(self):
        # Create variable with TRUE if nationality is USA
        x = self.windows['Xstd'] < self.std_thresh
        y = self.windows['Ystd'] < self.std_thresh
        z = self.windows['Zstd'] < self.std_thresh
        self.static_windows = self.windows[x & y & z].copy()

    def calc_static_windows_means(self):
        for i in self.static_windows.index:
            a = int(self.static_windows.loc[i]['Df_data_index'])
            b = int(a + self.static_win_len)
            self.static_windows.loc[i, 'Xmean'] = self.data.loc[range(a, b)]['x'].mean()
            self.static_windows.loc[i, 'Ymean'] = self.data.loc[range(a, b)]['y'].mean()
            self.static_windows.loc[i, 'Zmean'] = self.data.loc[range(a, b)]['z'].mean()

    def choose_n_static_windows(self, n=6, random_cycles=100, method=None):
        # set default to first
        if method is None:
            method = 'random'
        if n > self.static_windows.shape[0]:
            print("error: not enough static windows")
            return

        # Various methods to choose n static windows
        ind = None
        if method == 'temp':
            ind = [7, 16, 25, 34, 43, 52]
        elif method == 'random':
            def calc_dist(p):
                p = p.reset_index()
                d = 0
                for a in range(p.shape[0]-1):
                    for b in range(a+1, p.shape[0]):
                        d += m.sqrt((p.loc[a, 'Xmean'] - p.loc[b, 'Xmean'])**2 +
                                    (p.loc[a, 'Ymean'] - p.loc[b, 'Ymean'])**2 +
                                    (p.loc[a, 'Zmean'] - p.loc[b, 'Zmean'])**2)
                return d

            dist = 0
            full_ind = self.static_windows.index.values
            ind = None
            for i in range(random_cycles):
                temp_ind = random.sample(full_ind, n)
                temp_win = self.static_windows.loc[temp_ind].copy()
                temp_dist = calc_dist(temp_win)
                if temp_dist > dist:
                    dist = temp_dist
                    ind = temp_ind

        elif method == 'first_n':
            ind = self.static_windows.index[range(0, n)]
        else:
            pass
        win = self.static_windows.loc[ind]
        self.xyz_data = win.copy()
        self.xyz_data = self.xyz_data.drop(['TSstart', 'Duration', 'Df_data_index', 'Xstd', 'Ystd', 'Zstd'], axis=1)

    def find_calibration_matrix(self, ptype=6):
        x = self.xyz_data['Xmean'].tolist()
        y = self.xyz_data['Ymean'].tolist()
        z = self.xyz_data['Zmean'].tolist()

        def f(p):
            return abs(sum(np.array(equations(p))**2) - 0)

        if ptype == 6:
            def equations(p):
                ox, oy, oz, sx, sy, sz = p
                fun = []
                for i in range(self.xyz_data.shape[0]):
                    e = ((x[i] - ox)/sx)**2 + ((y[i] - oy)/sy)**2 + ((z[i] - oz)/sz)**2 - 1
                    fun.append(e)
                return fun

            pre_optim_var = optimize.fmin(f, (0., 0., 0., 1000., 1000., 1000.), maxfun=2000)
            off_x, off_y, off_z, slope_x, slope_y, slope_z = optimize.fsolve(equations, pre_optim_var)
            return off_x, off_y, off_z, slope_x, slope_y, slope_z
        elif ptype == 9:
            def equations(p):
                bx, by, bz, kx, ky, kz, a_yz, a_zy, a_zx = p
                fun = []
                for i in range(self.xyz_data.shape[0]):
                    xi = bx + kx * x[i] + a_yz * kx * y[i] - kx * z[i] * (a_zy - a_yz * a_zx)
                    yi = by + ky * y[i] + a_zx * ky * z[i]
                    zi = bz + kz * z[i]

                    e = (xi ** 2 + yi ** 2 + zi ** 2) - 1
                    fun.append(e)
                return fun
            pre_optim_var = optimize.fmin(f, (0., 0., 0., 90., 90., 90., 1000., 1000., 1000.), maxfun=2000)
            r_bx, r_by, r_bz, r_kx, r_ky, r_kz, r_a_yz, r_a_zy, r_a_zx = optimize.fsolve(equations, pre_optim_var)
            return r_bx, r_by, r_bz, r_kx, r_ky, r_kz, r_a_yz, r_a_zy, r_a_zx

        # Note - the warning on optimize.fmin  occurs after over maxfun (default 1000) function evaluations were made.
        # The default is maxfun = 1000. This is sufficient, so we can live with the warning since this step is actually
        # only a pre-optimization.
        # Warning: Maximum numberof function evaluations has been exceeded.

