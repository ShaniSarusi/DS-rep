# External imports
import pickle
from math import sqrt, ceil
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Imports from our packages
import Gait.GaitUtils.algorithms as uts
import Gait.config as c
from Utils.Preprocessing.denoising import moving_average_no_nans, butter_lowpass_filter, butter_highpass_filter
from Utils.Preprocessing.projections import project_gravity
from Utils.DescriptiveStatistics.descriptive_statistics import cv, mean_and_std
from Utils.DataHandling.data_processing import chunk_it


class StepDetection:
    def __init__(self, p_acc, p_sample):
        self.acc = p_acc
        self.summary_table = []
        self.sampling_rate = c.sampling_rate
        self.lhs = []
        self.rhs = []
        self.both = []
        self.both_abs = []

        # init result table
        cols = ['sc_true', 'cadence_true', 'speed_true', 'duration', 'sc_ensemble', 'idx_ensemble',
                'sc1_comb', 'idx1_comb', 'sc2_both', 'idx2_both', 'sc3_lhs', 'idx3_lhs', 'sc4_rhs', 'idx4_rhs',
                'step_dur_all', 'step_dur_side1', 'step_dur_side2', 'step_time_var_side1', 'step_time_var_side2',
                'step_time_asymmetry',
                'stride_dur_all', 'stride_dur_side1', 'stride_dur_side2', 'stride_time_var_side1',
                'stride_time_var_side2',
                'asymmetry_pci', 'idx_lhs2rhs_ratio', 'cadence']
        self.res = pd.DataFrame(index=p_sample['SampleId'], columns=cols)
        self.res['sc_true'] = p_sample['StepCount']
        self.res['cadence_true'] = p_sample['CadenceWithCrop']
        self.res['speed_true'] = p_sample['SpeedWithCrop']
        self.res['duration'] = p_sample['DurationWithCrop']

    def select_specific_samples(self, sample_ids):
        if isinstance(sample_ids, int):
            self.acc = [self.acc[sample_ids]]
            self.res = self.res.iloc[sample_ids:sample_ids+1]
        else:
            self.acc = [self.acc[i] for i in sample_ids]
            self.res = self.res.iloc[[i for i in sample_ids]]

    def bf(self, p_type, order=5, freq=6):
        for i in range(len(self.lhs)):
            if p_type == 'lowpass':
                self.lhs[i] = butter_lowpass_filter(self.lhs[i], freq, self.sampling_rate, order)
                self.rhs[i] = butter_lowpass_filter(self.rhs[i], freq, self.sampling_rate, order)
            if p_type == 'highpass':
                self.lhs[i] = butter_highpass_filter(self.lhs[i], freq, self.sampling_rate, order)
                self.rhs[i] = butter_highpass_filter(self.rhs[i], freq, self.sampling_rate, order)

    def select_signal(self, norm_or_vertical, win_size=None):
        print("\rRunning: Selecting " + norm_or_vertical + " signal")
        self.lhs = []
        self.rhs = []
        for i in range(len(self.acc)):
            if norm_or_vertical == 'vertical':
                # left
                x = self.acc[i]['lhs']['x']; y = self.acc[i]['lhs']['y']; z = self.acc[i]['lhs']['z']
                ver, hor = project_gravity(x, y, z, num_samples_per_interval=win_size)
                self.lhs.append(ver)
                # right
                x = self.acc[i]['rhs']['x']; y = self.acc[i]['rhs']['y']; z = self.acc[i]['rhs']['z']
                ver, hor = project_gravity(x, y, z, num_samples_per_interval=win_size)
                self.rhs.append(ver)
                pass
            else:
                self.lhs.append(self.acc[i]['lhs']['n'])
                self.rhs.append(self.acc[i]['rhs']['n'])

    def mva(self, p_type='nans', win_size=30, which=None):
        print("\rRunning: Moving average")
        for i in range(len(self.lhs)):
            if which == 'combined':
                if p_type == 'regular':
                    self.both[i] = pd.Series(self.both[i]).rolling(window=win_size, center=True).mean()
                    self.both_abs[i] = pd.Series(self.both_abs[i]).rolling(window=win_size, center=True).mean()
                if p_type == 'nans':
                    self.both[i] = moving_average_no_nans(self.both[i], win_size)
                    self.both_abs[i] = moving_average_no_nans(self.both_abs[i], win_size)
            else:
                if p_type == 'regular':
                    self.lhs[i] = pd.Series(self.lhs[i]).rolling(window=win_size, center=True).mean()
                    self.rhs[i] = pd.Series(self.rhs[i]).rolling(window=win_size, center=True).mean()
                if p_type == 'nans':
                    self.lhs[i] = moving_average_no_nans(self.lhs[i], win_size)
                    self.rhs[i] = moving_average_no_nans(self.rhs[i], win_size)

    def combine_signals(self):
        print("\rRunning: Combine signals")
        for i in range(len(self.lhs)):
            self.both.append(self.lhs[i] + self.rhs[i])
            self.both_abs.append(np.abs(self.lhs[i]) + np.abs(self.rhs[i]))

    def mean_normalization(self):
        print("\rRunning: Mean normalization")
        for i in range(len(self.lhs)):
            self.lhs[i] = self.lhs[i] - self.lhs[i].mean()
            self.rhs[i] = self.rhs[i] - self.rhs[i].mean()

    def step_detect_single_side_wpd_method(self, side=None, peak_type='scipy', p1=10, p2=20):
        if side == 'lhs':
            data = self.lhs
            col = 'idx3_lhs'
            s = 'left'
        elif side == 'rhs':
            data = self.rhs
            col = 'idx4_rhs'
            s = 'right'
        else:
            return
        for i in range(len(self.lhs)):
            print("\rRunning: Step detect single side (" + s + ") - wpd method, sample " + str(i + 1) + " from "
                  + str(len(self.lhs)))
            idx = uts.detect_peaks(data[i], peak_type, p1, p2)
            if idx is not None:
                self.res.set_value(self.res.index[i], col, [j for j in idx])
            else:
                self.res.set_value(self.res.index[i], col, [])

    def step_detect_overlap_method(self, win_merge, win_size_remove_adjacent_peaks, peak_type='scipy', p1=2, p2=15):
        for i in range(len(self.lhs)):
            print("\rRunning: Step detect two wrist devices - overlap method, sample " + str(i + 1) + ' from '
                  + str(len(self.lhs)))
            # detect peaks in each side. Go for high recall and low precision
            idx_l = uts.detect_peaks(self.lhs[i], peak_type, p1, p2)
            idx_r = uts.detect_peaks(self.rhs[i], peak_type, p1, p2)
            # merge peaks by choosing overlap
            peaks_both = uts.merge_peaks(idx_l, idx_r, self.lhs[i][idx_l], self.rhs[i][idx_r], 'keep_max', win_merge)
            idx_both = uts.remove_adjacent_peaks(peaks_both, win_size_remove_adjacent_peaks)
            # save result
            self.res.set_value(self.res.index[i], 'idx2_both', idx_both)

    def step_detect_combined_signal_method(self, min_hz=0.3, max_hz=2.0, factor=1.1, peak_type='p_utils', p1=0.5,
                                           p2=30):
        for i in range(len(self.lhs)):
            print("\rRunning: Step detect - combined signal method, sample " + str(i + 1) + ' from '
                  + str(len(self.lhs)))
            # choose which of 2 combined signal options to use. Choose by looking for most sine-like signal
            # 1) lhs + rhs     2) abs(lhs) + abs(rhs)
            sco1 = uts.score_max_peak(self.both[i], c.sampling_rate, min_hz, max_hz, show=False)
            sco2 = uts.score_max_peak(self.both_abs[i], c.sampling_rate, min_hz, max_hz, show=False)
            if (factor*sco1) > sco2:
                cb = self.both[i].as_matrix()
            else:
                cb = self.both_abs[i].as_matrix()

            idx = uts.detect_peaks(cb, peak_type, p1, p2)
            self.res.set_value(self.res.index[i], 'idx1_comb', idx)

    def ensemble_result_v1(self, win_size_merge_lhs_rhs, win_merge_lr_both):
        for i in range(len(self.lhs)):
            print("\rRunning: Integrating peaks, sample " + str(i + 1) + ' from ' + str(len(self.lhs)))
            lhs = pd.DataFrame({'idx': self.res.iloc[i]['idx3_lhs'],
                                'maxsignal': self.lhs[i][self.res.iloc[i]['idx3_lhs']].tolist()})
            rhs = pd.DataFrame({'idx': self.res.iloc[i]['idx4_rhs'],
                                'maxsignal': self.rhs[i][self.res.iloc[i]['idx4_rhs']].tolist()})
            lr = pd.concat([lhs, rhs]).sort_values(by='idx').reset_index(drop=True)
            idx_lr = uts.remove_adjacent_peaks(lr, win_size_merge_lhs_rhs)
            val = uts.merge_peaks(self.res.iloc[i]['idx2_both'], idx_lr, win_merge_lr_both, p_type='keep_first_signal')
            self.res.set_value(self.res.index[i], 'idx_ensemble', val)

    def ensemble_result_v2(self, win_size=20, thresh=2, w1=1.0, w2=1.0, w3=1.0, w4=1.0):
        for i in range(len(self.lhs)):
            print("\rRunning: Ensembl result " + str(i + 1) + ' from ' + str(len(self.lhs)))
            num_t = len(self.lhs[i])
            s1 = np.zeros(num_t); s2 = np.zeros(num_t); s3 = np.zeros(num_t); s4 = np.zeros(num_t)
            s1[self.res.iloc[i]['idx1_comb']] = 1
            s2[self.res.iloc[i]['idx2_both']] = 1
            s3[self.res.iloc[i]['idx3_lhs']] = 1
            s4[self.res.iloc[i]['idx4_rhs']] = 1
            # s1 = uts.max_filter(s1, win_size)
            s1 = moving_average_no_nans(s1, win_size) * w1
            s2 = moving_average_no_nans(s2, win_size) * w2
            s3 = moving_average_no_nans(s3, win_size) * w3
            s4 = moving_average_no_nans(s4, win_size) * w4
            x = s1 + s2 + s3 + s4
            val = uts.detect_peaks(x, peak_type='p_utils', param1=float(thresh)/win_size, param2=win_size)
            self.res.set_value(self.res.index[i], 'idx_ensemble', val)
            # x2 = (x >= thr).astype(int)

    def remove_weak_signals(self, thresh=0):
        for j in range(len(self.lhs)):
            print("\rRunning: Removing weak signals, sample " + str(j + 1) + ' from ' + str(len(self.lhs)))
            maxsig = [max(self.lhs[j][i], self.rhs[j][i]) for i in self.res.iloc[j]['idx_ensemble']]
            val = [i for i, maxval in zip(self.res.iloc[j]['idx_ensemble'], maxsig) if maxval > thresh]
            self.res.set_value(self.res.index[j], 'idx_ensemble', val)

    def calculate_lhs_to_rhs_signal_ratio(self, which):
        print("\rRunning: Adding left to right signal ratio to determine step side")
        for i in range(len(self.lhs)):
            idx = self.res.iloc[i][which]
            ratio = self.lhs[i][idx] / self.rhs[i][idx]
            self.res.set_value(self.res.index[i], 'idx_lhs2rhs_ratio', ratio.as_matrix())

    def add_gait_metrics(self):
        print("\rRunning: Adding gait metrics")
        self.add_total_step_count()
        self.add_gait_cadence()
        self.add_step_and_stride_durations()
        # TODO currently using number of indices as time to calculate step durations. Need to add timestamp later on
        self.add_step_and_stride_time_variability()
        # self.add_step_time_variability()
        self.add_step_and_stride_time_asymmetry()
        # self.add_gait_pci_asymmetry()

    def add_step_and_stride_durations(self):
        # TODO currently using number of indices as time to calculate step and stride durations.
        # TODO Need to add timestamp later on
        print("\rRunning: Adding step and stride durations")
        for i in range(self.res.shape[0]):
            res_idx = self.res.index[i]
            step_durations = np.diff(sc.res.loc[res_idx]['idx_ensemble'])
            self.res.set_value(res_idx, 'step_dur_all', step_durations)
            self.res.set_value(res_idx, 'step_dur_side1', step_durations[::2])
            self.res.set_value(res_idx, 'step_dur_side2', step_durations[1::2])
            stride_durations = []
            for j in range(1, len(step_durations)):
                stride_durations.append(step_durations[j-1] + step_durations[j])
            self.res.set_value(res_idx, 'stride_dur_all', np.asarray(stride_durations))
            self.res.set_value(res_idx, 'stride_dur_side1', np.asarray(stride_durations[::2]))
            self.res.set_value(res_idx, 'stride_dur_side2', np.asarray(stride_durations[1::2]))

    def add_step_and_stride_time_variability(self):
        print("\rRunning: Adding stride and step time variability")
        for i in range(self.res.shape[0]):
            res_idx = self.res.index[i]
            self.res.set_value(res_idx, 'step_time_var_side1', cv(self.res.loc[res_idx]['step_dur_side1']))
            self.res.set_value(res_idx, 'step_time_var_side2', cv(self.res.loc[res_idx]['step_dur_side2']))
            self.res.set_value(res_idx, 'stride_time_var_side1', cv(self.res.loc[res_idx]['stride_dur_side1']))
            self.res.set_value(res_idx, 'stride_time_var_side2', cv(self.res.loc[res_idx]['stride_dur_side2']))

    def add_step_and_stride_time_asymmetry(self):
        print("\rRunning: Adding stride and step time asymmetry")
        for i in range(self.res.shape[0]):
            res_idx = self.res.index[i]
            # step
            side1 = sc.res.loc[res_idx]['step_dur_side1'].mean()
            side2 = sc.res.loc[res_idx]['step_dur_side2'].mean()
            step_time_asymmetry = 100.0*(np.abs(side1 - side2) / np.mean([side1, side2]))
            self.res.set_value(res_idx, 'step_time_asymmetry', step_time_asymmetry)
            # stride
            # side1 = sc.res.loc[res_idx]['stride_dur_side1'].mean()
            # side2 = sc.res.loc[res_idx]['stride_dur_side2'].mean()
            # stride_time_asymmetry = 100.0*(np.abs(side1 - side2) / np.mean([side1, side2]))
            # self.res.set_value(res_idx, 'stride_time_asymmetry', stride_time_asymmetry)

    def create_results_table(self, ids, save_name=None):
        idx = ['Step count', 'Cadence', 'Step time asymmetry (%)',
               'Stride time variability side1 (CV)', 'Stride time variability side2 (CV)',
               'Step time variability side1 (CV)', 'Step time variability side2 (CV)']
        df = pd.DataFrame(index=idx, columns=['true', 'alg', 'mean error', 'rmse', 'rmse_comb', 'rmse_both', 'rmse_lhs',
                                              'rmse_rhs'])
        df.index.name = 'metric'
        # step count
        df.set_value('Step count', 'true', mean_and_std(self.res.loc[ids]['sc_true']))
        df.set_value('Step count', 'alg', mean_and_std(self.res.loc[ids]['sc_ensemble']))
        err = (self.res.loc[ids]['sc_ensemble'].mean() - self.res.loc[ids]['sc_true'].mean()) / \
            self.res.loc[ids]['sc_true'].mean()
        rmse = sqrt(mean_squared_error(self.res.loc[ids]['sc_true'], self.res.loc[ids]['sc_ensemble']))
        nrmse = rmse / self.res.loc[ids]['sc_true'].mean()
        df.set_value('Step count', 'mean error', str(round(100.0*err, 2)) + "%")
        df.set_value('Step count', 'rmse', str(round(100.0*nrmse, 2)) + "%")

        nrmse = sqrt(mean_squared_error(self.res.loc[ids]['sc_true'], self.res.loc[ids]['sc1_comb'])) / \
            self.res.loc[ids]['sc_true'].mean()
        df.set_value('Step count', 'rmse_comb', str(round(100.0 * nrmse, 2)) + "%")
        nrmse = sqrt(mean_squared_error(self.res.loc[ids]['sc_true'], self.res.loc[ids]['sc2_both'])) / \
            self.res.loc[ids]['sc_true'].mean()
        df.set_value('Step count', 'rmse_both', str(round(100.0 * nrmse, 2)) + "%")
        nrmse = sqrt(mean_squared_error(self.res.loc[ids]['sc_true'], self.res.loc[ids]['sc3_lhs'])) / \
            self.res.loc[ids]['sc_true'].mean()
        df.set_value('Step count', 'rmse_lhs', str(round(100.0 * nrmse, 2)) + "%")
        nrmse = sqrt(mean_squared_error(self.res.loc[ids]['sc_true'], self.res.loc[ids]['sc4_rhs'])) / \
            self.res.loc[ids]['sc_true'].mean()
        df.set_value('Step count', 'rmse_rhs', str(round(100.0 * nrmse, 2)) + "%")

        # Cadence
        df.set_value('Cadence', 'true', mean_and_std(self.res.loc[ids]['cadence_true']))
        df.set_value('Cadence', 'alg', mean_and_std(self.res.loc[ids]['cadence']))
        err = (self.res.loc[ids]['cadence'].mean() - self.res.loc[ids]['cadence_true'].mean()) / self.res.loc[ids][
            'cadence_true'].mean()
        rmse = sqrt(mean_squared_error(self.res.loc[ids]['cadence_true'], self.res.loc[ids]['cadence']))
        nrmse = rmse / self.res.loc[ids]['cadence_true'].mean()
        df.set_value('Cadence', 'mean error', str(round(100.0*err, 2)) + "%")
        df.set_value('Cadence', 'rmse', str(round(100.0*nrmse, 2)) + "%")

        # Asymmetry - stride and step time
        # df.set_value('Stride time asymmetry (%)', 'alg',
        # GaitUtils.mean_and_std(self.res.loc[ids]['stride_time_asymmetry']))
        df.set_value('Step time asymmetry (%)', 'alg', mean_and_std(self.res.loc[ids]['step_time_asymmetry']))

        # Variability - stride and step time
        df.set_value('Stride time variability side1 (CV)', 'alg', cv(self.res.loc[ids]['stride_time_var_side1']))
        df.set_value('Stride time variability side2 (CV)', 'alg', cv(self.res.loc[ids]['stride_time_var_side2']))
        df.set_value('Step time variability side1 (CV)', 'alg', cv(self.res.loc[ids]['step_time_var_side1']))
        df.set_value('Step time variability side2 (CV)', 'alg', cv(self.res.loc[ids]['step_time_var_side2']))

        # store and save
        self.summary_table = df
        if save_name is not None:
            self.summary_table.to_csv(save_name)

    def save(self, path):
        print("\rRunning: Saving data")
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def add_total_step_count(self):
        print("\rRunning: Adding step count")
        for i in range(self.res.shape[0]):
            res_idx = self.res.index[i]
            self.res.set_value(res_idx, 'sc1_comb', len(self.res.iloc[i]['idx1_comb']))
            self.res.set_value(res_idx, 'sc2_both', len(self.res.iloc[i]['idx2_both']))
            self.res.set_value(res_idx, 'sc3_lhs', len(self.res.iloc[i]['idx3_lhs']))
            self.res.set_value(res_idx, 'sc4_rhs', len(self.res.iloc[i]['idx4_rhs']))
            self.res.set_value(res_idx, 'sc_ensemble', len(self.res.iloc[i]['idx_ensemble']))

    def add_gait_cadence(self):
        print("\rRunning: Adding cadence")
        for i in range(self.res.shape[0]):
            res_idx = self.res.index[i]
            val = 60*self.res.iloc[i]['sc_ensemble']/self.res.iloc[i]['duration']
            self.res.set_value(res_idx, 'cadence', val)

    def plot_results(self, which, ptype=1, idx=None, save_name=None, show=True, p_rmse=False):
        if idx is None:
            sc_alg = self.res[which]
            sc_true = self.res['sc_true']
        else:
            sc_alg = self.res.loc[idx][which]
            sc_true = self.res.loc[idx]['sc_true']

        if ptype == 1:
            plt.scatter(sc_alg, sc_true, color='b')
            plt.ylabel('Visual step count', fontsize=22)
            plt.xlabel('Algorithm step count', fontsize=22)
            highest = max(max(sc_alg), max(sc_true))
            ax_max = int(ceil(highest / 10.0)) * 10
            plt.ylim(0, ax_max)
            plt.xlim(0, ax_max)
            x = np.arange(0, ax_max)
            plt.plot(x, x)
        else:
            diff = sc_alg-sc_true
            n_max = max(abs(diff))
            plt.scatter(sc_true, diff, color='b')
            plt.ylabel('Algorithm steps - visual steps', fontsize=22)
            plt.ylim(-n_max - 2, n_max + 2)
            plt.xlim(min(sc_true) - 5, max(sc_true) + 5)
            plt.xlabel('Visual step count', fontsize=22)
            plt.axhline(0, color='r')
            highest = max(max(sc_alg), max(sc_true))
            ax_max = int(ceil(highest / 10.0)) * 10

        plt.tick_params(axis='both', which='major', labelsize=18)

        # calculate rms
        rmse = sqrt(mean_squared_error(sc_true, sc_alg))
        true_mean = (sum(sc_true) / len(sc_true))
        nrmse = rmse / true_mean
        plt.text(0.05 * ax_max, 0.9 * ax_max, 'RMS = ' + str(round(rmse, 2)), fontsize=20)
        plt.text(0.05 * ax_max, 0.8 * ax_max, 'NRMS = ' + str(round(nrmse, 2)), fontsize=20)
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        if show:
            plt.show()
        if p_rmse:
            return round(rmse, 2)

    def plot_trace(self, id_num, side='both', add_val=0, tight=False, show=True):
        t = self.res.index == id_num
        i = [j for j, x in enumerate(t) if x][0]
        if side == 'lhs':
            plt.plot(self.lhs[i] + add_val, label='Left')
        if side == 'rhs':
            plt.plot(self.rhs[i] + add_val, label='Right')
        if side == 'both':
            plt.plot(self.lhs[i] + add_val, label='Left')
            plt.plot(self.rhs[i] + add_val, label='Right')
        if side == 'combined':
            plt.plot(self.both[i] + add_val, label='Combined')
        if side == 'combined_abs':
            plt.plot(self.both_abs[i] + add_val, label='Combined_abs')
        if side == 'lhs2rhs_ratio':
            ratio = self.lhs[i]/self.rhs[i]
            plt.plot(ratio, label='Lhs2rhs_ratio')
        if side == 'lhs_minus_rhs':
            diff = self.lhs[i] - self.rhs[i]
            plt.plot(diff, label='Lhs_minus_rhs')
        plt.xlabel('Time (128Hz)', fontsize=28)
        plt.ylabel('Acceleration (m/s2)', fontsize=28)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(fontsize=18)
        if tight:
            plt.tight_layout()
        if show:
            plt.show()

    def plot_step_idx(self, id_num, which, p_color, tight=False, show=True):
        t = self.res.index == id_num
        i = [j for j, x in enumerate(t) if x][0]
        idx = self.res.iloc[i][which]
        # p_color examples:  'b', 'g', 'k'
        for i in range(len(idx)):
            plt.axvline(idx[i], color=p_color, ls='-.', lw=2)
        if tight:
            plt.tight_layout()
        if show:
            plt.show()


if __name__ == "__main__":
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
        sample = pickle.load(fp)
    with open(join(c.pickle_path, 'acc'), 'rb') as fp:
        acc = pickle.load(fp)
    id_nums = sample[sample['StepCount'].notnull()].index.tolist()  # use only samples with step count

    # preprocessing
    sc = StepDetection(acc, sample)
    sc.select_specific_samples(id_nums)
    # sc.select_signal('norm')
    # sc.select_signal('vertical')
    sc.select_signal('vertical', win_size=c.sampling_rate*5)
    sc.mva(win_size=30)  # other options: sc.mva(p_type='regular', win_size=40) or sc.bf('lowpass', order=5, freq=6)
    sc.mean_normalization()

    # step detection - WPD single side
    sc.step_detect_single_side_wpd_method(side='lhs', peak_type='scipy', p1=10, p2=17)
    sc.step_detect_single_side_wpd_method(side='rhs', peak_type='scipy', p1=10, p2=17)
    sc.step_detect_overlap_method(win_merge=30, win_size_remove_adjacent_peaks=40, peak_type='scipy', p1=2, p2=15)

    # step detection - WPD single side
    sc.combine_signals()
    sc.mva(win_size=40, which='combined')  # another de-noising option: sc.mva(win_size=40, which='combined')
    sc.step_detect_combined_signal_method(min_hz=0.3, max_hz=2.0, factor=1.1, peak_type='p_utils', p1=0.5, p2=30)

    # integrate the 4 algorithms (lhs, rhs, both-merge, and combined signal)
    sc.ensemble_result_v1(win_size_merge_lhs_rhs=30, win_merge_lr_both=22)
    sc.ensemble_result_v2(win_size=10, thresh=1.5, w1=1, w2=0.8, w3=1, w4=1)  # TODO do some sliding window. maybe keep only windows with 2 or more peaks, winsize 10

    sc.calculate_lhs_to_rhs_signal_ratio('idx_ensemble')
    sc.remove_weak_signals()

    # create results output
    sc.add_gait_metrics()
    sc.save(path=join(c.pickle_path, 'sc_alg'))
