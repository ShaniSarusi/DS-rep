import Gait.config as c

# External imports
import pickle
from math import sqrt, ceil
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Imports from Utils
from Utils.SignalProcessing.peak_detection_and_handling import merge_adjacent_peaks_from_two_signals, \
    run_scipy_peak_detection, run_peak_utils_peak_detection, merge_adjacent_peaks_from_single_signal, \
    score_max_peak_within_fft_frequency_range
from Utils.DescriptiveStatistics.descriptive_statistics import cv, mean_and_std
from Utils.Preprocessing.denoising import moving_average_no_nans, butter_lowpass_filter, butter_highpass_filter
from Utils.Preprocessing.projections import project_gravity


class StepDetection:
    def __init__(self, p_acc, p_sample, p_apdm_measures=None, p_apdm_events=None):
        self.acc = p_acc
        self.sample = p_sample
        self.apdm_measures = p_apdm_measures
        self.apdm_events = p_apdm_events
        self.summary_table = []
        self.sampling_rate = c.sampling_rate
        self.lhs = []
        self.rhs = []
        self.combined_signal = []
        self.combined_signal_abs = []
        self.res = pd.DataFrame()

        # Initialize result table
        cols = ['sc_ensemble', 'idx_ensemble',
                'sc1_comb', 'idx1_comb', 'sc2_both', 'idx2_both', 'sc3_lhs', 'idx3_lhs', 'sc4_rhs', 'idx4_rhs',
                'step_dur_all', 'step_dur_side1', 'step_dur_side2', 'step_time_var_side1', 'step_time_var_side2',
                'step_time_asymmetry', 'stride_dur_all', 'stride_dur_side1', 'stride_dur_side2',
                'stride_time_var_side1', 'stride_time_var_side2', 'asymmetry_pci', 'idx_lhs2rhs_ratio', 'cadence']

    def set_manual_count_result(self):
        self.res = self.res(index=self.sample['SampleId'])
        self.res['sc_manual'] = self.sample['StepCount']
        self.res['cadence_manual'] = self.sample['CadenceWithCrop']
        self.res['speed_manual'] = self.sample['SpeedWithCrop']
        self.res['duration'] = self.sample['DurationWithCrop']

    def select_specific_samples(self, sample_ids):
        if isinstance(sample_ids, int):
            self.acc = [self.acc[sample_ids]]
            self.res = self.res.iloc[sample_ids:sample_ids+1]
        else:
            self.acc = [self.acc[i] for i in sample_ids]
            self.res = self.res.iloc[[i for i in sample_ids]]

    def step_detection_single_side(self, side='lhs', signal_to_use='norm', smoothing=None, mva_win=20,
                                   vert_win=None, butter_freq=10, peak_type='scipy', peak_param1=10, peak_param2=20,
                                   weak_signal_thresh=None, verbose=True):
        # Choose side
        data = [self.acc[i][side] for i in range(len(self.acc))]

        # Dimensionality reduction (3 to 1): Choose norm, vertical, or vertical with windows
        if verbose: print("Running: Selecting " + signal_to_use + " signal")
        if signal_to_use == 'norm':
            data = [data[i]['n'] for i in range(len(data))]
        if signal_to_use == 'vertical':
            data = [project_gravity(data[i]['x'], data[i]['y'], data[i]['z'], num_samples_per_interval=vert_win,
                                    return_only_vertical=True) for i in range(len(data))]

        # Smoothing
        if smoothing == 'mva':
            data = [moving_average_no_nans(data[i], mva_win) for i in range(len(data))]
        if smoothing == 'butter':
            data = [butter_lowpass_filter(data[i], butter_freq, self.sampling_rate, order=5) for i in range(len(data))]
        # TODO add option for lowpass and highpass bf. maybe just do in single one, with low zero being highpass.

        # Mean normalization
        data = [data[i] - data[i].mean() for i in range(len(data))]

        # Peak detection
        idx = None
        if peak_type == 'scipy':
            idx = [run_scipy_peak_detection(data[i], peak_param1, peak_param2) for i in range(len(data))]
        if peak_type == 'p_utils':
            idx = [run_peak_utils_peak_detection(data[i], peak_param1, peak_param2) for i in range(len(data))]

        # TODO need to implement this
        # Remove weak signals
        if weak_signal_thresh is not None:
            pass

        # Save results
        res = pd.Series(idx, name='idx_' + side)
        self.res = pd.concat([self.res, res], axis=1)

    def step_detection_two_sides_overlap(self, signal_to_use='norm', smoothing=None, mva_win=15,
                                         vert_win=None, butter_freq=12, peak_type='scipy', peak_param1=2,
                                         peak_param2=15, win_size_merge=30, win_size_remove_adjacent_peaks=40,
                                         verbose=True):

        # Set data
        lhs = [self.acc[i]['lhs'] for i in range(len(self.acc))]
        rhs = [self.acc[i]['rhs'] for i in range(len(self.acc))]

        # Dimensionality reduction (3 to 1): Choose norm, vertical, or vertical with windows
        if verbose: print("Running: Selecting " + signal_to_use + " signal")
        if signal_to_use == 'norm':
            lhs = [lhs[i]['n'] for i in range(len(lhs))]
            rhs = [rhs[i]['n'] for i in range(len(rhs))]
        if signal_to_use == 'vertical':
            lhs = [project_gravity(lhs[i]['x'], lhs[i]['y'], lhs[i]['z'], num_samples_per_interval=vert_win,
                                   return_only_vertical=True) for i in range(len(lhs))]
            rhs = [project_gravity(rhs[i]['x'], rhs[i]['y'], rhs[i]['z'], num_samples_per_interval=vert_win,
                                   return_only_vertical=True) for i in range(len(rhs))]

        # Smoothing
        if smoothing == 'mva':
            lhs = [moving_average_no_nans(lhs[i], mva_win) for i in range(len(lhs))]
            rhs = [moving_average_no_nans(rhs[i], mva_win) for i in range(len(rhs))]
        if smoothing == 'butter':
            lhs = [butter_lowpass_filter(lhs[i], butter_freq, self.sampling_rate, order=5) for i in range(len(lhs))]
            rhs = [butter_lowpass_filter(rhs[i], butter_freq, self.sampling_rate, order=5) for i in range(len(rhs))]
        # TODO add option for lowpass and highpass bf. maybe just do in single one, with low zero being highpass.

        # Mean normalization
        lhs = [lhs[i] - lhs[i].mean() for i in range(len(lhs))]
        rhs = [rhs[i] - rhs[i].mean() for i in range(len(rhs))]

        # Peak detection
        if peak_type == 'scipy':
            idx_lhs = [run_scipy_peak_detection(lhs[i], peak_param1, peak_param2) for i in range(len(lhs))]
            idx_rhs = [run_scipy_peak_detection(rhs[i], peak_param1, peak_param2) for i in range(len(rhs))]
        if peak_type == 'p_utils':
            idx_lhs = [run_peak_utils_peak_detection(lhs[i], peak_param1, peak_param2) for i in range(len(lhs))]
            idx_rhs = [run_peak_utils_peak_detection(rhs[i], peak_param1, peak_param2) for i in range(len(rhs))]

        # Merge adjacent peaks from both sides into single peaks
        merged_peaks = [merge_adjacent_peaks_from_two_signals(idx_lhs[i], idx_rhs[i], lhs[i][idx_lhs], rhs[i][idx_rhs],
                                                              'keep_max', win_size_merge) for i in range(len(lhs))]

        # Merge adjacent peaks from the 'merged peaks' before
        idx = [merge_adjacent_peaks_from_single_signal(merged_peaks[i], win_size_remove_adjacent_peaks) for i in
               range(len(merged_peaks))]

        # Save results
        res = pd.Series(idx, name='idx_' + 'overlap')
        self.res = pd.concat([self.res, res], axis=1)

    def step_detection_two_sides_combined_signal(self, signal_to_use='norm', smoothing=None, mva_win=15, vert_win=None,
                                                 butter_freq=12, mva_win_combined=40, min_hz=0.3, max_hz=2.0,
                                                 factor=1.1, peak_type='p_utils', peak_param1=0.5, peak_param2=30,
                                                 verbose=True):

        # Set data
        lhs = [self.acc[i]['lhs'] for i in range(len(self.acc))]
        rhs = [self.acc[i]['rhs'] for i in range(len(self.acc))]

        # Dimensionality reduction (3 to 1): Choose norm, vertical, or vertical with windows
        if verbose: print("Running: Selecting " + signal_to_use + " signal")
        if signal_to_use == 'norm':
            lhs = [lhs[i]['n'] for i in range(len(lhs))]
            rhs = [rhs[i]['n'] for i in range(len(rhs))]
        if signal_to_use == 'vertical':
            lhs = [project_gravity(lhs[i]['x'], lhs[i]['y'], lhs[i]['z'], num_samples_per_interval=vert_win,
                                   return_only_vertical=True) for i in range(len(lhs))]
            rhs = [project_gravity(rhs[i]['x'], rhs[i]['y'], rhs[i]['z'], num_samples_per_interval=vert_win,
                                   return_only_vertical=True) for i in range(len(rhs))]

        # Smoothing
        if smoothing == 'mva':
            lhs = [moving_average_no_nans(lhs[i], mva_win) for i in range(len(lhs))]
            rhs = [moving_average_no_nans(rhs[i], mva_win) for i in range(len(rhs))]
        if smoothing == 'butter':
            lhs = [butter_lowpass_filter(lhs[i], butter_freq, self.sampling_rate, order=5) for i in range(len(lhs))]
            rhs = [butter_lowpass_filter(rhs[i], butter_freq, self.sampling_rate, order=5) for i in range(len(rhs))]
        # TODO add option for lowpass and highpass bf. maybe just do in single one, with low zero being highpass.

        # Mean normalization
        lhs = [lhs[i] - lhs[i].mean() for i in range(len(lhs))]
        rhs = [rhs[i] - rhs[i].mean() for i in range(len(rhs))]

        # Combine signals
        combined_signal = [lhs[i] + rhs[i] for i in range(len(lhs))]
        combined_signal_abs = [np.abs(lhs[i]) + np.abs(rhs[i]) for i in range(len(lhs))]

        # Save the combined signal to enable plotting
        self.combined_signal = combined_signal
        self.combined_signal_abs = combined_signal_abs

        # Smooth the combined signal
        combined_signal = [moving_average_no_nans(combined_signal[i], mva_win_combined) for i in range(len(lhs))]
        combined_signal_abs = [moving_average_no_nans(combined_signal_abs[i], mva_win_combined) for i in
                               range(len(lhs))]

        # Select which combined signal to use by choosing the more oscillatory signal (sine-like). Do this using fft.
        score_combined = [score_max_peak_within_fft_frequency_range(combined_signal[i], self.sampling_rate, min_hz,
                                                                    max_hz, show=False) for i in range(len(lhs))]
        score_combined = [factor * score_combined[i] for i in range(len(lhs))]
        score_combined_abs = [score_max_peak_within_fft_frequency_range(combined_signal_abs[i], self.sampling_rate,
                              min_hz, max_hz, show=False) for i in range(len(lhs))]
        signal = [combined_signal[i].as_matrix() if (score_combined > score_combined_abs) else
                  combined_signal_abs[i].as_matrix() for i in range(len(lhs))]

        # Run peak detection
        idx = None
        if peak_type == 'scipy':
            idx = [run_scipy_peak_detection(signal[i], peak_param1, peak_param2) for i in range(len(signal))]
        if peak_type == 'p_utils':
            idx = [run_peak_utils_peak_detection(signal[i], peak_param1, peak_param2) for i in range(len(signal))]

        # Save results
        res = pd.Series(idx, name='idx_' + 'combined')
        self.res = pd.concat([self.res, res], axis=1)

    def ensemble_result_v1(self, win_size_merge_lhs_rhs, win_merge_lr_both):
        for i in range(len(self.lhs)):
            print("\rRunning: Integrating peaks, sample " + str(i + 1) + ' from ' + str(len(self.lhs)))
            lhs = pd.DataFrame({'idx': self.res.iloc[i]['idx_lhs'],
                                'maxsignal': self.lhs[i][self.res.iloc[i]['idx_lhs']].tolist()})
            rhs = pd.DataFrame({'idx': self.res.iloc[i]['idx_rhs'],
                                'maxsignal': self.rhs[i][self.res.iloc[i]['idx_rhs']].tolist()})
            lr = pd.concat([lhs, rhs]).sort_values(by='idx').reset_index(drop=True)
            idx_lr = merge_adjacent_peaks_from_single_signal(lr, win_size_merge_lhs_rhs)
            val = merge_adjacent_peaks_from_two_signals(self.res.iloc[i]['idx_overlap'], idx_lr, win_merge_lr_both,
                                                        p_type='keep_first_signal')
            self.res.set_value(self.res.index[i], 'idx_ensemble', val)

    def ensemble_result_v2(self, win_size=20, thresh=2, w1=1.0, w2=1.0, w3=1.0, w4=1.0):
        for i in range(len(self.lhs)):
            print("\rRunning: Ensemble result " + str(i + 1) + ' from ' + str(len(self.lhs)))
            num_t = len(self.lhs[i])
            s1 = np.zeros(num_t); s2 = np.zeros(num_t); s3 = np.zeros(num_t); s4 = np.zeros(num_t)
            s1[self.res.iloc[i]['idx_combined']] = 1
            s2[self.res.iloc[i]['idx_overlap']] = 1
            s3[self.res.iloc[i]['idx_lhs']] = 1
            s4[self.res.iloc[i]['idx_rhs']] = 1
            # s1 = max_filter(s1, win_size)
            s1 = moving_average_no_nans(s1, win_size) * w1
            s2 = moving_average_no_nans(s2, win_size) * w2
            s3 = moving_average_no_nans(s3, win_size) * w3
            s4 = moving_average_no_nans(s4, win_size) * w4
            x = s1 + s2 + s3 + s4
            val = run_peak_utils_peak_detection(x, float(thresh)/win_size, win_size)

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
        # Find 'idx_' columns
        cols = [col for col in self.res.columns if 'idx_' in col]
        n_samples = self.res.shape[0]

        self.add_total_step_count_and_cadence(cols, n_samples)
        self.add_step_and_stride_durations(cols, n_samples)
        self.add_step_and_stride_time_variability(cols, n_samples)
        self.add_step_time_asymmetry(cols, n_samples)
        # self.add_gait_pci_asymmetry()

    def add_total_step_count_and_cadence(self, cols, n_samples):
        for col in cols:
            # Step count
            val = [len(self.res.iloc[i][col]) for i in range(n_samples)]
            res_steps = pd.Series(val, name=col.replace('idx_', 'sc_'))
            # Cadence
            val = [60 * self.res.iloc[i][col] / self.res.iloc[i]['duration'] for i in range(n_samples)]
            res_cadence = pd.Series(val, name=col.replace('idx_', 'cadence_'))
            self.res = pd.concat([self.res, res_steps, res_cadence], axis=1)

    def add_step_and_stride_durations(self, cols, n_samples):
        for col in cols:
            step_idx = [self.res[i][col] for i in range(n_samples)]
            step_timestamps = [self.acc[i]['lhs']['ts'].iloc[step_idx] for i in range(n_samples)]

            # TODO check if below is in seconds. It is not, so make it so.
            step_durations = [np.diff(step_timestamps[i]).total_seconds()[0] for i in range(n_samples)]

            stride_durations = []
            for i in range(n_samples):
                stride_dur_i = []
                step_dur_i = step_durations[i]
                for j in range(1, len(step_dur_i)):
                    stride_time = step_dur_i[j - 1] + step_dur_i[j]
                    stride_dur_i.append(stride_time)
                stride_durations.append(stride_dur_i)

            # Save step values
            val = [step_durations[i] for i in range(n_samples)]
            res = pd.Series(val, name=col.replace('idx_', 'step_durations_all_'))
            self.res = pd.concat([self.res, res], axis=1)

            val = [step_durations[i][::2] for i in range(n_samples)]
            res = pd.Series(val, name=col.replace('idx_', 'step_durations_side1_'))
            self.res = pd.concat([self.res, res], axis=1)

            val = [step_durations[i][1::2] for i in range(n_samples)]
            res = pd.Series(val, name=col.replace('idx_', 'step_durations_side2_'))
            self.res = pd.concat([self.res, res], axis=1)

            # Save stride values
            val = [stride_durations[i] for i in range(n_samples)]
            res = pd.Series(val, name=col.replace('idx_', 'stride_durations_all_'))
            self.res = pd.concat([self.res, res], axis=1)

            val = [stride_durations[i][::2] for i in range(n_samples)]
            res = pd.Series(val, name=col.replace('idx_', 'stride_durations_side1_'))
            self.res = pd.concat([self.res, res], axis=1)

            val = [stride_durations[i][1::2] for i in range(n_samples)]
            res = pd.Series(val, name=col.replace('idx_', 'stride_durations_side2_'))
            self.res = pd.concat([self.res, res], axis=1)

    def add_step_and_stride_time_variability(self, cols, n_samples):
        for col in cols:
            val = [cv(self.res.iloc[i][col.replace('idx_', 'step_durations_side1_')]) for i in range(n_samples)]
            res = pd.Series(val, name=col.replace('idx_', 'step_time_var_side1_'))
            self.res = pd.concat([self.res, res], axis=1)

            val = [cv(self.res.iloc[i][col.replace('idx_', 'step_durations_side2_')]) for i in range(n_samples)]
            res = pd.Series(val, name=col.replace('idx_', 'step_time_var_side2_'))
            self.res = pd.concat([self.res, res], axis=1)

            val = [cv(self.res.iloc[i][col.replace('idx_', 'stride_durations_side1_')]) for i in range(n_samples)]
            res = pd.Series(val, name=col.replace('idx_', 'stride_time_var_side1_'))
            self.res = pd.concat([self.res, res], axis=1)

            val = [cv(self.res.iloc[i][col.replace('idx_', 'stride_durations_side2_')]) for i in range(n_samples)]
            res = pd.Series(val, name=col.replace('idx_', 'stride_time_var_side2_'))
            self.res = pd.concat([self.res, res], axis=1)

    def add_step_time_asymmetry(self, cols, n_samples):
        for col in cols:
            side1 = [self.res.iloc[i][col.replace('idx_', 'step_durations_side1_')].median() for i in range(n_samples)]
            side2 = [self.res.iloc[i][col.replace('idx_', 'step_durations_side2_')].median() for i in range(n_samples)]

            step_time_asymmetry = [100.0*(np.abs(side1[i] - side2[i]) / np.mean([side1[i], side2[i]])) for i in
                                   range(n_samples)]
            res = pd.Series(step_time_asymmetry, name=col.replace('idx_', 'step_time_asymmetry_'))
            self.res = pd.concat([self.res, res], axis=1)

    def create_results_table(self, ids, algo='lhs', save_name=None):
        # Define data frame
        idx = ['Step count', 'Cadence', 'Step time asymmetry (%)',
               'Stride time variability side1 (CV)', 'Stride time variability side2 (CV)',
               'Step time variability side1 (CV)', 'Step time variability side2 (CV)']
        df = pd.DataFrame(index=idx, columns=['manual', 'alg', 'mean error', 'rmse', 'rmse_combined', 'rmse_overlap',
                                              'rmse_lhs', 'rmse_rhs'])
        df.index.name = 'metric'

        # Step count
        df.set_value('Step count', 'manual', mean_and_std(self.res.loc[ids]['sc_manual']))
        df.set_value('Step count', 'alg', mean_and_std(self.res.loc[ids]['sc_' + algo]))
        err = (self.res.loc[ids]['sc_' + algo].mean() - self.res.loc[ids]['sc_manual'].mean()) / \
            self.res.loc[ids]['sc_manual'].mean()
        rmse = sqrt(mean_squared_error(self.res.loc[ids]['sc_manual'], self.res.loc[ids]['sc_' + algo]))
        normalized_rmse = rmse / self.res.loc[ids]['sc_manual'].mean()
        df.set_value('Step count', 'mean error', str(round(100.0*err, 2)) + "%")
        df.set_value('Step count', 'rmse', str(round(100.0*normalized_rmse, 2)) + "%")

        algorithms = ['lhs', 'rhs', 'combined', 'both']
        for algorithm in algorithms:
            normalized_rmse = sqrt(
                mean_squared_error(self.res.loc[ids]['sc_manual'], self.res.loc[ids]['sc_' + algorithm])) / \
                              self.res.loc[ids]['sc_manual'].mean()
            df.set_value('Step count', 'rmse_' + algorithm, str(round(100.0 * normalized_rmse, 2)) + "%")

        # Cadence
        df.set_value('Cadence', 'manual', mean_and_std(self.res.loc[ids]['cadence_manual']))
        df.set_value('Cadence', 'alg', mean_and_std(self.res.loc[ids]['cadence_' + algo]))
        err = (self.res.loc[ids]['cadence_' + algo].mean() - self.res.loc[ids]['cadence_manual'].mean()) / \
            self.res.loc[ids]['cadence_manual'].mean()
        rmse = sqrt(mean_squared_error(self.res.loc[ids]['cadence_manual'], self.res.loc[ids]['cadence_' + algo]))
        normalized_rmse = rmse / self.res.loc[ids]['cadence_manual'].mean()
        df.set_value('Cadence', 'mean error', str(round(100.0*err, 2)) + "%")
        df.set_value('Cadence', 'rmse', str(round(100.0*normalized_rmse, 2)) + "%")

        # Asymmetry - stride and step time
        # df.set_value('Stride time asymmetry (%)', 'alg',
        # GaitUtils.mean_and_std(self.res.loc[ids]['stride_time_asymmetry']))
        df.set_value('Step time asymmetry (%)', 'alg', mean_and_std(self.res.loc[ids]['step_time_asymmetry_' + algo]))

        # Variability - stride and step time
        df.set_value('Stride time variability side1 (CV)', 'alg', cv(self.res.loc[ids]['stride_time_var_side1_' +
                     algo]))
        df.set_value('Stride time variability side2 (CV)', 'alg', cv(self.res.loc[ids]['stride_time_var_side2' + algo]))
        df.set_value('Step time variability side1 (CV)', 'alg', cv(self.res.loc[ids]['step_time_var_side1' + algo]))
        df.set_value('Step time variability side2 (CV)', 'alg', cv(self.res.loc[ids]['step_time_var_side2' + algo]))

        # Store and save
        self.summary_table = df
        if save_name is not None:
            self.summary_table.to_csv(save_name)

    def save(self, path):
        print("\rRunning: Saving data")
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def plot_step_count_comparison_scatter(self, sc_algo_column_name='sc_lhs', ptype=1, idx=None, save_name=None,
                                           show=True, p_rmse=False):
        if idx is None:
            sc_alg = self.res[sc_algo_column_name]
            sc_manual = self.res['sc_manual']
        else:
            sc_alg = self.res.loc[idx][sc_algo_column_name]
            sc_manual = self.res.loc[idx]['sc_manual']

        if ptype == 1:
            plt.scatter(sc_alg, sc_manual, color='b')
            plt.ylabel('Visual step count', fontsize=22)
            plt.xlabel('Algorithm step count', fontsize=22)
            highest = max(max(sc_alg), max(sc_manual))
            ax_max = int(ceil(highest / 10.0)) * 10
            plt.ylim(0, ax_max)
            plt.xlim(0, ax_max)
            x = np.arange(0, ax_max)
            plt.plot(x, x)
        else:
            diff = sc_alg-sc_manual
            n_max = max(abs(diff))
            plt.scatter(sc_manual, diff, color='b')
            plt.ylabel('Algorithm steps - visual steps', fontsize=22)
            plt.ylim(-n_max - 2, n_max + 2)
            plt.xlim(min(sc_manual) - 5, max(sc_manual) + 5)
            plt.xlabel('Visual step count', fontsize=22)
            plt.axhline(0, color='r')
            highest = max(max(sc_alg), max(sc_manual))
            ax_max = int(ceil(highest / 10.0)) * 10

        plt.tick_params(axis='both', which='major', labelsize=18)

        # calculate rms
        rmse = sqrt(mean_squared_error(sc_manual, sc_alg))
        manual_mean = (sum(sc_manual) / len(sc_manual))
        nrmse = rmse / manual_mean
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

    def plot_signal_trace(self, id_num, side='both', add_val=0, show=True, font_small=True):
        t = self.res.index == id_num
        i = [j for j, x in enumerate(t) if x][0]

        if side == 'lhs':
            x = np.array(range(len(self.lhs[i])))/float(self.sampling_rate)
            plt.plot(x, self.lhs[i] + add_val, label='Left')
        if side == 'rhs':
            x = np.array(range(len(self.rhs[i]))) / float(self.sampling_rate)
            plt.plot(x, self.rhs[i] + add_val, label='Right')
        if side == 'both':
            x = np.array(range(len(self.lhs[i]))) / float(self.sampling_rate)
            plt.plot(x, self.lhs[i] + add_val, label='Left')
            plt.plot(x, self.rhs[i] + add_val, label='Right')
        if side == 'combined':
            x = np.array(range(len(self.combined_signal[i]))) / float(self.sampling_rate)
            plt.plot(x, self.combined_signal[i] + add_val, label='Combined')
        if side == 'combined_abs':
            x = np.array(range(len(self.combined_signal_abs[i]))) / float(self.sampling_rate)
            plt.plot(x, self.combined_signal_abs[i] + add_val, label='Combined_abs')
        if side == 'lhs2rhs_ratio':
            x = np.array(range(len(self.lhs[i]))) / float(self.sampling_rate)
            ratio = self.lhs[i]/self.rhs[i]
            plt.plot(x, ratio, label='Lhs2rhs_ratio')
        if side == 'lhs_minus_rhs':
            x = np.array(range(len(self.lhs[i]))) / float(self.sampling_rate)
            diff = self.lhs[i] - self.rhs[i]
            plt.plot(x, diff, label='Lhs_minus_rhs')
        if font_small:
            big = 18
            med = 14
            small = 12
        else:
            big = 28
            med = 24
            small = 18
        # plt.xlabel('Time (128Hz)', fontsize=big)
        plt.xlabel('Time (seconds)', fontsize=big)
        plt.ylabel('Acceleration (m/s2)', fontsize=big)
        plt.xticks(fontsize=med)
        plt.yticks(fontsize=med)
        plt.legend(fontsize=small)
        plt.tight_layout()
        if show:
            plt.show()

    def plot_step_idx(self, id_num, column_name='idx_lhs', p_color='k', show=True):
        # id_num is list of indices or a sample id
        if type(id_num) is list:
            idx = [int(128*i) for i in id_num]
        else:
            t = self.res.index == id_num
            i = [j for j, x in enumerate(t) if x][0]
            idx = self.res.iloc[i][column_name]
        # p_color examples:  'b', 'g', 'k'
        for i in range(len(idx)):
            plt.axvline(idx[i]/float(self.sampling_rate), color=p_color, ls='-.', lw=2)
        plt.tight_layout()
        if show:
            plt.show()


if __name__ == "__main__":
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
        sample = pickle.load(fp)
    with open(join(c.pickle_path, 'acc'), 'rb') as fp:
        acc = pickle.load(fp)
    with open(join(c.pickle_path, 'apdm_measures'), 'rb') as fp:
        apdm_measures = pickle.load(fp)
    with open(join(c.pickle_path, 'apdm_events'), 'rb') as fp:
        apdm_events = pickle.load(fp)

    # Use only samples with step count
    id_nums = sample[sample['StepCount'].notnull()].index.tolist()

    # Preprocessing
    sd = StepDetection(acc, sample, apdm_measures, apdm_events)
    sd.select_specific_samples(id_nums)
    sd.set_manual_count_result()

    # Run the step detection algorithms
    sd.step_detection_single_side(side='lhs', signal_to_use='norm', smoothing='mva', mva_win=20, vert_win=None,
                                  butter_freq=10, peak_type='scipy', peak_param1=10, peak_param2=20)
    sd.step_detection_single_side(side='rhs', signal_to_use='norm', smoothing='mva', mva_win=20, vert_win=None,
                                  butter_freq=10, peak_type='scipy', peak_param1=10, peak_param2=20)
    sd.step_detection_two_sides_overlap(signal_to_use='norm', smoothing='mva', mva_win=15,
                                     vert_win=None, butter_freq=12, peak_type='scipy', peak_param1=2, peak_param2=15,
                                     win_size_merge=30, win_size_remove_adjacent_peaks=40, verbose=True)
    sd.step_detection_two_sides_combined_signal(signal_to_use='norm', smoothing='mva', mva_win=15, vert_win=None,
                                             butter_freq=12, mva_win_combined=40, min_hz=0.3, max_hz=2.0,
                                             factor=1.1, peak_type='p_utils', peak_param1=0.5, peak_param2=30)

    # Integrate the 4 algorithms (lhs, rhs, both-merge, and combined signal)
    sd.ensemble_result_v1(win_size_merge_lhs_rhs=30, win_merge_lr_both=22)
    sd.ensemble_result_v2(win_size=10, thresh=1.5, w1=1, w2=0.8, w3=1, w4=1)  # TODO do some sliding window. maybe keep only windows with 2 or more peaks, winsize 10

    sd.calculate_lhs_to_rhs_signal_ratio('idx_ensemble')
    sd.remove_weak_signals()

    # Create results output
    sd.add_gait_metrics()
    sd.save(path=join(c.pickle_path, 'sc_alg'))
