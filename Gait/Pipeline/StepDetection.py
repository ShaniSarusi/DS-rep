import pickle
from os.path import join
import numpy as np
import pandas as pd
import Gait.Resources.config as c
from Utils.BasicStatistics.statistics_functions import cv
from Utils.Preprocessing.denoising import moving_average_no_nans
from Utils.Preprocessing.other_utils import normalize_max_min, reduce_dim_3_to_1
from Utils.SignalProcessing.peak_detection_and_handling import run_peak_utils_peak_detection, \
    merge_peaks_from_two_signals


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
        self.res = pd.DataFrame()
        self._set_manual_count_result()

    def _set_manual_count_result(self):
        if self.res.shape == (0, 0):
            self.res = pd.DataFrame(index=self.sample['SampleId'])
        self.res['sc_manual'] = self.sample['StepCount']
        self.res['cadence_manual'] = self.sample['CadenceWithCrop']
        self.res['speed_manual'] = self.sample['SpeedWithCrop']
        self.res['duration'] = self.sample['DurationWithCrop']

    def select_specific_samples(self, sample_ids):
        if isinstance(sample_ids, int):
            self.acc = [self.acc[sample_ids]]
            # Check if self.res has been filled
            if len(self.res) > 0:
                self.res = self.res.iloc[sample_ids:sample_ids + 1]
                if self.apdm_measures is not None:
                    self.apdm_measures = self.apdm_measures.iloc[sample_ids:sample_ids + 1]
                if self.apdm_events is not None:
                    self.apdm_events = self.apdm_events.iloc[sample_ids:sample_ids + 1]
        else:
            self.acc = [self.acc[i] for i in sample_ids]
            # Check if self.res has been filled
            if len(self.res) > 0:
                self.res = self.res.iloc[[i for i in sample_ids]]
                if self.apdm_measures is not None:
                    self.apdm_measures = self.apdm_measures.iloc[[i for i in sample_ids]]
                if self.apdm_events is not None:
                    self.apdm_events = self.apdm_events.iloc[[i for i in sample_ids]]

    def normalize_norm(self, use_single_max_min_for_all_samples=True):
        lhs = [self.acc[i]['lhs']['n'] for i in range(len(self.acc))]
        lhs = normalize_max_min(lhs, use_single_max_min_for_all_samples=use_single_max_min_for_all_samples)
        rhs = [self.acc[i]['rhs']['n'] for i in range(len(self.acc))]
        rhs = normalize_max_min(rhs, use_single_max_min_for_all_samples=use_single_max_min_for_all_samples)
        for i in range(len(self.acc)):
            self.acc[i]['lhs']['n'] = lhs[i]
            self.acc[i]['rhs']['n'] = rhs[i]

    def step_detection_single_side(self, side='lhs', signal_to_use='norm', vert_win=None,
                                   use_single_max_min_for_all_samples=True, smoothing=None, mva_win=20,
                                   peak_min_thr=0.2, peak_min_dist=20, verbose=True, do_normalization=True):
        if verbose: print("Running: step_detection_single_side on side: " + side)

        # Choose side
        data = [self.acc[i][side] for i in range(len(self.acc))]

        # Dimensionality reduction (3 to 1): Choose norm, vertical, or vertical with windows
        data = reduce_dim_3_to_1(data, signal_to_use, vert_win, verbose)

        # Normalize between 0 and 1 using min and max signal of all samples
        if do_normalization:
            data = normalize_max_min(data, use_single_max_min_for_all_samples=use_single_max_min_for_all_samples)

        # Smoothing
        if smoothing == 'mva':
            data = [moving_average_no_nans(data[i], mva_win) for i in range(len(data))]
            if verbose: print("\tStep: Smoothing, using " + smoothing + " with window size " + str(mva_win))

        # Peak detection
        if verbose: print("\tStep: Peak detection-1st order with min peak height [0-1]: " + str(peak_min_thr) +
                          " and min distance between peaks: " + str(peak_min_dist))
        idx = [run_peak_utils_peak_detection(data[i], peak_min_thr, peak_min_dist) for i in range(len(data))]

        # Save results
        if verbose: print("\tStep: Saving results")
        alg_name = side
        res = pd.Series(idx, index=self.res.index, name='idx_' + alg_name)
        self.res = pd.concat([self.res, res], axis=1)

    def step_detection_fusion_high_level(self, signal_to_use='norm', vert_win=None,
                                         use_single_max_min_for_all_samples=True, smoothing=None, mva_win=15,
                                         peak_min_thr=0.2, peak_min_dist=20, fusion_type='intersect',
                                         intersect_win=25, union_min_dist=20, union_min_thresh=0.5,
                                         verbose=True, do_normalization=True):

        if verbose: print("Running: step_detection_fusion_high_level - " + str(fusion_type))

        # Set data
        lhs = [self.acc[i]['lhs'] for i in range(len(self.acc))]
        rhs = [self.acc[i]['rhs'] for i in range(len(self.acc))]

        # Dimensionality reduction (3 to 1): Choose norm, vertical, or vertical with windows
        lhs = reduce_dim_3_to_1(lhs, signal_to_use, vert_win, verbose)
        rhs = reduce_dim_3_to_1(rhs, signal_to_use, vert_win, False)

        # Normalize between 0 and 1 using min and max signal of all samples
        if do_normalization:
            lhs = normalize_max_min(lhs, use_single_max_min_for_all_samples=use_single_max_min_for_all_samples)
            rhs = normalize_max_min(rhs, use_single_max_min_for_all_samples=use_single_max_min_for_all_samples)

        # Smoothing
        if smoothing == 'mva':
            if verbose: print("\tStep: Smoothing, using " + smoothing + " with window size " + str(mva_win))
            lhs = [moving_average_no_nans(lhs[i], mva_win) for i in range(len(lhs))]
            rhs = [moving_average_no_nans(rhs[i], mva_win) for i in range(len(rhs))]

        # Peak detection
        if verbose: print("\tStep: Peak detection-1st order with min peak height [0-1]: " + str(peak_min_thr) +
                          " and min distance between peaks: " + str(peak_min_dist))
        idx_lhs = [run_peak_utils_peak_detection(lhs[i], peak_min_thr, peak_min_dist) for i in range(len(lhs))]
        idx_rhs = [run_peak_utils_peak_detection(rhs[i], peak_min_thr, peak_min_dist) for i in range(len(rhs))]

        # Merge adjacent peaks from both sides into single peaks
        if verbose:
            if fusion_type == 'intersect':
                print("\tStep: Merging peaks from both sides. Max distance for intersect: " + str(intersect_win))
            else:
                print("\tStep: Merging peaks from both sides. Max distance for intersect: " + str(intersect_win) +
                      ". Min dist for union: " + str(union_min_dist) + ". Min peak thresh-union: " +
                      str(union_min_thresh))
        idx = [merge_peaks_from_two_signals(idx_lhs[i], idx_rhs[i], lhs[i][idx_lhs[i]], rhs[i][idx_rhs[i]],
                                            merge_type=fusion_type, intersect_win_size=intersect_win,
                                            union_min_dist=union_min_dist, union_min_thresh=union_min_thresh) for i
                                            in range(len(lhs))]

        # Save results
        if verbose: print("\tStep: Saving results")
        alg_name = 'fusion_high_level_' + str(fusion_type)
        res = pd.Series(idx, index=self.res.index, name='idx_' + alg_name)
        self.res = pd.concat([self.res, res], axis=1)

    def step_detection_fusion_low_level(self, signal_to_use='norm', vert_win=None, smoothing=None, mva_win=15,
                                        use_single_max_min_for_all_samples=True, fusion_type='sum', mva_win_combined=40,
                                        peak_min_thr=0.2, peak_min_dist=30, verbose=True, do_normalization=True):

        if verbose: print("Running: step_detection_fusion_low_level with fusion type: " + str(fusion_type))

        # Set data
        lhs = [self.acc[i]['lhs'] for i in range(len(self.acc))]
        rhs = [self.acc[i]['rhs'] for i in range(len(self.acc))]

        # Dimensionality reduction (3 to 1): Choose norm, vertical, or vertical with windows
        lhs = reduce_dim_3_to_1(lhs, signal_to_use, vert_win, verbose)
        rhs = reduce_dim_3_to_1(rhs, signal_to_use, vert_win, False)

        # Normalize between 0 and 1 using min and max signal of all samples
        if do_normalization:
            lhs = normalize_max_min(lhs, use_single_max_min_for_all_samples=use_single_max_min_for_all_samples)
            rhs = normalize_max_min(rhs, use_single_max_min_for_all_samples=use_single_max_min_for_all_samples)

        # Smoothing
        if smoothing == 'mva':
            if verbose: print("\tStep: Smoothing, using " + smoothing + " with window size " + str(mva_win))
            lhs = [moving_average_no_nans(lhs[i], mva_win) for i in range(len(lhs))]
            rhs = [moving_average_no_nans(rhs[i], mva_win) for i in range(len(rhs))]

        # Combine signals
        if verbose: print("\tStep: Combining signals- " + str(fusion_type))
        if fusion_type == 'sum':
            self.combined_signal = [lhs[i] + rhs[i] for i in range(len(lhs))]
        if fusion_type == 'diff':
            self.combined_signal = [np.abs(lhs[i] - rhs[i]) for i in range(len(lhs))]

        # Smooth the combined signal
        if verbose: print("\tStep: Smoothing the combined signals using mva with window " + str(mva_win_combined))
        signal = [moving_average_no_nans(x, mva_win_combined) for x in self.combined_signal]

        # Run peak detection
        if verbose: print("\tStep: Peak detection-1st order with min peak height [0-1]: " + str(peak_min_thr) +
                          " and min distance between peaks: " + str(peak_min_dist))
        idx = [run_peak_utils_peak_detection(x, peak_min_thr, peak_min_dist) for x in signal]

        # Save results
        if verbose: print("\tStep: Saving results")
        alg_name = 'fusion_low_level_' + str(fusion_type)
        res = pd.Series(idx, index=self.res.index, name='idx_' + alg_name)
        self.res = pd.concat([self.res, res], axis=1)

    def add_gait_metrics(self, verbose=True, max_dist_from_apdm=1234.5):
        if verbose: print("\rRunning: Adding gait metrics")
        # Find 'idx_' columns
        cols = [col for col in self.res.columns if 'idx_' in col]
        for col in cols:
            self.total_step_count_and_cadence(col)
            self.step_and_stride_durations(col, max_dist_from_apdm)
            self.step_and_stride_time_variability(col)
            self.step_time_asymmetry(col)

    def total_step_count_and_cadence(self, col):
        # Step count
        n_samples = self.res.shape[0]
        val = [len(self.res.iloc[i][col]) for i in range(n_samples)]
        res_steps = pd.Series(val, index=self.res.index, name=col.replace('idx_', 'sc_'))
        # Cadence
        val = [60 * res_steps.iloc[i] / self.res.iloc[i]['duration'] for i in range(n_samples)]
        res_cadence = pd.Series(val, index=self.res.index, name=col.replace('idx_', 'cadence_'))
        self.res = pd.concat([self.res, res_steps, res_cadence], axis=1)

    def step_and_stride_durations(self, col, max_dist_from_apdm=1234.5):
        step_durations = []
        n_samples = self.res.shape[0]
        for i in range(n_samples):
            step_idx = self.res[col].iloc[i]
            step_durations_i = []
            if len(step_idx) > 0:
                step_times = np.array(self.acc[i]['lhs']['ts'].iloc[step_idx] - self.acc[i]['lhs']['ts'].iloc[0])\
                             / np.timedelta64(1, 's')
                if max_dist_from_apdm != 1234.5:
                    apdm_i = np.array(self.apdm_events['Gait - Lower Limb - Toe Off L (s)'].iloc[i] +
                                              self.apdm_events['Gait - Lower Limb - Toe Off R (s)'].iloc[i])
                    if not np.any(np.isnan(apdm_i)):
                        step_times = np.array([step_time for step_time in step_times if
                                               np.min(np.abs(apdm_i - step_time)) < max_dist_from_apdm])
                        step_durations_i = np.diff(step_times)
            step_durations.append(step_durations_i)

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
        res = pd.Series(val, index=self.res.index, name=col.replace('idx_', 'step_durations_all_'))
        self.res = pd.concat([self.res, res], axis=1)

        val = [step_durations[i][::2] for i in range(n_samples)]
        res = pd.Series(val, index=self.res.index, name=col.replace('idx_', 'step_durations_side1_'))
        self.res = pd.concat([self.res, res], axis=1)

        val = [step_durations[i][1::2] for i in range(n_samples)]
        res = pd.Series(val, index=self.res.index, name=col.replace('idx_', 'step_durations_side2_'))
        self.res = pd.concat([self.res, res], axis=1)

        # Save stride values
        val = [stride_durations[i] for i in range(n_samples)]
        res = pd.Series(val, index=self.res.index, name=col.replace('idx_', 'stride_durations_all_'))
        self.res = pd.concat([self.res, res], axis=1)

        val = [stride_durations[i][::2] for i in range(n_samples)]
        res = pd.Series(val, index=self.res.index, name=col.replace('idx_', 'stride_durations_side1_'))
        self.res = pd.concat([self.res, res], axis=1)

        val = [stride_durations[i][1::2] for i in range(n_samples)]
        res = pd.Series(val, index=self.res.index, name=col.replace('idx_', 'stride_durations_side2_'))
        self.res = pd.concat([self.res, res], axis=1)

    def step_and_stride_time_variability(self, col):
        n_samples = self.res.shape[0]
        val = [cv(self.res.iloc[i][col.replace('idx_', 'step_durations_side1_')]) for i in range(n_samples)]
        res = pd.Series(val, index=self.res.index, name=col.replace('idx_', 'step_time_var_side1_'))
        self.res = pd.concat([self.res, res], axis=1)

        val = [cv(self.res.iloc[i][col.replace('idx_', 'step_durations_side2_')]) for i in range(n_samples)]
        res = pd.Series(val, index=self.res.index, name=col.replace('idx_', 'step_time_var_side2_'))
        self.res = pd.concat([self.res, res], axis=1)

        val = [cv(self.res.iloc[i][col.replace('idx_', 'stride_durations_side1_')]) for i in range(n_samples)]
        res = pd.Series(val, index=self.res.index, name=col.replace('idx_', 'stride_time_var_side1_'))
        self.res = pd.concat([self.res, res], axis=1)

        val = [cv(self.res.iloc[i][col.replace('idx_', 'stride_durations_side2_')]) for i in range(n_samples)]
        res = pd.Series(val, index=self.res.index, name=col.replace('idx_', 'stride_time_var_side2_'))
        self.res = pd.concat([self.res, res], axis=1)

    def step_time_asymmetry(self, col):
        n_samples = self.res.shape[0]
        # all gait cycle asymmetries per sample
        asym_gait_cycles = []
        for i in range(n_samples):
            side1 = self.res.iloc[i][col.replace('idx_', 'step_durations_side1_')]
            side2 = self.res.iloc[i][col.replace('idx_', 'step_durations_side2_')]
            asym_gait_cycles_sample_i = []
            num_gait_cycles = min(len(side1), len(side2))
            for j in range(num_gait_cycles):
                m = np.mean([side1[j], side2[j]])
                if m <= 0:
                    val = np.nan
                else:
                    val = np.abs(side1[j] - side2[j]) / m
                asym_gait_cycles_sample_i.append(val)
            asym_gait_cycles.append(asym_gait_cycles_sample_i)
        res = pd.Series(asym_gait_cycles, index=self.res.index, name=col.replace('idx_', 'step_time_asymmetry_values_'))
        self.res = pd.concat([self.res, res], axis=1)

        # Median per sample
        asym_median = []
        for i in range(n_samples):
            asym_gait_cycles_sample_i = asym_gait_cycles[i]
            val = np.nan
            if len(asym_gait_cycles_sample_i) > 0:
                if not np.any(np.isnan(asym_gait_cycles_sample_i)):
                    val = np.median(asym_gait_cycles_sample_i)
            asym_median.append(val)
        res = pd.Series(asym_median, index=self.res.index, name=col.replace('idx_', 'step_time_asymmetry_median_'))
        self.res = pd.concat([self.res, res], axis=1)

    def save(self, path):
        print("\rRunning: Saving data")
        with open(path, 'wb') as f:
            pickle.dump(self, f)


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

    # Run the step detection algorithms
    sd.step_detection_single_side(side='lhs', signal_to_use='norm', vert_win=None,
                               use_single_max_min_for_all_samples=True, smoothing=None, mva_win=20,
                               peak_min_thr=0.2, peak_min_dist=20, verbose=True)

    sd.step_detection_fusion_high_level(signal_to_use='norm', vert_win=None,
                                         use_single_max_min_for_all_samples=True, smoothing=None, mva_win=15,
                                         peak_min_thr=0.2, peak_min_dist=20, fusion_type='intersect',
                                         intersect_win=25, union_min_dist=20, union_min_thresh=0.5,
                                         verbose=True)

    sd.step_detection_fusion_low_level(signal_to_use='norm', vert_win=None, smoothing=None, mva_win=15,
                                        use_single_max_min_for_all_samples=True, fusion_type='sum', mva_win_combined=40,
                                        peak_min_thr=0.2, peak_min_dist=30, verbose=True)


    # Create results output
    sd.add_gait_metrics(max_dist_from_apdm=0.8)
    sd.save(path=join(c.pickle_path, 'sc_alg'))
