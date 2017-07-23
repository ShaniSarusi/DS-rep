from Gait.Pipeline.StepDetection import StepDetection
from os.path import join
import pickle
import Gait.config as c

with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
    sample = pickle.load(fp)
with open(join(c.pickle_path, 'acc'), 'rb') as fp:
    acc = pickle.load(fp)
with open(join(c.pickle_path, 'apdm_measures'), 'rb') as fp:
    apdm_measures = pickle.load(fp)
with open(join(c.pickle_path, 'apdm_events'), 'rb') as fp:
    apdm_events = pickle.load(fp)

id = 224
sc = StepDetection(acc, sample, apdm_measures, apdm_events)
sc.select_specific_samples(id)
sc.select_signal('norm')
# sc.plot_signal_trace(id, 'lhs', add_val=10)
sc.plot_signal_trace(id, 'lhs', font_small=True)
sc.plot_signal_trace(id, side='rhs')
sc.mva(win_size=30)  # other options: sc.mva(p_type='regular', win_size=40) or sc.bf('lowpass', order=5, freq=6)
sc.mean_normalization()
sc.combine_signals()
sc.mva(win_size=20, which='combined')  # another de-noising option: sc.mva(win_size=40, which='combined')
#sc.plot_signal_trace(id, 'lhs')
sc.plot_signal_trace(id, 'lhs', add_val=10)
sc.plot_signal_trace(id, side='rhs', add_val=5)
sc.plot_signal_trace(id, 'combined', tight=True)
#sc.plot_signal_trace(id, 'combined_abs', tight=True)

# step detection
sc.step_detect_single_side_wpd_method(side='lhs', peak_type='scipy', p1=10, p2=20)
sc.step_detect_single_side_wpd_method(side='rhs', peak_type='scipy', p1=10, p2=20)
#sc.step_detect_single_side_wpd_method(side='lhs', peak_type='scipy', p1=2, p2=15)
#sc.step_detect_single_side_wpd_method(side='rhs', peak_type='scipy', p1=2, p2=15)
sc.step_detect_overlap_method(win_merge=30, win_size_remove_adjacent_peaks=40, peak_type='scipy', p1=2, p2=15)
sc.step_detect_combined_signal_method(min_hz=0.3, max_hz=2.0, factor=1.1, peak_type='p_utils', p1=0.5, p2=30)

# integrate
sc.ensemble_result_v1(win_size_merge_lhs_rhs=30, win_merge_lr_both=22)
sc.ensemble_result_v2(win_size=10, thresh=1.5, w1=1, w2=0.8, w3=1, w4=1)

# Plot APDM
apdm_idx_initial_left = sc.apdm_events.iloc[id]['Gait - Lower Limb - Initial Contact L (s)']
apdm_idx_toe_off_left = sc.apdm_events.iloc[id]['Gait - Lower Limb - Toe Off L (s)']
apdm_idx_midswing_left = sc.apdm_events.iloc[id]['Gait - Lower Limb - Midswing L (s)']
sc.plot_step_idx(apdm_idx_initial_left)
sc.plot_step_idx(apdm_idx_toe_off_left, p_color='r')
sc.plot_step_idx(apdm_idx_midswing_left, p_color='g')
sc.plot_signal_trace(id, 'lhs', font_small=True)
sc.plot_signal_trace(id, side='rhs')




sc.plot_step_idx(id, 'idx1_comb', 'r')
sc.plot_step_idx(id, 'idx2_both', 'k')
sc.plot_step_idx(id, 'idx3_lhs', 'b')
sc.plot_step_idx(id, 'idx4_rhs', 'g', tight=True)


sc.calculate_lhs_to_rhs_signal_ratio('idx_ensemble')
sc.remove_weak_signals()

