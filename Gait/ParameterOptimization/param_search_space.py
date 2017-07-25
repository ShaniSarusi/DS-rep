from hyperopt import hp
import Gait.config as c

#################################################################
# Dictionary for: step_detection_single_side
#################################################################
d = dict()

# Side to work on
d['side'] = 'lhs'

# Signal to use
d['signal_to_use'] = hp.choice('signal_to_use', ['norm', 'vertical'])
d['do_windows_if_vertical'] = hp.choice('do_windows_if_vertical', [True, False])
d['vert_win'] = c.sampling_rate * (1 + hp.randint('vert_win', 30))

# choose which smoothing to perform (or none at all if mva_win is equal to ~1
d['smoothing'] = hp.choice('smoothing', ['mva', 'butter'])
d['mva_win'] = 1 + hp.randint('mva_win', 50)
d['butter_freq'] = 1 + 0.1 * hp.randint('butter_freq', 140)

# Peak detection algorithm and parameters
# TODO - read into peak_utils more how to use
d['peak_type'] = hp.choice('peak_type', ['scipy', 'peak_utils'])
d['p1_sc'] = 1 + hp.randint('p1_sc', 21)
d['p2_sc'] = d['p1_sc'] + 1 + hp.randint('p2_sc', 40)
d['p1_pu'] = hp.uniform('p1_pu', 0.1, 1)
d['p2_pu'] = 5 + 5 * hp.randint('p2_pu', 9)

# Choose whether to remove weak signals
d['remove_weak_signals'] = hp.choice('remove_weak_signals', [False])
d['weak_signal_thresh'] = hp.uniform('weak_signal_thresh', -1, 0.5)

# Save
space_single_side = d

#################################################################
# Dictionary for: step_detection_two_sides_overlap
#################################################################
d = dict()

# Signal to use
d['signal_to_use'] = hp.choice('signal_to_use', ['norm', 'vertical'])
d['do_windows_if_vertical'] = hp.choice('do_windows_if_vertical', [True, False])
d['vert_win'] = c.sampling_rate * (1 + hp.randint('vert_win', 30))

# choose which smoothing to perform (or none at all if mva_win is equal to ~1
d['smoothing'] = hp.choice('smoothing', ['mva', 'butter'])
d['mva_win'] = 1 + hp.randint('mva_win', 50)
d['butter_freq'] = 1 + 0.1 * hp.randint('butter_freq', 140)

# Peak detection algorithm and parameters
d['peak_type'] = hp.choice('peak_type', ['scipy', 'peak_utils'])
d['p1_sc'] = 1 + hp.randint('p1_sc', 21)
d['p2_sc'] = d['p1_sc'] + 1 + hp.randint('p2_sc', 40)
d['p1_pu'] = hp.uniform('p1_pu', 0.1, 1)
d['p2_pu'] = 5 + 5 * hp.randint('p2_pu', 9)

# Signal merging parameters
d['win_size_merge'] = 1 + hp.randint('win_size_merge', 40)
d['win_size_remove_adjacent_peaks'] = 1 + hp.randint('win_size_remove_adjacent_peaks', 30)


# Save
space_overlap = d

#################################################################
# Dictionary for: step_detection_two_sides_combined_signal
#################################################################
d = dict()

# Signal to use
d['signal_to_use'] = hp.choice('signal_to_use', ['norm', 'vertical'])
d['do_windows_if_vertical'] = hp.choice('do_windows_if_vertical', [True, False])
d['vert_win'] = c.sampling_rate * (1 + hp.randint('vert_win', 30))

# Smoothing to perform on signals before combining(or none at all if mva_win is equal to ~1)
d['smoothing'] = hp.choice('smoothing', ['mva', 'butter'])
d['mva_win'] = 1 + hp.randint('mva_win', 50)
d['butter_freq'] = 1 + 0.1 * hp.randint('butter_freq', 140)

# Smoothing after combining signals
d['mva_win_combined'] = 1 + hp.randint('mva_win_combined', 50)

# Choosing the more sine-like combined signal
d['min_hz'] = hp.uniform('min_hz', 0.1, 2)
d['max_hz'] = d['min_hz'] + 0.1 + hp.uniform('max_hz', 0.1, 3)
d['factor'] = 0.5 + 0.1 * hp.randint('butter_freq', 10)


# Peak detection algorithm and parameters
d['peak_type'] = hp.choice('peak_type', ['scipy', 'peak_utils'])
d['p1_sc'] = 1 + hp.randint('p1_sc', 21)
d['p2_sc'] = d['p1_sc'] + 1 + hp.randint('p2_sc', 40)
d['p1_pu'] = hp.uniform('p1_pu', 0.1, 1)
d['p2_pu'] = 5 + 5 * hp.randint('p2_pu', 9)

# Save
space_combined = d



##############################################################################################################
## DEBUG SPACE

# # # Signal to use
# space['signal_to_use'] = 'norm'
# space['do_windows_if_vertical'] = True
# space['vert_win'] = 2451.864390791455
#
# # choose which smoothing to perform (or none at all if mva_win is equal to ~1
# space['smoothing'] = 'mva'
# space['mva_win'] = 48
# space['butter_freq'] = 1 + 0.1 * hp.randint('butter_freq', 140)
#
# # choose peak detection algorithm and parameters
# # TODO - read into peak_utils more how to use
# space['peak_type'] = 'scipy'
# space['p1_sc'] = 1 + hp.randint('p1_sc', 21)
# space['p2_sc'] = space['p1_sc'] + 1 + hp.randint(40)
# space['p1_pu'] = 0.378
# space['p2_pu'] = 40
#
# # Choose whether to remove weak signals
# space['remove_weak_signals'] = hp.choice('remove_weak_signals', [False])
# space['weak_signal_thresh'] = hp.uniform('weak_signal_thresh', -1, 0.5)
#
# # Save dict in appropriate variable name
# space_single_side = space

