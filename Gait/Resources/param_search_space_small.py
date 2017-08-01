from copy import copy

from hyperopt import hp

#################################################################
# Dictionary for: step_detection_single_side
#################################################################
d = dict()

# Signal to use
d['signal_to_use'] = 'norm'
d['do_windows_if_vertical'] = False
d['vert_win'] = 5

# choose which smoothing to perform (or none at all if mva_win is equal to ~1
d['smoothing'] = 'mva'
d['mva_win'] = 5 + hp.randint('mva_win', 50)
d['butter_freq'] = 5

# Peak detection algorithm and parameters
d['peak_type'] = 'scipy'
d['p1_sc'] = 1 + hp.randint('p1_sc', 20)
d['p2_sc'] = d['p1_sc'] + 1 + hp.randint('p2_sc', 40)
d['p1_pu'] = 0.1
d['p2_pu'] = 10

# Choose whether to remove weak signals
d['remove_weak_signals'] = False
d['weak_signal_thresh'] = 0.5

# Save
space_single_side_lhs = d
space_single_side_lhs['side'] = 'lhs'

space_single_side_rhs = copy(d)
space_single_side_rhs['side'] = 'rhs'

#################################################################
# Dictionary for: step_detection_two_sides_overlap
#################################################################
d = dict()

# Signal to use
d['signal_to_use'] = 'norm'
d['do_windows_if_vertical'] = False
d['vert_win'] = 5

# choose which smoothing to perform (or none at all if mva_win is equal to ~1
# d['smoothing'] = hp.choice('smoothing', ['mva', 'butter'])
d['smoothing'] = 'mva'
d['mva_win'] = 5 + hp.randint('mva_win', 50)
d['butter_freq'] = 5

# Peak detection algorithm and parameters
d['peak_type'] = 'scipy'
d['p1_sc'] = 1 + hp.randint('p1_sc', 20)
d['p2_sc'] = d['p1_sc'] + 1 + hp.randint('p2_sc', 20)
d['p1_pu'] = 0.1
d['p2_pu'] = 10

# Signal merging parameters
d['win_size_merge'] = 2 + hp.randint('win_size_merge', 40)
d['win_size_remove_adjacent_peaks'] = 2 + hp.randint('win_size_remove_adjacent_peaks', 30)


# Save
space_overlap = d

#################################################################
# Dictionary for: step_detection_two_sides_combined_signal
#################################################################
d = dict()

# Signal to use
d['signal_to_use'] = 'norm'
d['do_windows_if_vertical'] = False
d['vert_win'] = 5

# Smoothing to perform on signals before combining(or none at all if mva_win is equal to ~1)
# d['smoothing'] = hp.choice('smoothing', ['mva', 'butter'])
d['smoothing'] = 'mva'
d['mva_win'] = 5 + hp.randint('mva_win', 50)
d['butter_freq'] = 3.5

# Smoothing after combining signals
d['mva_win_combined'] = 5 + hp.randint('mva_win_combined', 50)

# Choosing the more sine-like combined signal
d['min_hz'] = hp.uniform('min_hz', 0.1, 2)
d['max_hz'] = d['min_hz'] + 0.1 + hp.uniform('max_hz', 0.1, 5)
d['factor'] = 0.5 + 0.1 * hp.randint('factor', 10)


# Peak detection algorithm and parameters
d['peak_type'] = 'scipy'
d['p1_sc'] = 1 + hp.randint('p1_sc', 20)
d['p2_sc'] = d['p1_sc'] + 1 + hp.randint('p2_sc', 50)
d['p1_pu'] = 0.1
d['p2_pu'] = 10

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

