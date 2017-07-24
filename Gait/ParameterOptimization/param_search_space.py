from hyperopt import hp
import Gait.config as c

# Initialize dictionary
space = dict()

# Side to work on
space['side'] = 'lhs'

# Signal to use
space['signal_to_use'] = hp.choice('signal_to_use', ['norm', 'vertical'])
space['do_windows_if_vertical'] = hp.choice('do_windows_if_vertical', [True, False])
space['vert_win'] = c.sampling_rate * (1 + hp.randint('vert_win', 30))

# choose which smoothing to perform (or none at all if mva_win is equal to ~1
space['smoothing'] = hp.choice('smoothing', ['mva', 'butter'])
space['mva_win'] = 1 + hp.randint('mva_win', 50)
space['butter_freq'] = 1 + 0.1 * hp.randint('butter_freq', 140)

# choose peak detection algorithm and parameters
# TODO - read into peak_utils more how to use
space['peak_type'] = hp.choice('peak_type', ['scipy', 'peak_utils'])
space['p1_sc'] = 1 + hp.randint('p1_sc', 21)
space['p2_sc'] = 1 + hp.randint('p2_sc', 41)
space['p1_pu'] = hp.uniform('p1_pu', 0.1, 1)
space['p2_pu'] = 5 + 5 * hp.randint('p2_pu', 9)

# Choose whether to remove weak signals
space['remove_weak_signals'] = hp.choice('remove_weak_signals', [False])
space['weak_signal_thresh'] = hp.uniform('weak_signal_thresh', -1, 0.5)

# Save dict in appropriate variable name
space_single_side = space



##############################################################################################################
## DEBUG SPACE

# # # Signal to use
# space['signal_to_use'] = 'norm'
# space['do_windows_if_vertical'] = True
# space['vert_win'] = 2451.864390791455
#
# # choose which smoothing to perform (or none at all if mva_win is equal to ~1
# space['smoothing'] = 'mva'
# space['mva_win'] = 12
# space['butter_freq'] = 1 + 0.1 * hp.randint('butter_freq', 140)
#
# # choose peak detection algorithm and parameters
# # TODO - read into peak_utils more how to use
# space['peak_type'] = 'peak_utils'
# space['p1_sc'] = 1 + hp.randint('p1_sc', 21)
# space['p2_sc'] = 1 + hp.randint('p2_sc', 41)
# space['p1_pu'] = 0.378
# space['p2_pu'] = 40
#
# # Choose whether to remove weak signals
# space['remove_weak_signals'] = hp.choice('remove_weak_signals', [False])
# space['weak_signal_thresh'] = hp.uniform('weak_signal_thresh', -1, 0.5)
#
# # Save dict in appropriate variable name
# space_single_side = space

