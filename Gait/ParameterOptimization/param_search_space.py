from hyperopt import hp

# Initialize dictionary
space = dict()

# Side to work on
space['side'] = 'lhs'

# Signal to use
space['signal_to_use'] = hp.choice('signal_to_use', ['norm', 'vertical'])

# TODO add vertical window range (NOne or some range [128 to 30*128]

# choose which smoothing to perform (or none at all if mva_win is equal to ~1
space['smoothing'] = hp.choice('smoothing', ['mva', 'butter'])
space['mva_win'] = 1 + hp.randint('mva_win', 50)
space['butter_freq'] = 1 + 0.1 * hp.randint('butter_freq', 6)

# choose peak detection algorithm and parameters  TODO - read into peak_utils more how to use
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
