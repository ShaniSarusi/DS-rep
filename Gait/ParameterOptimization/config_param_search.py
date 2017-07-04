from hyperopt import hp

# initialize dictionary
space = dict()

# choose which signal to use
space['signal'] = hp.choice('signal', ['norm', 'vertical'])

# choose which smoothing to perform (or none at all if mva_win is equal to ~1
space['smoothing'] = hp.choice('smoothing', ['mva', 'butter'])
space['mva_win'] = 1 + hp.randint('mva_win', 50)
# space['butter_freq_single_side'] = hp.uniform('butter_freq_single_side', 1, 6)
space['butter_freq_single_side'] = 1 + 0.1 * hp.randint('p1_sc', 6)

# choose peak detection algorithm and parameters  TODO - read into peak_utils more how to use
space['peak_type'] = hp.choice('peak_type', ['scipy', 'peak_utils'])
space['p1_sc'] = 1 + hp.randint('p1_sc', 21)
space['p2_sc'] = 1 + hp.randint('p2_sc', 41)
space['p1_pu'] = hp.uniform('p1_pu', 0.1, 1)
space['p2_pu'] = 5 + 5 * hp.randint('p2_pu', 9)

# Choose whether to remove weak signals
space['remove_weak_signals'] = hp.choice('remove_weak_signals', [False])
space['weak_signal_thresh'] = hp.uniform('weak_signal_thresh', -1, 0.5)

# save dict in appropriate variable name
space_single_side = space
