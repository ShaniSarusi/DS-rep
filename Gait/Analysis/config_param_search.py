from hyperopt import hp

# single side space
space = dict()
space['signal'] = hp.choice('signal', ['norm', 'vertical'])
space['smoothing'] = hp.choice('smoothing', ['mva', 'butter'])
space['mva_win'] = 5 + 5 * hp.randint('mva_win', 9)
space['butter_freq_single_side'] = hp.uniform('butter_freq_single_side', 1, 6)
space['peaktype'] = hp.choice('peaktype', ['scipy', 'peak_utils'])
space['p1_sc'] = 1 + hp.randint('p1_sc', 21)
space['p2_sc'] = 1 + hp.randint('p2_sc', 41)
space['p1_pu'] = hp.uniform('p1_pu', 0.1, 1)
space['p2_pu'] = 5 + 5 * hp.randint('p2_pu', 9)
space['remove_weak_signals'] = hp.choice('remove_weak_signals', [False])
space['weak_signal_thresh'] = hp.uniform('weak_signal_thresh', -1, 0.5)
space_single_side = space
