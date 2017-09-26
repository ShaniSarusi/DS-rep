from hyperopt import hp


# Reduction in search space *****
# mva_win [2-71] instead of [2-129] - 55% of previous space
# peak_min_thr [0.01-0.60] instead of [0.01-0.99] - 60% of previous space
# peak_min_dist [0.01-0.60] instead of [10-79] - 60% of previous space
# union_min_thresh [0.01-0.5] instead of [0.01-0.99] - 50% of previous space
# mva_win_combined [2-70] instead of [2-129] - 55% of previous space


#################################################################
# Dictionary for: step_detection_single_side [Search space size is 20% of original]
#################################################################
space_single_side = dict()
space_single_side['mva_win'] = 2 + hp.randint('mva_win', 70)
space_single_side['peak_min_thr'] = 0.01 + 0.01 * hp.randint('peak_min_thr', 60)
space_single_side['peak_min_dist'] = 10 + hp.randint('peak_min_dist', 70)

#################################################################
# Dictionary for: step_detection_fusion_high_level_intersect [Search space size is 20% of original]
#################################################################
space_fusion_high_level_intersect = dict()
space_fusion_high_level_intersect['mva_win'] = 2 + hp.randint('mva_win', 70)
space_fusion_high_level_intersect['peak_min_thr'] = 0.01 + 0.01 * hp.randint('peak_min_thr', 60)
space_fusion_high_level_intersect['peak_min_dist'] = 10 + hp.randint('peak_min_dist', 70)
space_fusion_high_level_intersect['intersect_win'] = 1 + hp.randint('intersect_win',
                                                                    space_fusion_high_level_intersect['peak_min_dist'])

#################################################################
# Dictionary for: step_detection_fusion_high_level_union [Search space size is 10% of original]
#################################################################
space_fusion_high_level_union = dict()
space_fusion_high_level_union['mva_win'] = 2 + hp.randint('mva_win', 70)
space_fusion_high_level_union['peak_min_thr'] = 0.01 + 0.01 * hp.randint('peak_min_thr', 60)
space_fusion_high_level_union['peak_min_dist'] = 10 + hp.randint('peak_min_dist', 70)
space_fusion_high_level_union['intersect_win'] = 1 + hp.randint('intersect_win',
                                                                space_fusion_high_level_union['peak_min_dist'])
# union specific
space_fusion_high_level_union['union_min_dist'] = space_fusion_high_level_union['intersect_win'] + \
                                                  hp.randint('union_min_dist', 64)
space_fusion_high_level_union['union_min_thresh'] = 0.01 + 0.01 * hp.randint('union_min_thresh', 50)

#################################################################
# Dictionary for: step_detection_fusion_low_level [Search space size is 11% of original]
#################################################################
space_fusion_low_level = dict()
space_fusion_low_level['mva_win'] = 2 + hp.randint('mva_win', 70)
space_fusion_low_level['mva_win_combined'] = 2 + hp.randint('mva_win_combined', 70)
space_fusion_low_level['peak_min_thr'] = 0.01 + 0.01 * hp.randint('peak_min_thr', 60)
space_fusion_low_level['peak_min_dist'] = 10 + hp.randint('peak_min_dist', 70)
