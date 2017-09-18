from hyperopt import hp

#################################################################
# Dictionary for: step_detection_single_side
#################################################################
space_single_side = dict()
space_single_side['mva_win'] = 2 + hp.randint('mva_win', 128)
space_single_side['peak_min_thr'] = 0.01 + 0.01 * hp.randint('peak_min_thr', 99)
space_single_side['peak_min_dist'] = 10 + hp.randint('peak_min_dist', 118)

#################################################################
# Dictionary for: step_detection_fusion_high_level_intersect
#################################################################
space_fusion_high_level_intersect = dict()
space_fusion_high_level_intersect['mva_win'] = 2 + hp.randint('mva_win', 128)
space_fusion_high_level_intersect['peak_min_thr'] = 0.01 + 0.01 * hp.randint('peak_min_thr', 99)
space_fusion_high_level_intersect['peak_min_dist'] = 10 + hp.randint('peak_min_dist', 118)
space_fusion_high_level_intersect['intersect_win'] = 1 + hp.randint('intersect_win',
                                                                    space_fusion_high_level_intersect['peak_min_dist'])

#################################################################
# Dictionary for: step_detection_fusion_high_level_union
#################################################################
space_fusion_high_level_union = dict()
space_fusion_high_level_union['mva_win'] = 2 + hp.randint('mva_win', 128)
space_fusion_high_level_union['peak_min_thr'] = 0.01 + 0.01 * hp.randint('peak_min_thr', 99)
space_fusion_high_level_union['peak_min_dist'] = 10 + hp.randint('peak_min_dist', 118)
space_fusion_high_level_union['intersect_win'] = 1 + hp.randint('intersect_win',
                                                                space_fusion_high_level_union['peak_min_dist'])
# union specific
space_fusion_high_level_union['union_min_dist'] = space_fusion_high_level_union['intersect_win'] + \
                                                  hp.randint('union_min_dist', 64)
space_fusion_high_level_union['union_min_thresh'] = 0.01 + 0.01 * hp.randint('union_min_thresh', 99)

#################################################################
# Dictionary for: step_detection_fusion_low_level
#################################################################
space_fusion_low_level = dict()
space_fusion_low_level['mva_win'] = 2 + hp.randint('mva_win', 128)
space_fusion_low_level['mva_win_combined'] = 2 + hp.randint('mva_win_combined', 128)
space_fusion_low_level['peak_min_thr'] = 0.01 + 0.01 * hp.randint('peak_min_thr', 99)
space_fusion_low_level['peak_min_dist'] = 10 + hp.randint('peak_min_dist', 118)
