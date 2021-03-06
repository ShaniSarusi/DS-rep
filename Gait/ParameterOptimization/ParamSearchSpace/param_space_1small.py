from hyperopt import hp

#################################################################
# Dictionary for: step_detection_single_side
#################################################################
space_single_side = dict()
space_single_side['mva_win'] = 2 + hp.randint('mva_win', 70)
space_single_side['peak_min_thr'] = 0.01 + 0.01 * hp.randint('peak_min_thr', 60)
space_single_side['peak_min_dist'] = 10 + hp.randint('peak_min_dist', 70)

#################################################################
# Dictionary for: step_detection_fusion_high_level_intersect
#################################################################
space_fusion_high_level_intersect = dict()
space_fusion_high_level_intersect['mva_win'] = 2 + hp.randint('mva_win', 70)
space_fusion_high_level_intersect['peak_min_thr'] = 0.01 + 0.01 * hp.randint('peak_min_thr', 60)
space_fusion_high_level_intersect['peak_min_dist'] = 2 + hp.randint('peak_min_dist', 80)
space_fusion_high_level_intersect['intersect_win'] = 1 + hp.randint('intersect_win',
                                                                    space_fusion_high_level_intersect['peak_min_dist'])


#################################################################
# Dictionary for: step_detection_fusion_high_level_union_two_stages
#################################################################
space_fusion_high_level_union_two_stages = dict()
space_fusion_high_level_union_two_stages['mva_win'] = 2 + hp.randint('mva_win', 80)
space_fusion_high_level_union_two_stages['peak_min_thr'] = 0.01 + 0.01 * hp.randint('peak_min_thr', 60)
space_fusion_high_level_union_two_stages['peak_min_dist'] = 2 + hp.randint('peak_min_dist', 80)
space_fusion_high_level_union_two_stages['intersect_win'] = 1 + hp.randint('intersect_win',
                                                                           space_fusion_high_level_union_two_stages['peak_min_dist'])
# union specific
space_fusion_high_level_union_two_stages['union_min_dist'] = space_fusion_high_level_union_two_stages['intersect_win'] + \
                                                  hp.randint('union_min_dist', 70)
space_fusion_high_level_union_two_stages['union_min_thresh'] = 0.01 + 0.01 * hp.randint('union_min_thresh', 60)

#################################################################
# Dictionary for: step_detection_fusion_high_level_union_one_stage
#################################################################
space_fusion_high_level_union_one_stage = dict()
space_fusion_high_level_union_one_stage['mva_win'] = 2 + hp.randint('mva_win', 80)
space_fusion_high_level_union_one_stage['peak_min_thr'] = 0.01 + 0.01 * hp.randint('peak_min_thr', 60)
space_fusion_high_level_union_one_stage['peak_min_dist'] = 2 + hp.randint('peak_min_dist', 80)
space_fusion_high_level_union_one_stage['union_min_dist'] = 10 + hp.randint('union_min_dist', 70)

#################################################################
# Dictionary for: step_detection_fusion_low_level
#################################################################
space_fusion_low_level = dict()
space_fusion_low_level['mva_win'] = 2 + hp.randint('mva_win', 80)
space_fusion_low_level['mva_win_combined'] = 2 + hp.randint('mva_win_combined', 60)
space_fusion_low_level['peak_min_thr'] = 0.01 + 0.01 * hp.randint('peak_min_thr', 60)
space_fusion_low_level['peak_min_dist'] = 10 + hp.randint('peak_min_dist', 70)
