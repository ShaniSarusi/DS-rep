# functions (i.e. algorithms) to optimize
from Gait.Resources.gait_utils import create_sd_class_for_obj_functions, get_obj_function_results


def objective_step_detection_single_side_lhs(p):
    # s = create_sd_class_for_obj_functions()
    # s.normalize_norm()
    # s.select_specific_samples(p['sample_ids'])
    s = p['s']
    s.step_detection_single_side(side='lhs', signal_to_use='norm', vert_win=None,
        use_single_max_min_for_all_samples=True, smoothing='mva', mva_win=p['mva_win'], peak_min_thr=p['peak_min_thr'],
                                 peak_min_dist=p['peak_min_dist'], verbose=p['verbose'], do_normalization=False)
    s.add_gait_metrics(verbose=False, max_dist_from_apdm=p['max_dist_from_apdm'])
    return get_obj_function_results(s, 'lhs', p['metric'], verbose=p['verbose'])


def objective_step_detection_single_side_rhs(p):
    s = create_sd_class_for_obj_functions()
    s.normalize_norm()
    s.select_specific_samples(p['sample_ids'])
    s.step_detection_single_side(side='rhs', signal_to_use='norm', vert_win=None,
        use_single_max_min_for_all_samples=True, smoothing='mva', mva_win=p['mva_win'], peak_min_thr=p['peak_min_thr'],
                                 peak_min_dist=p['peak_min_dist'], verbose=p['verbose'], do_normalization=False)
    s.add_gait_metrics(verbose=False, max_dist_from_apdm=p['max_dist_from_apdm'])
    return get_obj_function_results(s, 'rhs', p['metric'], verbose=p['verbose'])


def step_detection_fusion_high_level_intersect(p):
    s = create_sd_class_for_obj_functions()
    s.normalize_norm()
    s.select_specific_samples(p['sample_ids'])
    s.step_detection_fusion_high_level(signal_to_use='norm', vert_win=None, use_single_max_min_for_all_samples=True,
        smoothing='mva', mva_win=p['mva_win'], peak_min_thr=p['peak_min_thr'], peak_min_dist=p['peak_min_dist'],
        fusion_type='intersect', intersect_win=p['intersect_win'], union_min_dist=20, union_min_thresh=0.5,
        verbose=p['verbose'], do_normalization=False)
    s.add_gait_metrics(verbose=False, max_dist_from_apdm=p['max_dist_from_apdm'])
    return get_obj_function_results(s, 'fusion_high_level_intersect', p['metric'], verbose=p['verbose'])


def step_detection_fusion_high_level_union_two_stages(p):
    s = create_sd_class_for_obj_functions()
    s.normalize_norm()
    s.select_specific_samples(p['sample_ids'])
    s.step_detection_fusion_high_level(signal_to_use='norm', vert_win=None, use_single_max_min_for_all_samples=True,
        smoothing='mva', mva_win=p['mva_win'], peak_min_thr=p['peak_min_thr'], peak_min_dist=p['peak_min_dist'],
        fusion_type='union_two_stages', intersect_win=p['intersect_win'], union_min_dist=p['union_min_dist'],
        union_min_thresh=p['union_min_thresh'], verbose=p['verbose'], do_normalization=False)
    s.add_gait_metrics(verbose=False, max_dist_from_apdm=p['max_dist_from_apdm'])
    return get_obj_function_results(s, 'fusion_high_level_union_two_stages', p['metric'], verbose=p['verbose'])


def step_detection_fusion_high_level_union_one_stage(p):
    s = create_sd_class_for_obj_functions()
    s.normalize_norm()
    s.select_specific_samples(p['sample_ids'])
    s.step_detection_fusion_high_level(signal_to_use='norm', vert_win=None, use_single_max_min_for_all_samples=True,
        smoothing='mva', mva_win=p['mva_win'], peak_min_thr=p['peak_min_thr'], peak_min_dist=p['peak_min_dist'],
        fusion_type='union_one_stage', union_min_dist=p['union_min_dist'], verbose=p['verbose'], do_normalization=False)
    s.add_gait_metrics(verbose=False, max_dist_from_apdm=p['max_dist_from_apdm'])
    return get_obj_function_results(s, 'fusion_high_level_union_one_stage', p['metric'], verbose=p['verbose'])


def step_detection_fusion_low_level_sum(p):
    s = create_sd_class_for_obj_functions()
    s.normalize_norm()
    s.select_specific_samples(p['sample_ids'])
    s.step_detection_fusion_low_level(signal_to_use='norm', vert_win=None, smoothing='mva', mva_win=p['mva_win'],
        use_single_max_min_for_all_samples=True, fusion_type='sum', mva_win_combined=p['mva_win_combined'],
        peak_min_thr=p['peak_min_thr'], peak_min_dist=p['peak_min_dist'], verbose=p['verbose'], do_normalization=False)
    s.add_gait_metrics(verbose=False, max_dist_from_apdm=p['max_dist_from_apdm'])
    return get_obj_function_results(s, 'fusion_low_level_sum', p['metric'], verbose=p['verbose'])


def step_detection_fusion_low_level_diff(p):
    s = create_sd_class_for_obj_functions()
    s.normalize_norm()
    s.select_specific_samples(p['sample_ids'])
    s.step_detection_fusion_low_level(signal_to_use='norm', vert_win=None, smoothing='mva', mva_win=p['mva_win'],
        use_single_max_min_for_all_samples=True, fusion_type='diff', mva_win_combined=p['mva_win_combined'],
        peak_min_thr=p['peak_min_thr'], peak_min_dist=p['peak_min_dist'], verbose=p['verbose'], do_normalization=False)
    s.add_gait_metrics(verbose=False, max_dist_from_apdm=p['max_dist_from_apdm'])
    return get_obj_function_results(s, 'fusion_low_level_diff', p['metric'], verbose=p['verbose'])
