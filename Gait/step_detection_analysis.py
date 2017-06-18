from Gait.Pipeline.StepDetection import StepDetection
import pickle
from os.path import join
import numpy as np
import Gait.config as config

# load algorithm results
with open(join(config.common_path, 'Steps', 'sc_alg1'), 'rb') as fp:
    sc = pickle.load(fp)

# create indices for filtering
with open(join(config.pickle_path, 'metadata_sample'), 'rb') as fp:
    sample = pickle.load(fp)
all_samples = sample[sample['StepCount'].notnull()].index.tolist()
reg = sample[sample['WalkType'] == 'Regular']['SampleId'].as_matrix()
fast = sample[sample['PaceInstructions'] == 'H']['SampleId'].as_matrix()
med = sample[sample['PaceInstructions'] == 'M']['SampleId'].as_matrix()
slow = sample[sample['PaceInstructions'] == 'L']['SampleId'].as_matrix()

reg = np.intersect1d(all_samples, reg)
pd_sim = np.setdiff1d(all_samples, reg)
reg_fast = np.intersect1d(reg, fast)
reg_med = np.intersect1d(reg, med)
reg_slow = np.intersect1d(reg, slow)


# create results table
sc.create_results_table(ids=reg, save_name=join(config.results_path, 'res_summary.csv'))

# Plot results
rmse1 = sc.plot_results('sc1_comb', idx=reg, save_name=join(config.results_path, 'sc1_comb.png'), p_rmse=True)
rmse2 = sc.plot_results('sc2_both', idx=reg, save_name=join(config.results_path, 'sc2_both.png'), p_rmse=True)
rmse3 = sc.plot_results('sc3_lhs', idx=reg, save_name=join(config.results_path, 'sc3_lhs.png'), p_rmse=True)
rmse4 = sc.plot_results('sc4_rhs', idx=reg, save_name=join(config.results_path, 'sc4_rhs.png'), p_rmse=True)
rmse5 = sc.plot_results('sc_ensemble', idx=reg, save_name=join(config.results_path, 'sc5_ensemble.png'), p_rmse=True)

print('RMS results. 1comb: ' + str(rmse1) + ' 2both: ' + str(rmse2) + ' 3lhs: ' + str(rmse3) + ' 4rhs: ' + str(rmse4) + ' 5ensemble: ' + str(rmse5))
