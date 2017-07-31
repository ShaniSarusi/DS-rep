from Gait.Pipeline.StepDetection import StepDetection
import pickle
from os.path import join
import numpy as np
import Gait.config as c
from Gait.Pipeline.gait_utils import set_filters, split_by_person
from Utils.DataHandling.data_processing import multi_intersect

# Load algorithm results
with open(join(c.pickle_path, 'sc_alg'), 'rb') as fp:
    sc = pickle.load(fp)
f = set_filters(exp=2)
walk_tasks = [1, 2, 3, 4, 5, 6, 7, 10]
n = split_by_person()

inp_all = f['notnull']
inp_good = np.intersect1d(inp_all, f['quality_good'])

reg = multi_intersect((f['notnull'], f['quality_good'], f['type_walk']))
reg = np.intersect1d(f['arm_status_free'], inp_good)
# reg = np.intersect1d(f['task_1'], inp)

# Create results table
sc.create_results_table(ids=reg, save_name=join(c.results_path, 'res_summary.csv'))

# Loop over tasks
res1 = []
res2 = []
for i in walk_tasks:
    name = 'task_' + str(i)
    idxs = np.intersect1d(f[name], inp_good)
    res1.append(sc.plot_step_count_comparison_scatter('sc_combined', idx=idxs, save_name=join(c.results_path, 'tmp.png'), p_rmse=True))
    res2.append(sc.plot_step_count_comparison_scatter('sc_overlap', idx=idxs, save_name=join(c.results_path, 'tmp.png'), p_rmse=True))
print(res1)
print(res2)

# Loop over people
nres1 = dict()
nres2 = dict()
for name in n.keys():
    idxs = np.intersect1d(n[name], inp_good)
    if len(idxs) == 0:
        continue
    nres1[name] = sc.plot_step_count_comparison_scatter('sc_combined', idx=idxs, save_name=join(c.results_path, 'tmp.png'), p_rmse=True)
    nres2[name] = sc.plot_step_count_comparison_scatter('sc_overlap', idx=idxs, save_name=join(c.results_path, 'tmp.png'), p_rmse=True)
print(nres1)
print(nres2)


# Plot results
rmse1 = sc.plot_step_count_comparison_scatter('sc_combined', ptype=2, idx=inp_good, save_name=join(c.results_path, 'sc_combined.png'), p_rmse=True)
rmse2 = sc.plot_step_count_comparison_scatter('sc_overlap', idx=reg, save_name=join(c.results_path, 'sc_overlap.png'), p_rmse=True)
rmse3 = sc.plot_step_count_comparison_scatter('sc_lhs', idx=reg, save_name=join(c.results_path, 'sc_lhs.png'), p_rmse=True)
rmse4 = sc.plot_step_count_comparison_scatter('sc_rhs', idx=reg, save_name=join(c.results_path, 'sc_rhs.png'), p_rmse=True)
# rmse5 = sc.plot_step_count_comparison_scatter('sc_ensemble', idx=reg, save_name=join(c.results_path, 'sc_ensemble.png'), p_rmse=True)

print('RMS results. 1combined: ' + str(rmse1) + ' 2overlap: ' + str(rmse2) + ' 3lhs: ' + str(rmse3) + ' 4rhs: ' +
      str(rmse4))
