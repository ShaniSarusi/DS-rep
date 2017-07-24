from Gait.Pipeline.StepDetection import StepDetection
from os.path import join
import pickle
import Gait.config as c
import Gait.Pipeline.gait_specific_utils as pre
import matplotlib.pyplot as plt
import scipy.stats as s

with open(join(c.pickle_path, 'sc_alg'), 'rb') as fp:
    sc = pickle.load(fp)
f = pre.set_filters(exp=2)

# Compare step asymmetry
apdm = sc.apdm_measures['asymmetry_step_time'][f['type_walk']]
intel = sc.res['step_time_asymmetry'][f['type_walk']]
corr = s.pearsonr(apdm, intel)

# Compare stride time variability
apdm = sc.apdm_measures['asymmetry_step_time'][f['type_walk']]
intel = sc.res['step_time_asymmetry'][f['type_walk']]
pear = s.pearsonr(apdm, intel)
spear = s.spearmanr(apdm, intel)