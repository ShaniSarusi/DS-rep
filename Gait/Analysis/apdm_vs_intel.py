from Gait.Pipeline.StepDetection import StepDetection
from os.path import join
import pickle
import Gait.config as c
from Gait.Pipeline.gait_utils import set_filters
import matplotlib.pyplot as plt
import scipy.stats as s

with open(join(c.pickle_path, 'sc_alg'), 'rb') as fp:
    sd = pickle.load(fp)
f = set_filters(exp=2)

# Compare step asymmetry
apdm = sd.apdm_measures['asymmetry_step_time'][f['type_walk']]
intel = sd.res['step_time_asymmetry'][f['type_walk']]
corr = s.pearsonr(apdm, intel)

# Compare stride time variability
apdm = sd.apdm_measures['asymmetry_step_time'][f['type_walk']]
intel = sd.res['step_time_asymmetry'][f['type_walk']]
pear = s.pearsonr(apdm, intel)
spear = s.spearmanr(apdm, intel)