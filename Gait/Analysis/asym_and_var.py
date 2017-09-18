import pickle
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import Gait.Resources.config as c
from Utils.DataHandling.data_processing import string_to_float_list

do_algorithm = True
input_file = join('C:\\Users\\zwaks\\Desktop\\GaitPaper\\param_per5', 'gait_measures_all.csv')

with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
    sample = pickle.load(fp)
with open(join(c.pickle_path, 'apdm_measures'), 'rb') as fp:
    apdm_measures = pickle.load(fp)

# remove alg_stride duration outliers: choose side1 or side 2
# above thresh > 3 + top5 or top10
max_stride_duration = 2
top_percent_outliers = 5
remove = 'z-score'
remove = 'mean_same'
z = 2
apdm_side = 'lhs'
alg_side = 'side2'
alg_name = 'high_level_fusion_union'

data = pd.read_csv(input_file)
ids = data['SampleId'].tolist()
apdm_measures = apdm_measures.loc[ids]
apdm = np.array(apdm_measures['stride_time_var_' + apdm_side].as_matrix(), dtype=float)
apdm_mean = apdm_measures['stride_time_mean_' + apdm_side]

a = data['stride_durations_' + alg_side + '_' + alg_name]
alg_var = []
alg_mean = []
for i in range(len(a)):
    b = np.array(string_to_float_list(a[i]))
    var_i = np.nan
    if len(b) > 0:
        # Remove above threshold - get rid of missed steps or strides not counted due to distance from apdm
        b = b[b < max_stride_duration]

        if remove == 'remove_pctile':
            # Remove top x percent
            b = b[b < np.percentile(b, 100-top_percent_outliers)]
        if remove == 'z-score':
            b = b[b < b.mean() + z*b.std()]
        else:  # remove is equalize means by removing from each side
            while len(b) > 3:
                if b[1:-1].mean() > apdm_mean.iloc[i]:
                    b = b[1:-1]
                else:
                    break
        var_i = b.std() / b.mean()
        mean_b = np.mean(b)
    # add cv
    alg_var.append(var_i)
    alg_mean.append(mean_b)

alg_var = np.array(alg_var)

# remove blanks
idx_nans = np.union1d(np.where(np.isnan(apdm))[0], np.where(np.isnan(alg_var))[0])
alg_var = [i for j, i in enumerate(alg_var) if j not in idx_nans]
apdm = [i for j, i in enumerate(apdm) if j not in idx_nans]
apdm_mean = [i for j, i in enumerate(apdm_mean) if j not in idx_nans]


r_lhs, _ = pearsonr(apdm, alg_var)
# r_lhs, _ = pearsonr(apdm_mean, alg_var)
print(r_lhs)

plt.scatter(apdm, alg_var)
#plt.scatter(apdm_mean, alg_var)
plt.xlabel('APDM')
plt.ylabel('alg')
