import pickle
import numpy as np
import pandas as pd
from os.path import join, sep
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
# Imports from within package
import Gait.Resources.config as c
from Gait.Resources.gait_utils import calc_sc_for_excluded_ids

do_train_or_excluded = 'train'
#do_train_or_excluded = 'excluded'
do_algorithm = True
dirpath = join('C:', sep, 'Users', 'zwaks', 'Desktop', 'GaitPaper')
input_file = join(dirpath, 'a_cad10k_param4small_oct22_final', 'gait_measures.csv')

###################################################################
# read data
with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
    sample = pickle.load(fp)
    apdm_sc = sample['CadenceApdmMean'].tolist()
    manual_sc = sample['CadenceWithCrop'].tolist()

# get train and excluded ids
with open(join(dirpath,'a_cad10k_param4small_oct22_final', 'ids'), 'rb') as fp:
    tr_exc_ids = pickle.load(fp)
    if do_train_or_excluded == 'train':
        ids = tr_exc_ids['train']
        alg_data = pd.read_csv(input_file)
        alg_sc = alg_data['cadence_apdm_fusion_high_level_union_one_stage'].tolist()
        alg_sc = alg_data['cadence_apdm_rhs'].tolist()
    else:
        ids = tr_exc_ids['excluded']
        input_file = join(dirpath, 'a_cad10k_param4small_oct22_final',
                          'step_detection_fusion_high_level_union_one_stage_walk_taskall_all.csv')
        alg_sc = calc_sc_for_excluded_ids(input_file, ids)

apdm_sc = [apdm_sc[i] for i in ids]
manual_sc = [manual_sc[i] for i in ids]
# remove blanks
idx_nans = np.union1d(np.where(np.isnan(apdm_sc))[0], np.where(np.isnan(manual_sc))[0])
idx_nans = np.union1d(idx_nans, np.where(np.isnan(alg_sc))[0])
apdm_sc = [i for j, i in enumerate(apdm_sc) if j not in idx_nans]
manual_sc = [i for j, i in enumerate(manual_sc) if j not in idx_nans]
alg_sc = [i for j, i in enumerate(alg_sc) if j not in idx_nans]

if do_algorithm:
    y = alg_sc
    save_name = join(dirpath, 'apdm_vs_count_cadence_' + do_train_or_excluded + '.png')
    plt.ylabel('Wrists-union of events\n(steps/minute)', fontsize=11)
else:
    y = manual_sc
    save_name = join(dirpath, 'apdm_vs_alg_cadence_' + do_train_or_excluded + '.png')
    plt.ylabel('Manual step count by subject\n(steps/minute)', fontsize=11)


# fix zero point
y = [y[i] if y[i] > 0 else apdm_sc[i] for i in range(len(y))]

plt.scatter(apdm_sc, y)
plt.xlabel('Ankle acceleromers\n(steps/minute)', fontsize=11)

val_max = max(max(apdm_sc), max(y))
val_min = min(min(apdm_sc), min(y))
val_range = val_max - val_min
ax_min = val_min - 0.05 * val_range
ax_max = val_max + 0.05 * val_range
if do_algorithm:
    ax_min = 60
    ax_max = 150
ax_range = ax_max - ax_min
plt.xlim(ax_min, ax_max)
plt.ylim(ax_min, ax_max)
plt.xticks(np.arange(ax_min, ax_max, 20), fontsize=10)
plt.yticks(np.arange(ax_min, ax_max, 20), fontsize=10)
plt.grid(linestyle='-', which='major', color='lightgrey', alpha=0.5)

# add correlation (R) value
r, _ = pearsonr(apdm_sc, y)
plt.text(ax_min + 0.1*ax_range, ax_min + 0.9*ax_range, "R = " + str(round(r, 3)), fontsize=11)

# Save and show
fig = plt.gcf()
fig.set_size_inches(4, 4)
plt.tight_layout()
plt.savefig(save_name, dpi=300)
plt.show()
