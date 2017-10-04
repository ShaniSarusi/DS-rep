import pickle
from os.path import join, sep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import Gait.Resources.config as c

do_algorithm = True
input_file = join('C:', sep, 'Users', 'zwaks', 'Desktop', 'GaitPaper','aa_param1small_500_sc_1002_v2', 'gait_measures.csv')

###################################################################
# read data
with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
    sample = pickle.load(fp)
data = pd.read_csv(input_file)

# get train and test ids
all_ids = sample[sample['CadenceApdmMean'].notnull()]['SampleId'].tolist()
tr_ids = data['SampleId'].tolist()

# test
ids_notnull = np.array(sample[sample['StepCount'].notnull()].index, dtype=int)
ids_jeremy_atia = np.array(sample[sample['Person'] == 'JeremyAtia'].index, dtype=int)
ids_efrat_wasserman = np.array(sample[sample['Person'] == 'EfratWasserman'].index, dtype=int)
ids_avishai_weingarten = np.array(sample[sample['Person'] == 'AvishaiWeingarten'].index, dtype=int)
ids_sternum_sensor_incorrect = np.array(sample[sample['Comments'] == 'Sternum was on back'].index, dtype=int)
ids_avishai_sternum = np.intersect1d(ids_avishai_weingarten, ids_sternum_sensor_incorrect)
ids_people_to_remove = np.sort(np.hstack((ids_jeremy_atia, ids_efrat_wasserman, ids_avishai_sternum)))
all_ids = np.setdiff1d(ids_notnull, ids_people_to_remove)
excluded_ids = np.setdiff1d(all_ids, tr_ids)

x = sample['CadenceApdmMean'].tolist()
y_count = sample['CadenceWithCrop'].tolist()

data = pd.read_csv(input_file)
ids = data['SampleId'].tolist()
y_alg = data['cadence_fusion_high_level_union_one_stage'].tolist()

x = [x[i] for i in ids]
y_count = [y_count[i] for i in ids]

# remove blanks
idx_nans = np.union1d(np.where(np.isnan(x))[0], np.where(np.isnan(y_count))[0])
idx_nans = np.union1d(idx_nans, np.where(np.isnan(y_alg))[0])
x = [i for j, i in enumerate(x) if j not in idx_nans]
y_count = [i for j, i in enumerate(y_count) if j not in idx_nans]
y_alg = [i for j, i in enumerate(y_alg) if j not in idx_nans]

if do_algorithm:
    y = y_alg
    save_name = 'apdm_vs_count_cadence.png'
    plt.ylabel('Wrists-union of events\n(steps/minute)', fontsize=11)
else:
    y = y_count
    save_name = 'apdm_vs_alg_cadence.png'
    plt.ylabel('Manual step count by subject\n(steps/minute)', fontsize=11)


# fix zero point
y = [y[i] if y[i]>0 else x[i] for i in range(len(y))]

plt.scatter(x, y)
plt.xlabel('Leg acceleromers\n(steps/minute)', fontsize=11)

val_max = max(max(x), max(y))
val_min = min(min(x), min(y))
val_range = val_max - val_min
ax_min = val_min - 0.05 * val_range
ax_max = val_max + 0.05 * val_range
if do_algorithm:
    ax_min = 60
    ax_max = 150
ax_range = ax_max - ax_min
plt.xlim(ax_min, ax_max)
plt.ylim(ax_min, ax_max)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# add correlation (R) value
r, _ = pearsonr(x, y)
plt.text(ax_min + 0.1*ax_range, ax_min + 0.9*ax_range, "R = " + str(round(r, 3)), fontsize=11)

#plt.subplots_adjust(top=0.9, left=0.9, bottom=0.2)
plt.grid(linestyle='-', which='major', color='lightgrey', alpha=0.5)
fig = plt.gcf()
fig.set_size_inches(4.5, 3.5)
plt.tight_layout()
plt.savefig(save_name, dpi=300)
