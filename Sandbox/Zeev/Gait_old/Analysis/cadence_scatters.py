import pickle
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from Sandbox.Zeev import Gait_old as c

do_algorithm = True
input_file = join('C:\\Users\\zwaks\\Desktop\\GaitPaper\\param_per5', 'gait_measures_all.csv')

with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
    sample = pickle.load(fp)

x = sample['CadenceApdmMean'].tolist()
y_count = sample['CadenceWithCrop'].tolist()

data = pd.read_csv(input_file)
ids = data['SampleId'].tolist()
y_alg = data['cadence_overlap_strong'].tolist()

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
    plt.ylabel('Wrist accelerometer cadence (union approach)\n (steps/minute)', fontsize=12)
else:
    y = y_count
    save_name = 'apdm_vs_alg_cadence.png'
    plt.ylabel('Manual step count by subject\n (steps/minute)', fontsize=12)


# fix zero point
y = [y[i] if y[i]>0 else x[i] for i in range(len(y))]

plt.scatter(x, y)
plt.xlabel('Leg acceleromer cadence (steps/minute)', fontsize=12)
plt.ylabel('Wrist accelerometer cadence (union approach)\n (steps/minute)', fontsize=12)

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
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# add correlation (R) value
r, _ = pearsonr(x, y)
plt.text(ax_min + 0.1*ax_range, ax_min + 0.9*ax_range, "R = " + str(round(r, 3)), fontsize=12)

#plt.subplots_adjust(top=0.9, left=0.9, bottom=0.2)
plt.tight_layout()
plt.savefig(save_name)
