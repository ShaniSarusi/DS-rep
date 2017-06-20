from Utils.DataHandling import reading_files_and_directories as ut
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from os.path import join

# The shake was done using 2 not synchronized pebble watches.
# Read data***************************
common_path = join('C:\Users\zwaks\Documents\Workspaces\GitHub', 'DataScientists', 'BiogenExamples')
lhs = ut.read_export_tool_csv(join(common_path, 'w_lft_rawa.csv'))
rhs = ut.read_export_tool_csv(join(common_path, 'w_rt_rawb.csv'))

# Calculate Norm signal
lhs['n'] = (lhs['x'] ** 2 + lhs['y'] ** 2 + lhs['z'] ** 2) ** 0.5
rhs['n'] = (rhs['x'] ** 2 + rhs['y'] ** 2 + rhs['z'] ** 2) ** 0.5

# Find signal offset using cross-correlation
lhs_shake_range = range(3000, 4000)
rhs_shake_range = range(2000, 3000)
a = lhs.loc[lhs_shake_range, 'n'].as_matrix()
b = rhs.loc[rhs_shake_range, 'n'].as_matrix()
offset = np.argmax(signal.correlate(a,b))
shift_by = a.shape[0] - offset
time_offset = rhs.loc[rhs_shake_range[0] + shift_by, 'ts'] - lhs.loc[lhs_shake_range[0],'ts']

# Plot shift
plot_with_shift = False
idx_st = 500;  idx_en = 550
if plot_with_shift:
    plt.plot(a[range(idx_st, idx_en)])
    plt.plot(b[range(idx_st + shift_by, idx_en + shift_by)])
else:
    plt.plot(a[range(idx_st, idx_en)])
    plt.plot(b[range(idx_st, idx_en)])

