# Two accelerometer pipeline
from os.path import join

import Gait.Resources.config as c
from Gait.TwoPebbles import TwoAcc
from Utils.DataHandling.reading_and_writing_files import read_export_tool_csv

# Read data
common_path = join(c.data_path, 'TwoPebbles_Exp2')
left_watch = join(common_path, 'w_lft_rawa.csv')
right_watch = join(common_path, 'w_rt_rawb.csv')
lhs = read_export_tool_csv(left_watch)
rhs = read_export_tool_csv(right_watch)

data = TwoAcc(lhs, rhs)
data.add_norm('both')
# Synchronize signal********************
# Find shake area using protocol and by plotting the signal  #example: plt.plot(lft.loc[range(100,8000),'norm'])
lhs_shake_range = range(3000, 4000)
rhs_shake_range = range(2000, 3000)
data.calc_offset(lhs_shake_range, rhs_shake_range)

# Plot synch
lhs_plot_range = range(3500, 3550)
data.plot_shift(lhs_plot_range, show_shift=False)
data.plot_shift(lhs_plot_range, show_shift=True)
