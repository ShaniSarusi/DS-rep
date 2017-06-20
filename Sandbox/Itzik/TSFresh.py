# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 17:07:00 2017

@author: imazeh
"""

import os
from os.path import join
import pandas as pd
import numpy as np

data_path = join('C:\\', 'Users', 'imazeh', 'Itzik', 'Health_prof', 'L_Dopa', 'Large_data/')
os.chdir(join('C:\\', 'Users', 'imazeh', 'Itzik', 'Health_prof', 'git_team', 'DataScientists'))


exec(open(join('LDopa', 'data_reading', 'load_from_csv.py')).read())
sample_freq = 50
window_size = 5
lab_x, lab_y, lab_z = read_data_windows(data_path, read_also_home_data=False, sample_freq=sample_freq, window_size=window_size)

data_sample = pd.DataFrame(lab_x).iloc[0:2]
n_of_signals = data_sample.shape[0]
n_of_elements = data_sample.shape[1]
signal_id = []
for i in range(n_of_signals):
    id_rep = [i]*n_of_elements
    signal_id.extend(id_rep)
time_id = np.tile(range(n_of_elements), n_of_signals)
acc = np.array(data_sample.stack(), dtype=pd.Series)
tsf_df = pd.DataFrame({'signal_id':signal_id, 'time':time_id, 'x':acc})
tsf_df = tsf_df[['signal_id', 'time', 'x']]

def convert_signal_for_ts_fresh(signals_data):
    