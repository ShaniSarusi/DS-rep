# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 17:07:00 2017

@author: imazeh
"""

import os
from os.path import join

data_path = join('C:\\', 'Users', 'imazeh', 'Itzik', 'Health_prof', 'L_Dopa', 'Large_data/')
os.chdir(join('C:\\', 'Users', 'imazeh', 'Itzik', 'Health_prof', 'git4', 'DataScientists'))


exec(open(join('LDopa', 'data_reading', 'load_from_csv.py')).read())
sample_freq = 50
window_size = 5
lab_x, lab_y, lab_z = read_data_windows(data_path, read_also_home_data=False, sample_freq=sample_freq, window_size=window_size)

def convert_signal_for_ts_fresh(signal):
    