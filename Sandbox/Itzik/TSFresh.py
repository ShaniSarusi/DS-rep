# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 17:07:00 2017

@author: imazeh
"""

import os
from os.path import join
import pandas as pd
import numpy as np
from tsfresh import extract_features

data_path = join('C:\\', 'Users', 'imazeh', 'Itzik', 'Health_prof', 'L_Dopa', 'Large_data/')
os.chdir(join('C:\\', 'Users', 'imazeh', 'Itzik', 'Health_prof', 'git_team', 'DataScientists'))


exec(open(join('LDopa', 'DataReading', 'load_from_csv.py')).read())
sample_freq = 50
window_size = 5
lab_x, lab_y, lab_z = read_data_windows(data_path, read_also_home_data=False, sample_freq=sample_freq, window_size=window_size)

def convert_signals_for_ts_fresh(signals_data, dimension_name):
    '''
    This function reads signals, each of which in a row in a data-frame or in a
    separate list, and converts them to a format compatible with the TSFresh
    package.
    '''
    if type(signals_data) != pd.core.frame.DataFrame:
        signals_data = pd.DataFrame(signals_data)
    n_of_signals = signals_data.shape[0]
    n_of_elements = signals_data.shape[1]
    signal_id = []
    for i in range(n_of_signals):
        id_rep = [i]*n_of_elements
        signal_id.extend(id_rep)
    time_id = np.tile(range(n_of_elements), n_of_signals)
    acc = np.array(signals_data.stack(), dtype=pd.Series)
    tsf_df = pd.DataFrame({'signal_id': signal_id, 'time': time_id,
                           dimension_name: acc})
    tsf_df = tsf_df[['signal_id', 'time', dimension_name]]
    return tsf_df

data_sample = pd.DataFrame(lab_x).iloc[0:5]
data_for_tsf = convert_signals_for_ts_fresh(data_sample, 'bla')

### Extract 222 features from the signal (calculated per signal id)
acc_extracted_features = extract_features(data_for_tsf, column_id="signal_id", column_sort="time")
