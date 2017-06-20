# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:52:08 2017

@author: imazeh
"""

import pandas as pd
import numpy as np
from tsfresh import extract_features


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
