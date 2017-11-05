#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 10:47:45 2017

@author: HealthLOB
"""

import os

os.chdir('C:/Users/awagner/Documents/DataScientists')

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from Utils.Connections.connections import read_from_s3

meta = read_from_s3('aws-athena-query-results-726895906399-us-west-2', 'clinic_users/Musc_data/MUSC data/CSV uploaded to S3/F509_MotorTask_INTEL_20170925_V2.csv', 
'AKIAIEOL4GFG77QPNLCA', '06/OfU2vRMAkLGt69PjvVOSRfe1seABRQzhErL++', encoding = 'latin-1' )
meta = meta.dropna(axis = 0, how = 'any', thresh = 16)

meta['date_start'] = meta['reported timestamp start'].apply(lambda x: dt.datetime.strptime(str(x), '%d%b%Y:%H:%M:%S.%f'))#'%m/%d/%Y %H:%M.%S'
meta['date_end'] = meta['reported timestamp end'].apply(lambda x: dt.datetime.strptime(str(x), '%d%b%Y:%H:%M:%S.%f'))

query_for_athena = []
for user in np.unique(meta.user_id_intel):
    meta_pateint = meta.iloc[np.where((meta.user_id_intel == user))]
    meta_pateint = meta_pateint.sort('date_start')

    meta_pateint['date_by_month'] = meta_pateint['date_start'].apply(lambda x: x.date())
    meta_pateint_list = []
    for i in np.unique(meta_pateint['date_by_month']):
        meta_pateint_list.append(meta_pateint.iloc[np.where((meta_pateint.date_by_month == i))])
        
        timestemp_down = str(meta_pateint_list[len(meta_pateint_list) - 1].date_start.iloc[0] - dt.timedelta(hours = 1)) + '.000'
        timestemp_up = str(meta_pateint_list[len(meta_pateint_list) - 1].date_start.iloc[len(meta_pateint_list[len(meta_pateint_list) - 1].date_start) - 1] +
                   dt.timedelta(hours = 1)) + '.000'

    
        query_for_athena.append("Select * from watch_accelerometer where user_id = " + str(user) + 
                    " and timestamp > timestamp " + "'" + timestemp_down + "'"
                    " and timestamp < timestamp " + "'" + timestemp_up + "'"
                    " ORDER BY timestamp ASC")


import pickle



with open('query_for_athena.pickle', 'wb') as handle:
    pickle.dump(query_for_athena, handle, protocol=pickle.HIGHEST_PROTOCOL)
