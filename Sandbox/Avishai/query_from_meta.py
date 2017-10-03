#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 10:47:45 2017

@author: HealthLOB
"""

import datetime as dt
import pandas as pd

meta = pd.read_csv('/home/lfaivish/PycharmProjects/Deepshit/DATA_FOLDER/musc_motor_tasks.csv', encoding = 'latin-1')
meta = meta.dropna(axis = 0, how = 'any', thresh = 3)

meta['date_start'] = meta['Tsstart'].apply(lambda x: dt.datetime.strptime(x, '%m/%d/%Y %H:%M.%S'))
meta['date_end'] = meta['Tsend'].apply(lambda x: dt.datetime.strptime(x, '%m/%d/%Y %H:%M.%S'))

query_for_athena = []
for user in np.unique(meta.user_id):
    meta_pateint = meta.iloc[np.where((meta.user_id == user))]
    meta_pateint = meta_pateint.sort('date_start')

    meta_pateint['date_by_month'] = meta_pateint['date_start'].apply(lambda x: x.date())
    meta_pateint_list = []
    for i in np.unique(meta_pateint['date_by_month']):
        meta_pateint_list.append(meta_pateint.iloc[np.where((meta_pateint.date_by_month == i))])
        
        timestemp_down = str(meta_pateint_list[len(meta_pateint_list) - 1].date_start.iloc[0] - dt.timedelta(hours = 1)) + '.000'
        timestemp_up = str(meta_pateint_list[len(meta_pateint_list) - 1].date_start.iloc[len(meta_pateint_list[len(meta_pateint_list) - 1].date_start) - 1] +
                   dt.timedelta(hours = 1)) + '.000'

    
        query_for_athena.append('Select * From production."Wacth_accelerometer" where user_id = ' + str(user) + 
                    ' and timestamp > timestamp ' + timestemp_down +
                    ' and timestamp < timestamp ' + timestemp_up + 
                    ' ORDER BY timestamp ASC')


import pickle



with open('query_for_athena.pickle', 'wb') as handle:
    pickle.dump(query_for_athena, handle, protocol=pickle.HIGHEST_PROTOCOL)
