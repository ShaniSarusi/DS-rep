#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:32:16 2017

@author: HealthLOB
"""

import os
os.chdir('C:/Users/awagner/Documents/DataScientists')

import pickle
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from Utils.Connections.connections import read_from_s3

pateint = 142561

meta = read_from_s3('aws-athena-query-results-726895906399-us-west-2', 'clinic_users/Musc_data/MUSC data/CSV uploaded to S3/F509_MotorTask_INTEL_20170925_V2.csv', 
'AKIAIEOL4GFG77QPNLCA', '06/OfU2vRMAkLGt69PjvVOSRfe1seABRQzhErL++', encoding = 'latin-1' )
meta = meta.dropna(axis = 0, how = 'any', thresh = 16)

my_data = read_from_s3('aws-athena-query-results-726895906399-us-west-2',
                       'clinic_users/Tagging_files_clinic/user_'+ str(pateint)+'_ses1.csv', 
'AKIAIEOL4GFG77QPNLCA', '06/OfU2vRMAkLGt69PjvVOSRfe1seABRQzhErL++' )

my_data['date'] = my_data['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
meta['date_start'] = meta['reported timestamp start'].apply(lambda x: dt.datetime.strptime(str(x), '%d%b%Y:%H:%M:%S.%f'))#'%m/%d/%Y %H:%M.%S'
meta['date_end'] = meta['reported timestamp end'].apply(lambda x: dt.datetime.strptime(str(x), '%d%b%Y:%H:%M:%S.%f'))



np.unique(meta['task name'])
date_border_up = my_data['date'][len(my_data['date']) - 1]
date_border_low = my_data['date'][0]

meta_pateint = meta.iloc[np.where((meta.user_id_intel == pateint) & (meta['date_start'] > date_border_low) &
                                  (meta['date_end'] < date_border_up))]
meta_pateint = meta_pateint.sort('date_start')
num = 1
#num_task = np.where((meta_pateint.Task == task) & (meta_pateint.user_id == 142592))[0][num]

print(meta_pateint.iloc[num])

index = np.where((my_data['date'] > meta_pateint['date_start'].iloc[num ]-dt.timedelta(0,10)) & 
                 (my_data['date'] < meta_pateint['date_end'].iloc[num ]+dt.timedelta(0,10)))

index2 = np.where((my_data['date'] >= meta_pateint['date_start'].iloc[num]) & 
                 (my_data['date'] <= meta_pateint['date_end'].iloc[num ]))


print(num)
alpha = 0.5
plt.plot(my_data.loc[index]['x'], alpha = alpha)
plt.plot(my_data.loc[index]['y'], alpha = alpha)
plt.plot(my_data.loc[index]['z'], alpha = alpha)
currentAxis = plt.gca()
currentAxis.set_title(meta_pateint['task name'].iloc[num])
currentAxis.set_ylim([-2,2])
currentAxis.add_patch(Rectangle((index2[0][0], -2), len(index2[0]), 4, fill=None, alpha=1, edgecolor ="red"))
plt.show()

curr = num + 1
while  meta_pateint.date_start.iloc[curr] ==  meta_pateint.date_start.iloc[num]:
    num = num + 1
    curr = num + 1
    
num = num + 1


####################
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 14:10:27 2017

@author: awagner
"""


