#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:32:16 2017

@author: HealthLOB
"""
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


user_number = 142594
ses_number = 1

my_data = read_from_s3('aws-athena-query-results-726895906399-us-west-2', 
'clinic_users/Tagging_files_clinic/user_' + str(user_number) + '_ses' + str(ses_number) + '.csv', 
'AKIAIEOL4GFG77QPNLCA', 
'06/OfU2vRMAkLGt69PjvVOSRfe1seABRQzhErL++' )

my_data['date'] = my_data['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))


#my_data['norma'] = my_data[['x', 'y', 'z']].apply(lambda row: np.sqrt(row['x']**2 + row['y']**2 + row['z']**2) , axis = 1)

np.unique(meta.Task)
date_border_up = my_data['date'][len(my_data['date']) - 1]
date_border_low = my_data['date'][0]

meta_pateint = meta.iloc[np.where((meta.user_id == user_number) & (meta['date_start'] > date_border_low) &
                                  (meta['date_end'] < date_border_up))]
meta_pateint = meta_pateint.sort('date_start')
num = 1
#num_task = np.where((meta_pateint.Task == task) & (meta_pateint.user_id == 142592))[0][num]

print(meta_pateint.iloc[num])

index = np.where((my_data['date'] > meta_pateint['date_start'].iloc[num ]-dt.timedelta(0,15)) & 
                 (my_data['date'] < meta_pateint['date_end'].iloc[num ]+dt.timedelta(0,15)))

index2 = np.where((my_data['date'] >= meta_pateint['date_start'].iloc[num]) & 
                 (my_data['date'] <= meta_pateint['date_end'].iloc[num ]))


print(num)
alpha = 0.5
plt.plot(range(len(index[0])),my_data.loc[index]['x'], alpha = alpha)
plt.plot(range(len(index[0])),my_data.loc[index]['y'], alpha = alpha)
plt.plot(range(len(index[0])),my_data.loc[index]['z'], alpha = alpha)
plt.xlabel('seconds')
plt.ylabel('accelerometer value')
currentAxis = plt.gca()
currentAxis.set_title(meta_pateint.Task.iloc[num])
currentAxis.set_ylim([-2,2])
plt.xticks(np.arange(0, len(index[0]), 1000.0))
currentAxis.add_patch(Rectangle((np.where(index[0] == index2[0][0])[0], -2), len(index2[0]), 4, fill=None, alpha=1, edgecolor ="red"))
plt.show()

curr = num + 1
while  meta_pateint.Tsstart.iloc[curr] ==  meta_pateint.Tsstart.iloc[num]:
    num = num + 1
    curr = num + 1
    
num = num + 1
    

