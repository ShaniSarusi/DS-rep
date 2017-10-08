# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 10:43:46 2017

@author: awagner
"""

count = 0

my_diff_num = 14.37
end_time = 20
Tagging_60['diss_from_hand'].loc[Tagging_60.date_start == Tagging_60.date_start[count]] = np.floor(my_diff_num) * 60 + (my_diff_num*100)%100
Tagging_60['duration'].loc[Tagging_60.date_start == Tagging_60.date_start[count]] = end_time


curr = count + 1
while  Tagging_60.date_start.iloc[curr] ==  Tagging_60.date_start.iloc[count]:
    count = count + 1
    curr = count + 1
    
count = count + 1


Tagging_60[['diss_from_hand', 'task', 'date_start', 'duration']].loc[count - 1]


diff_from_altenrat = Tagging_60['date_start'].copy()

only_alter = np.asarray(Tagging_60['diff'])[np.asarray(Tagging_60['task']) == 'alternating left hand movements']
_, idx = np.unique(only_alter, return_index=True)
idx = np.concatenate((idx, [30]))
alt_time = 0
numb_of_stand = 0
num_of_task = len(np.unique(Tagging_60['task']))
for i in range(len(diff_from_altenrat)):
    
    if(Tagging_60['task'].loc[i] == 'standing'):
       numb_of_stand = numb_of_stand + 1
       if numb_of_stand == 6:
           print(Tagging_60['task'].loc[i])
           print(i)
           numb_of_stand = 1
           alt_time = alt_time + 1
           print(alt_time)
           diff_from_altenrat[i] = pateint_60_hand_rotation_only['date_start'].iloc[6*alt_time] - dt.timedelta(0,int(only_alter[np.sort(idx)][alt_time]))
           continue
    diff_from_altenrat[i] = pateint_60_hand_rotation_only['date_start'].iloc[6*alt_time] - dt.timedelta(0,int(only_alter[np.sort(idx)][alt_time] - Tagging_60['diff'].iloc[i]))
                  
          
