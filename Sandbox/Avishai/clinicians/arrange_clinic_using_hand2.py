# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:35:10 2017

@author: awagner
"""

pateint_63_hand_rotation = pd.read_csv('C:/Users/awagner/Documents/clinicion/hand_movement_files/user_142561.csv')

pateint_63_hand_rotation['date_start'] = pateint_63_hand_rotation['date_start'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
pateint_63_hand_rotation['date_end'] = pateint_63_hand_rotation['date_end'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

pateint_63_hand_rotation_only = pateint_63_hand_rotation[pateint_63_hand_rotation['task name'] == 'alternating left hand movements']



Tagging_63 = meta_pateint.copy()
Tagging_63['diss_from_hand'] = 0
Tagging_63['duration'] = 30
Tagging_63['cycle'] = 2

Tagging_63 = Tagging_63.reset_index()
          
count = 0


##############################
cycle = 3
my_diff_num = 11.05
end_time = 18
Tagging_63['diss_from_hand'].loc[Tagging_63.date_start == Tagging_63.date_start[count]] = np.floor(my_diff_num) * 60 + (my_diff_num*100)%100
Tagging_63['duration'].loc[Tagging_63.date_start == Tagging_63.date_start[count]] = end_time
Tagging_63['cycle'].loc[Tagging_63.date_start == Tagging_63.date_start[count]] = cycle

curr = count + 1
while  Tagging_63.date_start.iloc[curr] ==  Tagging_63.date_start.iloc[count]:
    count = count + 1
    curr = count + 1  
count = count + 1
print(Tagging_63[['diss_from_hand', 'task name', 'date_start', 'duration', 'cycle']].loc[count - 1])
###############################

keep_in_copy = Tagging_63.copy()
keep_in_copy = keep_in_copy.drop('index', 1)
keep_in_copy.to_csv('C:/Users/awagner/Documents/clinicion/Tags_from_VIDEO/user_63.csv')
##############################
##Add scale 2
test_data = pateint_60_hand_rotation_only.iloc[range(6)]
pateint_60_hand_rotation_only = pateint_60_hand_rotation_only.append(test_data, 
                                                        ignore_index=True)
pateint_60_hand_rotation_only = pateint_60_hand_rotation_only.sort('date_start')
###############################

diff_from_altenrat = Tagging_63['date_start'].copy()

only_alter = np.asarray(Tagging_63['diss_from_hand'])\
                       [np.asarray(Tagging_63['task']) == 'alternating right hand movements']
_, idx = np.unique(only_alter, return_index=True)
#idx = np.concatenate((idx, [30]))
alt_time = 0
numb_of_stand = 0
num_of_task = len(np.unique(Tagging_63['task']))
for i in range(len(diff_from_altenrat)):
    
    if(Tagging_63['task'].loc[i] == 'standing'):
       numb_of_stand = numb_of_stand + 1
       if numb_of_stand == 6:
           print(Tagging_63['task'].loc[i])
           print(i)
           numb_of_stand = 1
           alt_time = alt_time + 1
           print(alt_time)
           diff_from_altenrat[i] = pateint_63_hand_rotation_only['date_start'].iloc[6*alt_time]\
                             - dt.timedelta(0,int(only_alter[np.sort(idx)][alt_time]- Tagging_63['diss_from_hand'].iloc[i]))
           continue
    diff_from_altenrat[i] = pateint_63_hand_rotation_only['date_start'].iloc[6*alt_time]\
                      - dt.timedelta(0,int(only_alter[np.sort(idx)][alt_time] - Tagging_63['diss_from_hand'].iloc[i]))
                  
          
#################################################
Tagging_63['real_date_start'] = diff_from_altenrat
dur = np.asarray(Tagging_63['duration'].apply(lambda x: int(x)))
Tagging_63['real_end_date'] = lmap(lambda x: diff_from_altenrat.iloc[x]\
          + dt.timedelta(0, int(dur[x])), range(len(dur)))
      
Tagging_63.to_csv('C:/Users/awagner/Documents/clinicion/Tags_from_VIDEO/cisuabg7_tag.csv')

################################################

num = 0
#num_task = np.where((meta_pateint.Task == task) & (meta_pateint.user_id == 142592))[0][num]

print(Tagging_63.iloc[num])

index = np.where((my_data['date'] > Tagging_63['real_date_start'].iloc[num]-dt.timedelta(0,15)) & 
                 (my_data['date'] < Tagging_63['real_end_date'].iloc[num ]+dt.timedelta(0,15)))

index2 = np.where((my_data['date'] >= Tagging_63['real_date_start'].iloc[num]) & 
                 (my_data['date'] <= Tagging_63['real_end_date'].iloc[num ]))


print(num)
alpha = 0.5
plt.plot(range(len(my_data.loc[index]['x'])),my_data.loc[index]['x'], alpha = alpha)
plt.plot(range(len(my_data.loc[index]['x'])),my_data.loc[index]['y'], alpha = alpha)
plt.plot(range(len(my_data.loc[index]['x'])),my_data.loc[index]['z'], alpha = alpha)
currentAxis = plt.gca()
currentAxis.set_title(Tagging_63.task.iloc[num])
currentAxis.set_ylim([-2,2])
currentAxis.add_patch(Rectangle((750, -2), len(index2[0]), 4, fill=None, alpha=1, edgecolor ="red"))
plt.show()

curr = num + 1
while  Tagging_63.date_start.iloc[curr] ==  Tagging_63.date_start.iloc[num]:
    num = num + 1
    curr = num + 1
    
num = num + 1


