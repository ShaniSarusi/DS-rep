# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 15:34:56 2017

@author: awagner
"""


def chunk_it(seq, n, shuffle=False):
    """
    Accept an array and return a list of the array broken into chunks.

    Input:
        seq (array or list): The input array
        n (int): Number of chunks to break the array into
        shuffle (boolean): default false. If true, the sequence is randomly shuffled before breaking into chunks.

    Output:
        out1 (list of arrays): List of n equal size arrays
    """

    if shuffle:
        np.random.shuffle(seq)
    avg = int(len(seq) /(n))
    print(avg)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg/2
    return out


x_data = chunk_it(np.asarray(my_data.x), int(my_data.shape[0]/250), shuffle=False)
y_data = chunk_it(np.asarray(my_data.y), int(my_data.shape[0]/250), shuffle=False)
z_data = chunk_it(np.asarray(my_data.z), int(my_data.shape[0]/250), shuffle=False)
time_data = chunk_it(np.asarray(my_data.timestamp), int(my_data.shape[0]/250), shuffle=False)

del x_data[-1]; del x_data[-1]
del y_data[-1]; del y_data[-1]
del z_data[-1]; del z_data[-1]
del time_data[-1]; del time_data[-1]

x_data = np.vstack(x_data)
y_data = np.vstack(y_data)
z_data = np.vstack(z_data)
time_data = np.vstack(time_data)

XYZ = np.stack((x_data, y_data, z_data), axis = 2)

clinic_ver, clinic_hor = projections.project_from_3_to_2_dims(x_data, y_data, z_data)

clinic_ver_denoised = Denoiseing_func.denoise_signal(clinic_ver)
clinic_hor_denoised = Denoiseing_func.denoise_signal(clinic_hor)

WavFeatures = WavTransform.WavTransform()
clinic_x_features = WavFeatures.createWavFeatures(x_data)
clinic_y_features = WavFeatures.createWavFeatures(y_data)
clinic_z_features = WavFeatures.createWavFeatures(z_data)
features_clinic = np.column_stack((clinic_x_features, clinic_y_features, clinic_z_features))

pred_task = optimized_model.predict_proba(features_clinic)

meta_to_save = pd.DataFrame(columns = meta_pateint.columns)
tagg_vec = np.zeros(len(pred_task[:,4]))
for i in range(1, len(tagg_vec)):
    if(pred_task[i,3]>0.3):
        tagg_vec[i] = tagg_vec[i-1] + 1 


time_data_start = time_data[np.where(tagg_vec==2)][:,1]
time_data_end = time_data[np.where(tagg_vec==2)][:,249]

time_data_start =  lmap(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'),
                        time_data_start)

time_data_end =  lmap(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'),
                        time_data_end)

c = 0
index = np.where((my_data['date'] > time_data_start[c]-dt.timedelta(0,263)) & 
                 (my_data['date'] < time_data_end[c]-dt.timedelta(0,257)))

index2 = np.where((my_data['date'] >= time_data_start[c]) & 
                 (my_data['date'] <= time_data_end[c]))


alpha = 0.5
plt.plot(my_data.loc[index]['x'], alpha = alpha)
plt.plot(my_data.loc[index]['y'], alpha = alpha)
plt.plot(my_data.loc[index]['z'], alpha = alpha)
currentAxis = plt.gca()
currentAxis.set_title('fuck')
currentAxis.set_ylim([-2,2])
currentAxis.add_patch(Rectangle((index2[0][0], -2), len(index2[0]), 4, fill=None, alpha=1, edgecolor ="red"))
plt.show()

print(meta_pateint.task[((meta_pateint['date_start'] < time_data_start[c]+dt.timedelta(0,2)) & 
                 (meta_pateint['date_end'] > time_data_end[c]))].iloc[0])

meta_temp = meta_pateint[((meta_pateint['date_start'] < time_data_start[c]+dt.timedelta(0,5)) & 
                 (meta_pateint['date_end'] > time_data_end[c]))]

meta_temp = meta_pateint[((meta_pateint['date_start'] < meta_pateint['date_start'][meta_pateint.task == np.unique(meta_pateint.task)[1]].iloc[27]+dt.timedelta(0,5)) & 
                 (meta_pateint['date_end'] > meta_pateint['date_end'][meta_pateint.task == np.unique(meta_pateint.task)[1]].iloc[27] - dt.timedelta(0,5)))]

if meta_temp.empty:
    meta_temp = meta_to_save.iloc[[0,1,2,3,4,5]]  
    #meta_temp = meta_to_save.iloc[[6,7,8,9,10,11]]  
    meta_temp.score = np.NaN
    meta_temp['ON/OFF'] = np.NaN
    meta_temp['task'] = 'shake'

meta_temp['date_start'] = (time_data_start[c] - dt.timedelta(0, 263)).replace(microsecond=0)
meta_temp['date_end'] =  (meta_temp['date_start'] + dt.timedelta(0, 5))
meta_to_save = meta_to_save.append(meta_temp, ignore_index=True)
c = c+1

meta_to_save = meta_to_save.sort('date_start')
meta_to_save.to_csv('C:/Users/awagner/Documents/clinicion/hand_movement_files/user_' + str(pateint) + '.csv')