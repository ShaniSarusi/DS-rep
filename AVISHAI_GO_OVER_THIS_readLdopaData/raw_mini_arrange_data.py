# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:22:18 2017

@author: awagner
"""
from os import listdir
from os.path import isfile, join
import numpy as np
import datetime

mypath = 'C:/Users/awagner/Documents/Avishai_only_giti/By user/'
onlyfiles_raw = [f for f in listdir(mypath) if isfile(join(mypath, f))]
df_mini = []; count = -1
for files in onlyfiles_raw:
    count = count + 1
    print count
    df = read_export_tool_csv(mypath + files)
    df['Tag_This'] = files
    df_mini.append(df)


mypath = 'D:/Raw Data Minitrial/Raw Data and Symptoms Report - Mini Trial (Nov-Dec 2016)/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles_meta =  [s for s in onlyfiles if s.startswith('rdpd')]

count = -1
meta_mini = []
for files in onlyfiles_meta:
    count = count + 1
    print count
    df = pd.read_excel(mypath + files,skiprows = 9,parse_cols = 5)
    df = df.dropna(axis=0, how='all'); df = df[1:-1]
    df['Full Start Time'] = map(lambda date1,date2: date1.replace(hour = date2.hour, minute = date2.minute),df['Date (YYYY/MM/DD)'],df['Start time (HH:MM AM/PM)'])
    df['Full End Time'] = map(lambda date1,date2: date1.replace(hour = date2.hour, minute = date2.minute),df['Date (YYYY/MM/DD)'],df['End time (HH:MM AM/PM)'])    
    meta_mini.append(df)
    

def TagTheRaw(data_meta,data_raw):
    Tremor = np.repeat('UnKnown', len(data_raw['ts']))
    Dyskinesia = np.repeat('UnKnown', len(data_raw['ts']))
    Same_Task = np.repeat(0, len(data_raw['ts']))
    for This_Task in range(1,len(data_meta['Full Start Time'])):
        TS, TE = data_meta['Full Start Time'][This_Task], data_meta['Full End Time'][This_Task]   
        res = np.where((pd.to_datetime(data_raw['ts'])>TS) & (pd.to_datetime(data_raw['ts'])<TE))
        Tremor[res] = data_meta['Occurrence of tremor'][This_Task]
        Dyskinesia[res] = data_meta['Occurrence of dyskinesia'][This_Task]
        Same_Task[res] = This_Task        
    return Tremor, Dyskinesia, Same_Task    

cons = 0
for files in onlyfiles_raw:
    print files
    meta_file = [s for s in onlyfiles_meta if s.startswith(files[0:6])]
    data_meta = meta_mini[onlyfiles_meta.index(meta_file[0])]
    data_raw = df_mini[onlyfiles_raw.index(files)]
    trem, Dys, Same_Task  = TagTheRaw(data_meta,data_raw)
    data_raw['trem'] = trem; data_raw['dys'] = Dys; data_raw['Same_Task'] = Same_Task + cons
    df_mini[onlyfiles_raw.index(files)] = data_raw
    cons = Same_Task[:-1]

df_miniz = [];  df_miniy = []; df_minix = []; df_trem = [];  df_dys = []; df_same_task = [];
df_table = pd.concat(df_mini,ignore_index=True)
df_miniz = (VecToIntervals(df_table,np.shape(df_table)[0]/250,250,axis = 'z')) 
df_miniy = (VecToIntervals(df_table,np.shape(df_table)[0]/250,250,axis = 'y'))
df_minix = (VecToIntervals(df_table,np.shape(df_table)[0]/250,250,axis = 'x'))
df_trem = (VecToIntervals(df_table,np.shape(df_table)[0]/250,250,axis = 'trem',Read_Type = 'str'))
df_dys = (VecToIntervals(df_table,np.shape(df_table)[0]/250,250,axis = 'dys',Read_Type = 'str'))
df_same_task = (VecToIntervals(df_table,np.shape(df_table)[0]/250,250,axis = 'Same_Task',Read_Type = 'str'))
df_Tag_this = (VecToIntervals(df_table,np.shape(df_table)[0]/250,250,axis = 'Tag_This',Read_Type = 'str'))
df_ts = (VecToIntervals(df_table,np.shape(df_table)[0]/250,250,axis = 'ts',Read_Type = 'str'))


HR_mini = []
XYZ = np.stack((df_minix/1000.0,df_miniy/1000.0,df_miniz/1000.0),axis=2)
XYZ = np.reshape(XYZ,(np.shape(XYZ)[0],np.shape(XYZ)[1]*np.shape(XYZ)[2]))
HR_mini = (map(projGrav,XYZ))

#HR_table = (np.vstack(HR_mini))
hor_mini = [s[1] for s in HR_mini]#HR_table[:,1,:]
ver_mini = [s[0] for s in HR_mini]# HR_table[:,0,:]

trem_tag = np.vstack(df_trem); dys_tag = np.vstack(df_dys)
trem_tag = trem_tag[:,0]; dys_tag = dys_tag[:,0];

#Men_mini = df_table[:,4]

verDenoise_mini =  map(denoise2, ver_mini)
horDenoise_mini = map(denoise2, hor_mini)


print("Doing toDWT"
verWav = map(toDWT, verDenoise_mini )
print("relative wavelet"
relVer_Home = map(lambda x:contrib(x,rel=True), verWav)
print("cont wavelet"
contVer_Home =map(lambda x:contrib(x,rel=False), verWav)


print("Doing toDWT"
horWav = map(toDWT, horDenoise_mini )
print("relative wavelet"
relHor_Home = map(lambda x:contrib(x,rel=True), horWav)
print("cont wavelet"
contHor_Home =map(lambda x:contrib(x,rel=False), horWav)


Same_half_min = np.zeros(len(df_table['x']))

Keep_count = False; cur_start = 0;
count = 0
for cur_ind in range(len(df_table['x'])):
    if(Keep_count==False):
        print cur_ind
        count = count + 1
        cur_start = cur_ind
        Keep_count = True
    Same_half_min[cur_ind] = count
    if(np.abs(df_table['ts'][cur_ind] - df_table['ts'][cur_start])>datetime.timedelta(minutes=0.5) ):
        Keep_count = False

df_same_half_task = np.zeros(shape = (len(Same_half_min)/250,250))
for index in range(len(Same_half_min)/250):
        df_same_half_task[index] = Same_half_min[(250*index):(250*index+250)]
df_same_half_task =df_same_half_task[:,120] 
    
TagHome = np.column_stack((contVer_Home,relVer_Home,contHor_Home,relHor_Home))    
pred_home = svc.predict_proba(TagHome)
prob_home_per_Task = make_intervals_class(pred_home[:,1],df_same_half_task,trem_tag)
prob_home_per_Task  = pd.DataFrame(prob_home_per_Task )
prob_home_mat_unique = prob_home_per_Task.drop_duplicates()
#clf.fit(prob_home_mat_unique[range(3)], prob_home_mat_unique[3])
final_results = clf.predict_proba(prob_home_per_Task[range(3)])[:,1]
final_results = np.where(final_results>0.4,1,0)
#final_results_str = map(lambda x: str(x), final_results)
confusion_matrix(trem_tag,final_results_str)

trem_binary = np.where(trem_tag== 'None',0,np.where(trem_tag=='UnKnow',2,1))
confusion_matrix(trem_binary[3000:],final_results[3000:])


with open('my_dumped_classifier.pkl', 'wb') as fid:
     pickle.dump(svc, fid)