# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:22:18 2017

@author: awagner
"""
from os import listdir
from os.path import isfile, join
import numpy as np
from Utils.DataHandling.reading_files_and_directories import read_export_tool_csv


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

count = -1
for files in onlyfiles_raw:
    print files
    meta_file = [s for s in onlyfiles_meta if s.startswith(files[0:6])]
    data_meta = meta_mini[onlyfiles_meta.index(meta_file[0])]
    data_raw = df_mini[onlyfiles_raw.index(files)]
    trem, Dys, Same_Task  = TagTheRaw(data_meta,data_raw)
    data_raw['trem'] = trem; data_raw['dys'] = Dys; data_raw['Same_Task'] = Same_Task
    
    df_mini[onlyfiles_raw.index(files)] = data_raw

df_miniz = [];  df_miniy = []; df_minix = []; df_trem = [];  df_dys = []; df_same_task = [];
for i in range(len(df_mini)):
   df_miniz.append(VecToIntervals(df_mini[i],np.shape(df_mini[i])[0]/250,250,axis = 'z')) 
   df_miniy.append(VecToIntervals(df_mini[i],np.shape(df_mini[i])[0]/250,250,axis = 'y'))
   df_minix.append(VecToIntervals(df_mini[i],np.shape(df_mini[i])[0]/250,250,axis = 'x'))
   df_trem.append(VecToIntervals(df_mini[i],np.shape(df_mini[i])[0]/250,250,axis = 'trem',Read_Type = 'str'))
   df_dys.append(VecToIntervals(df_mini[i],np.shape(df_mini[i])[0]/250,250,axis = 'dys',Read_Type = 'str'))
   df_same_task.append(VecToIntervals(df_mini[i],np.shape(df_mini[i])[0]/250,250,axis = 'Same_Task',Read_Type = 'str'))


HR_mini = []
for i in range(len(df_miniz)):
    XYZ = np.stack((df_minix[i]/1000.0,df_miniy[i]/1000.0,df_miniz[i]/1000.0),axis=2)
    XYZ = np.reshape(XYZ,(np.shape(XYZ)[0],np.shape(XYZ)[1]*np.shape(XYZ)[2]))
    HR_mini.append(map(projGrav,XYZ))

HR_table = (np.vstack(HR_mini))
hor_mini = HR_table[:,1,:]
ver_mini = HR_table[:,0,:]

trem_tag = np.vstack(df_trem); dys_tag = np.vstack(df_dys)
trem_tag = trem_tag[:,0]; dys_tag = dys_tag[:,0];

df_table = np.vstack(df_mini)
Men_mini = df_table[:,4]

verDenoise_mini =  map(denoise2, ver_mini)
horDenoise_mini = map(denoise2, hor_mini)


print "Doing toDWT"
verWav = map(toDWT, verDenoise_mini )
print "relative wavelet"
relVer = map(lambda x:contrib(x,rel=True), verWav)
print "cont wavelet"
contVer =map(lambda x:contrib(x,rel=False), verWav)


print "Doing toDWT"
horWav = map(toDWT, horDenoise_mini )
print "relative wavelet"
relHor = map(lambda x:contrib(x,rel=True), horWav)
print "cont wavelet"
contHor =map(lambda x:contrib(x,rel=False), horWav)

    
    

