# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 09:21:24 2017

@author: awagner
"""

import pyodbc
import pandas as pd
import numpy as np
import time
from Utils.Connections.connections import close_connection


# dsn = "ConnectSmoove2"
'''
Give you dsn (For example ConnectSmoove2) and read ALL of the labeled data
'''


def read_all_data(dsn):
    """
    Read data from sql using dsn you have
    
    Input:
        dsn (String): SQL dsn
    Output:
        res (numpy): A data set contain SessionId samples of X, Y, Z, task, userID, TSStart and TSEnd
                     per timestamp.
    """
    lab_tagged_sessions_harvard = np.concatenate([range(10100, 10136), range(10137, 10143)])  # ldhp 10136 is not good
    lab_tagged_sessions_mtsinai = np.concatenate([range(10150, 10152), range(10153, 10175)])  # IntelUsername starts with ldms
    session_ids = np.concatenate([lab_tagged_sessions_harvard, lab_tagged_sessions_mtsinai])
    # session_ids = np.array([10100])
    session_ids = lab_tagged_sessions_harvard

    ids_str = ','.join(['%d' % x for x in session_ids])
    text = ("select  acc.SessionId, acc.DeviceID, acc.TS, acc.X, acc.Y, acc.Z,  ann.AnnotationStrValue, "
            "ld.BradykinesiaGA, ld.DyskinesiaGA, ld.TremorGA, ld.TSStart, ld.TSEnd, ld.SubjectId, ld.IntelUsername "
            "FROM [dbo].[ProcessedAccelerometerData] as acc, [dbo].[SessionAnnotations] as ann, "
            "[dbo].[LDOPA_DataPerTask_VW] as ld where (acc.[SessionId] in (%s)) and (acc.DeviceId = 14) and "
            "(acc.SessionId = ann.SessionId) and (ann.SessionId = ld.SessionId) and (AnnotationName=\'Activity\') and "
            "(TS >= ann.TSStart) and (TS <= ann.TSEND) and (TS >= ld.TSStart) and (TS <= ld.TSEnd) ORDER BY [TS]")

    query = text % ids_str
    print('query started')
    conn = pyodbc.connect("DSN=" + dsn, autocommit=True)
    res = pd.read_sql(query, conn)
    close_connection(conn)
    return res


def arrange_res(res, path):
    """
    Arrange our dataset in new format
    
    Input:
        res (numpy)- A data set contain SessionId samples of X, Y, Z, task, userID, task cluster, TSStart and TSEnd
              per timestamp.
        path (string)- Path of the clusters per task
    Output:
        res - Organize dataset res with TaskID (Same Inteval TimeStamp) and tasks clusters
    
    """
    res[['TS', 'TSStart', 'TSEnd']] = res[['TS', 'TSStart', 'TSEnd']].apply(pd.to_datetime)
    res['AnnotationStrValue'] = res['AnnotationStrValue'].str.replace(' - .*', '')
    res.ix[(res.AnnotationStrValue.str.contains('Finger')) & (res.BradykinesiaGA.isnull()), 
           ['AnnotationStrValue']] = 'Rest finger to nose'
    res.ix[(res.AnnotationStrValue.str.contains('Finger')) & (res.BradykinesiaGA.notnull()), 
           ['AnnotationStrValue']] = 'Active finger to nose'
    res.ix[(res.AnnotationStrValue.str.contains('Alternating')) & (res.BradykinesiaGA.isnull()), 
           ['AnnotationStrValue']] = 'Rest alternating hand movements'
    res.ix[(res.AnnotationStrValue.str.contains('Alternating')) & (res.BradykinesiaGA.notnull()), 
           ['AnnotationStrValue']] = 'Active alternating hand movements'
    res.rename(columns={'AnnotationStrValue': 'Task'}, inplace=True)

    # Calculate norm
    print('calculating norm')
    res['Norm'] = (res.X**2 + res.Y**2 + res.Z**2)**0.5

    # Map tasks to clusters
    print('mapping tasks to clusters')
    map_t_c = pd.read_csv(path)
    res = pd.merge(res, map_t_c, left_on=res.Task, right_on=map_t_c.Task,
                   how='inner')
    res = res.drop('Task_y', 1)
    res.rename(columns={'Task_x': 'Task'}, inplace=True)

    # Add task id
    print('adding task ids - sorting')
    res = res.sort_values(['SessionId', 'TSStart'])
    res = res.reset_index(drop=True)
    res['TaskID'] = 1

    print('adding task ids - copying data')
    tmp = res.copy()
    tmp = tmp.drop(['DeviceID', 'TS', 'X', 'Y', 'Z', 'Task', 'BradykinesiaGA', 
                    'DyskinesiaGA', 'TremorGA', 'TSEnd',
                    'SubjectId', 'IntelUsername', 'Norm', 'TaskClusterId', 
                    'TaskClusterName', 'TaskID'], axis=1)
    print('adding task ids - dropping duplicates')
    st = tmp.drop_duplicates(keep='first').index.values
    st.sort()

    print('adding task ids - starting loop')
    for i in range(1, st.shape[0]):
        print(i)
        res['TaskID'].loc[range(st[i-1], st[i])] = i
        time.sleep(0.01)
    res['TaskID'].loc[range(st[i], res.shape[0])] = i + 1
    return res


def make_interval_from_all_data(res, window_size, slide_by, trim_start, trim_end,frequency):
    """
    Get dataset per timestamp and reorganize as intervals
    
    Input:
        res (numpy): The output of arrange_res function
        window_size (float): length of intervals in seconds
        slide_by (float): Next Inteval start time - This Inteval start time in seconds
        trim_start (float): For every task trim the start in seconds
        trim_start (float): For every task trim the end in seconds
        frequency (float): Samples frequency in Hertz
        
    Output:
        meta (pandas) - A dataFrame with SessionId', 'DeviceID', 'Task', 'BradykinesiaGA',
             'DyskinesiaGA', 'TremorGA', 'SubjectId','TSStart', 'TSEnd', 'IntelUsername', 
             'TaskClusterId', 'TaskClusterName
        raw_x (pandas)- X samples in intevals
        raw_y (pandas)- Y samples in intevals
        raw_z (pandas)- Z samples in intevals
    """
    raw = res.copy()
    raw = raw.drop(['SessionId', 'DeviceID', 'Task', 'BradykinesiaGA', 'DyskinesiaGA', 
    'TremorGA', 'SubjectId',
                    'TSStart', 'TSEnd', 'IntelUsername', 
                    'TaskClusterId', 'TaskClusterName'], axis=1)
    raw = raw.sort_values('TS')

    win_idx = pd.DataFrame()
    win_idx['st'] = raw['TaskID'].drop_duplicates(keep='first').sort_values().index + trim_start*frequency
    win_idx['en'] = raw['TaskID'].drop_duplicates(keep='last').sort_values().index - trim_end*frequency
    win_idx['len'] = win_idx['en'] - win_idx['st'] + 1
    win_idx['dur'] = (win_idx['len']-1)/frequency
    tmp = np.ceil((win_idx['dur'] - trim_start - trim_end - window_size)/slide_by)
    tmp[tmp < 0] = 0
    win_idx['num_samples'] = tmp.astype(np.int)
    tot_samples = sum(win_idx['num_samples'])

    map_ids = pd.DataFrame(index=range(tot_samples), columns=['SampleID', 'TaskID'])
    k = 0
    for i in range(win_idx.shape[0]):
        print(i)
        for j in range(win_idx['num_samples'][i]):
            # print(j)
            map_ids.loc[k, 'TaskID'] = i + 1
            map_ids.loc[k, 'SampleID'] = k + 1
            k += 1
    # a = pd.merge(map_ids, res, on='TaskID')

    print('calculating window start and end indices')
    samples_idx = pd.DataFrame(index=range(tot_samples), columns=['st', 'en'])
    k = 0
    gap = np.int(slide_by*frequency)
    win = np.int(window_size*frequency)
    for i in range(win_idx.shape[0]):
        for j in range(int(win_idx.loc[i, 'num_samples'])):
            samples_idx.loc[k, 'st'] = win_idx.loc[i, 'st'] + j*gap
            samples_idx.loc[k, 'en'] = win_idx.loc[i, 'st'] + win + j*gap - 1
            k += 1
    samples_idx.sort_values(by='st')
    samples_idx[['st', 'en']] = samples_idx[['st', 'en']].astype(int)

    meta = res.copy()
    meta = meta.drop(['TS', 'X', 'Y', 'Z', 'Norm'], axis=1)
    meta = meta.drop_duplicates()
    meta = pd.merge(map_ids, meta, on='TaskID')

    print('creating raw data matrices')
    raw_x = pd.DataFrame(index=range(tot_samples), columns=range(win))
    raw_y = pd.DataFrame(index=range(tot_samples), columns=range(win))
    raw_z = pd.DataFrame(index=range(tot_samples), columns=range(win))
    raw_n = pd.DataFrame(index=range(tot_samples), columns=range(win))

    xlist = []
    ylist = []
    zlist = []
    nlist = []
    for i in range(tot_samples):
    # print 'adding sample ', i + 1, '(', (i + 1) * 100 / tot_samples, '%) of ', tot_samples
        if i % 10 == 90:
            print('adding sample ', i + 1, '(', (i+1)*100/tot_samples, '%) of ', tot_samples)
        ran = range(samples_idx.loc[i, 'st'], samples_idx.loc[i, 'en']+1)
        xlist.append(raw.iloc[ran]['X'].values)
        ylist.append(raw.iloc[ran]['Y'].values)
        zlist.append(raw.iloc[ran]['Z'].values)
        nlist.append(raw.iloc[ran]['Norm'].values)
    # raw_x.loc[i] = raw.iloc[ran]['X'].values # this method is very slow - Pandas is not efficient at doing this
    print('done adding samples')
    raw_x = pd.DataFrame(xlist)
    raw_x['SampleID'] = meta['SampleID']
    raw_y = pd.DataFrame(ylist)
    raw_y['SampleID'] = meta['SampleID']
    raw_z = pd.DataFrame(zlist)
    raw_z['SampleID'] = meta['SampleID']
    raw_n = pd.DataFrame(nlist)
    raw_n['SampleID'] = meta['SampleID']

    return [meta, raw_x, raw_y, raw_z, raw_n]


def read_unlabeled_data(sec, freq, session_ids, dsn):
    """
    Read unlabeld data from sql
    
    Input:
        sec (Integer)- seconds of intevals
        freq (Integer) - frequency of data
        session_ids (list of integers) - list of session you want to read
        dsn (String) - your dsn for SQL
    
    Output:
        X_table (2d numpy array) - Intevals are rows for X 
        Y_table (2d numpy array) - Intevals are rows for Y 
        Z_table (2d numpy array) - Intevals are rows for Z 
    """
    X_table_ses = []
    Y_table_ses = []
    Z_table_ses = []
    for ses in session_ids:
        # ids_str = ','.join(['%d' % x for x in session_ids])
        print(ses)
        text = ("select  acc.SessionId, acc.DeviceID, acc.X, acc.Y, acc.Z "
                "FROM [dbo].[ProcessedAccelerometerData] as acc "
                "where (acc.[SessionId] in (%s)) and (acc.DeviceId = 14) "
                "ORDER BY [TS]")
        query = text % ses
        conn = pyodbc.connect("DSN=" + dsn, autocommit=True)
        res_home = pd.read_sql(query, conn)
        close_connection(conn)

        raw_home_x = np.ones((res_home.shape[0]/(sec*freq), sec*freq))
        raw_home_y = np.zeros((res_home.shape[0]/(sec*freq), sec*freq))
        raw_home_z = np.zeros((res_home.shape[0]/(sec*freq), sec*freq))
        for i in range(int(np.floor(res_home.shape[0]/(sec*freq)))):
            # print(i)
            raw_home_x[i] = np.asarray(res_home['X'][i*sec*freq:(i+1)*sec*freq])
            raw_home_y[i] = np.asarray(res_home['Y'][i*sec*freq:(i+1)*sec*freq])
            raw_home_z[i] = np.asarray(res_home['Z'][i*sec*freq:(i+1)*sec*freq])
        X_table_ses.append(raw_home_x)
        Y_table_ses.append(raw_home_y)
        Z_table_ses.append(raw_home_z)

    X_table = np.vstack(X_table_ses)
    Y_table = np.vstack(Y_table_ses)
    Z_table = np.vstack(Z_table_ses)

    return X_table, Y_table, Z_table


