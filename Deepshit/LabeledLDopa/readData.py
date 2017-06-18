import pyodbc
import pandas as pd
import numpy as np


def close_connection(co):
    csr = co.cursor()
    csr.close()
    del csr
    co.close()


if __name__ == "__main__":
    dsn = "PAP_DEV_SMOOVE2"  # dsn = "ConnectSmoove2"
    lab_tagged_sessions_harvard = np.array(range(10100, 10136) + range(10137, 10143))  # ldhp 10136 is not good
    lab_tagged_sessions_mtsinai = np.array(range(10150, 10152) + range(10153, 10175))  # IntelUsername starts with ldms
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
    print 'query started'
    conn = pyodbc.connect("DSN=" + dsn, autocommit=True)
    res = pd.read_sql(query, conn)
    close_connection(conn)
    # res.to_csv('res.csv')

    print 'mapping string names started'
    # set types, string changes, and column name changes
    res[['TS', 'TSStart', 'TSEnd']] = res[['TS', 'TSStart', 'TSEnd']].apply(pd.to_datetime)
    res['AnnotationStrValue'] = res['AnnotationStrValue'].str.replace(' - .*', '')
    res.ix[(res.AnnotationStrValue.str.contains('Finger')) & (res.BradykinesiaGA.isnull()), ['AnnotationStrValue']] = 'Rest finger to nose'
    res.ix[(res.AnnotationStrValue.str.contains('Finger')) & (res.BradykinesiaGA.notnull()), ['AnnotationStrValue']] = 'Active finger to nose'
    res.ix[(res.AnnotationStrValue.str.contains('Alternating')) & (res.BradykinesiaGA.isnull()), ['AnnotationStrValue']] = 'Rest alternating hand movements'
    res.ix[(res.AnnotationStrValue.str.contains('Alternating')) & (res.BradykinesiaGA.notnull()), ['AnnotationStrValue']] = 'Active alternating hand movements'
    res.rename(columns={'AnnotationStrValue': 'Task'}, inplace=True)

    # calculate norm
    print 'calculating norm'
    res['Norm'] = (res.X**2 + res.Y**2 + res.Z**2)**0.5

    # Map Tasks to clusters
    print 'mapping tasks to clusters'
    map_t_c = pd.read_csv('C:\Users\zwaks\Documents\Workspaces\GitHub\DataScientists\LabeledLDopa\Resources\mapTasksClusters.csv')
    res = pd.merge(res, map_t_c, left_on=res.Task, right_on=map_t_c.Task, how='inner')
    res = res.drop('Task_y', 1)
    res.rename(columns={'Task_x': 'Task'}, inplace=True)

    # add task id
    print 'adding task ids - sorting'
    res = res.sort_values(['SessionId', 'TSStart'])
    res = res.reset_index(drop=True)
    res['TaskID'] = 1

    print 'adding task ids - copying data'
    tmp = res.copy()
    tmp = tmp.drop(['DeviceID', 'TS', 'X', 'Y', 'Z', 'Task', 'BradykinesiaGA', 'DyskinesiaGA', 'TremorGA', 'TSEnd',
                    'SubjectId', 'IntelUsername', 'Norm', 'TaskClusterId', 'TaskClusterName', 'TaskID'], axis=1)
    print 'adding task ids - dropping duplicates'
    st = tmp.drop_duplicates(keep='first').index.values
    st.sort()

    print 'adding task ids - starting loop'
    for i in range(1, st.shape[0]):
        print i
        res.loc[range(st[i-1], st[i]), 'TaskID'] = i
        time.sleep(0.01)
    res.loc[range(st[i], res.shape[0]), 'TaskID'] = i + 1

    # Create window (sample) start and end indices
    print 'Create window (sample) start and end indices'
    # Params - Set window values in seconds
    window_size = 5
    slide_by = 2.5
    trim_start = 1
    trim_end = 1
    frequency = 50

    # calculation
    raw = res.copy()
    raw = raw.drop(['SessionId', 'DeviceID', 'Task', 'BradykinesiaGA', 'DyskinesiaGA', 'TremorGA', 'SubjectId',
                    'TSStart', 'TSEnd', 'IntelUsername', 'TaskClusterId', 'TaskClusterName'], axis=1)
    raw = raw.sort_values('TS')

    win_idx = pd.DataFrame()
    win_idx['st'] = raw['TaskID'].drop_duplicates(keep='first').sort_values().index
    win_idx['en'] = raw['TaskID'].drop_duplicates(keep='last').sort_values().index
    win_idx['len'] = win_idx['en'] - win_idx['st'] + 1
    win_idx['dur'] = (win_idx['len']-1)/frequency
    tmp = np.ceil((win_idx['dur'] - trim_end - trim_end - window_size)/(window_size - slide_by))
    tmp[tmp < 0] = 0
    win_idx['num_samples'] = tmp.astype(np.int)
    tot_samples = sum(win_idx['num_samples'])

    # Map taskID to sampleID
    map_ids = pd.DataFrame(index=range(tot_samples), columns=['SampleID', 'TaskID'])
    k = 0
    for i in range(win_idx.shape[0]):
        print i
        for j in range(win_idx['num_samples'][i]):
            print j
            map_ids.loc[k, 'TaskID'] = i + 1
            map_ids.loc[k, 'SampleID'] = k + 1
            k += 1
    a = pd.merge(map_ids, res, on='TaskID')

    # 1 - metadata
    print 'creating metadata'
    meta = res.copy()
    meta = meta.drop(['TS', 'X', 'Y', 'Z', 'Norm'], axis=1)
    meta = meta.drop_duplicates()
    meta = pd.merge(map_ids, meta, on='TaskID')


    # get start/end idx
    print 'calculating window start and end indices'
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

    # 2 - raw data matrices - x,y,z,norm
    print 'creating raw data matrices'
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
        if i % 10 == 9:
            print 'adding sample ',i + 1, '(', (i+1)*100/tot_samples, '%) of ', tot_samples
        ran = range(samples_idx.loc[i, 'st'], samples_idx.loc[i, 'en']+1)
        xlist.append(raw.iloc[ran]['X'].values)
        ylist.append(raw.iloc[ran]['Y'].values)
        zlist.append(raw.iloc[ran]['Z'].values)
        nlist.append(raw.iloc[ran]['Norm'].values)
        # raw_x.loc[i] = raw.iloc[ran]['X'].values # this method is very slow - Pandas is not efficient at doing this
    print 'done adding samples'
    raw_x = pd.DataFrame(xlist);  raw_x['SampleID'] = meta['SampleID']
    raw_y = pd.DataFrame(ylist);  raw_y['SampleID'] = meta['SampleID']
    raw_z = pd.DataFrame(zlist);  raw_z['SampleID'] = meta['SampleID']
    raw_n = pd.DataFrame(nlist);  raw_n['SampleID'] = meta['SampleID']

    # save
    print 'saving output'
    meta.to_csv('metadata.csv')
    raw_x.to_csv('C:/Users/awagner/Box Sync/Large_data/rawX.csv')
    raw_y.to_csv('C:/Users/awagner/Box Sync/Large_data/rawY.csv')
    raw_z.to_csv('C:/Users/awagner/Box Sync/Large_data/rawZ.csv')
    raw_n.to_csv('C:/Users/awagner/Box Sync/Large_data/rawN.csv')

