import pyodbc
import pandas as pd
import numpy as np
import pickle


##Read data from sql
##ses - session number to read from sql
def readDataFromSql(ses):
    conn = pyodbc.connect("DSN=PAP_DEV_SMOOVE2", autocommit=True)
    SessionSignal = "select * FROM[dbo].[ProcessedAccelerometerData] WHERE[SessionId] = " + str(ses) + " and [DeviceId] = 14 ORDER BY[TS]"
    df = pd.read_sql(SessionSignal, conn)
    return df

##VecToIntervals - Function that get vector and proccess it to intervals
##data - data frame returned from readDataFromSql
##num_intervals - number of intervals
##itervals_size - length of one interval
##axis - String, norma, X, Y or Z
def VecToIntervals(data,num_intervals,interval_size,axis = 'norma',Read_Type = 'int'):
    #norma = (data['X'] ** 2 + data['Y'] ** 2 + data['Z'] ** 2) ** 0.5
    IntervalMatrix = np.zeros(shape = (num_intervals,interval_size))
    if Read_Type == 'str':
        IntervalMatrix = np.chararray((num_intervals,interval_size),itemsize = 6)
        IntervalMatrix[:] = 'UNKOWN'
    print("hi")
    if axis != 'norma':
        norma = data[axis]
    print("hi2")
    for index in range(num_intervals):
        IntervalMatrix[index] = norma[(interval_size*index):(interval_size*index+interval_size)]
    #    
    return IntervalMatrix

mat_list = []

sessionid = np.concatenate(([11100,11102,11104,11105],range(11110,11132),range(11133,11142)))
for session in sessionid:
    print(session)
    df_test = readDataFromSql(session)
    interval_matrix = VecToIntervals(df_test, 4750, 250, axis = 'Z')
    mat_list.append(interval_matrix)

newFile = open('C:/Users/awagner/Box Sync/Large_data/zdata', 'wb')
#newFile = open('C:/Users/awagner/Box Sync/Large_data/normaData','wb')
pickle.dump(mat_list, newFile)

newFile.close()

