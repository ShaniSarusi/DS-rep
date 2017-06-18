import pickle
import numpy as np
import csv
import pandas as pd
import random
import time
import os

##Read data from pickle file: (Avishai- 'C:/Users/awagner/Box Sync/Large_data/normaData')
def ReadfromPickle(path):
    pkl_data = open(path,'rb')
    normdata = pickle.load(pkl_data)
    normdata = np.reshape(normdata, [np.shape(normdata)[0] * np.shape(normdata)[1], np.shape(normdata)[2]])
    pkl_data.close()
    return normdata

#Function for reading data from CSV files:
def readfromcsv(string,path,head = None):
    Fullpath = path+string
    if(path == 'C:/Users/awagner/Documents/Avishai_only_giti/PycharmProjects/Deepshit/'):
        Fullpath = 'C:/Users/awagner/Box Sync/Large_data/' + string
    data_pandas = pd.read_csv(Fullpath,header = head)
    return data_pandas

#tagging the intervals - NOT SO RELEVANT FOR NOW
num_sessions = 35
num_of_intervals_per_session = 4750  # EACH interval has 250 time points. The input to the network is 126 since AW did fourier. Another option is to start with the 250 and not the 126 that result from the fourier.
patient = []  # not list, just empty to initialize
for i in range(num_sessions):
    patient = np.concatenate([patient,i*np.ones(num_of_intervals_per_session)]) # results in 0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3. etc. basically tags the intervals


#Read the raw tagged data:
x_raw_with_tag = readfromcsv('x_raw_with_tag.csv', data_path, head=None)
x_raw_with_tag = x_raw_with_tag.as_matrix()
x_raw_with_tag  =  x_raw_with_tag[1:,range(1,251)]

y_raw_with_tag= readfromcsv('y_raw_with_tag.csv', data_path, head=None)
y_raw_with_tag = y_raw_with_tag.as_matrix()
y_raw_with_tag  = y_raw_with_tag[1:,range(1,251)]

z_raw_with_tag = readfromcsv('z_raw_with_tag.csv', data_path, head=None)
z_raw_with_tag = z_raw_with_tag.as_matrix()
z_raw_with_tag  = z_raw_with_tag[1:,range(1,251)]


#Read the raw un-tagged data:
x_raw_no_tag = readfromcsv('x_raw_no_tag.csv', data_path)
x_raw_no_tag = x_raw_no_tag.as_matrix()
x_raw_no_tag = np.array(x_raw_no_tag, dtype = 'object')

y_raw_no_tag = readfromcsv('y_raw_no_tag.csv', data_path)
y_raw_no_tag = y_raw_no_tag.as_matrix()
y_raw_no_tag = np.array(y_raw_no_tag, dtype = 'object')

z_raw_no_tag = readfromcsv('z_raw_no_tag.csv', data_path)
z_raw_no_tag = z_raw_no_tag.as_matrix()
z_raw_no_tag = np.array(z_raw_no_tag,dtype = 'object')


#Read the 2-dimensional projections of the raw un-tagged data:
ver = readfromcsv('ver.csv', data_path)
ver = ver.as_matrix()

hor = readfromcsv('hor.csv', data_path)
hor = hor.as_matrix()



# next step
#normdata = readfromcsv('normadata.csv',mother_path,head = 'infer')
#normdata = normdata.as_matrix().T
#normdata = np.delete(normdata,0,0)  # delete first row. TODO it may be possible to read in the file without the first line.

#ver = readfromcsv('VerAfterFilter.csv',mother_path)
#ver = ver.as_matrix()

#hor = readfromcsv('HorAfterFilter.csv',mother_path)
#hor = hor.as_matrix()
