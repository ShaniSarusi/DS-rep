# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:23:40 2017

@author: awagner
"""
import os
import sys

'''
Set directory for the data:
'''
if(os.getcwd()=='C:\\Users\\imazeh'):
    mother_path = os.getcwd()+'/Itzik/Health_prof/git_team/DataScientists/Deepshit/'
    data_path = os.getcwd()+'/Itzik/Health_prof/L_dopa/Large_data/'

sys.path.insert(0, mother_path)
#print(sys.executable)

import time
import pandas
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pywt
#from Important_Function_For_Deep.xgboost_from_gilad import XGBoost_Classifier2
from multiprocessing import Pool
from future.utils import lmap


'''
Read the raw data (both tagged and un-tagged) and the un-tagged data projection to 2-dimensions:
'''
print("Organize the data...")
exec(open(mother_path + 'Important_Function_For_Deep/organizeData.py').read())

'''
Load necessary functions:
'''
exec(open(mother_path + 'Important_Function_For_Deep/FunctionForPredWithDEEP.py').read())
exec(open(mother_path + 'Important_Function_For_Deep/butterFilter.py').read())
exec(open(mother_path + 'Important_Function_For_Deep/making_pred_with_log_reg.py').read())
#exec(open(mother_path + 'Important_Function_For_Deep/BuildNetwork.py').read())


print("Read the meta data")
meta = readfromcsv('metadata.csv', data_path, head = 'infer')
Dys = meta.DyskinesiaGA.as_matrix()
trem = meta.TremorGA.as_matrix()
brady = meta.BradykinesiaGA.as_matrix()
Dys[Dys>0] = 1
brady[brady>0] = 1
trem[trem>0] = 1
Task = meta.TaskClusterId.as_matrix()


"""
Denoising the vertical and horizantal axis - for the untagged data:
"""
print("Denoising vertical and horizantal axis")
verDenoise =  lmap(denoise2, ver)
horDenoise = lmap(denoise2, hor)


"""
The normalized Fourier tranform - for the untagged data:
"""
print("The normalized Fourier tranform")
verFilter =lmap(absfft, verDenoise)
horFilter = lmap(absfft, horDenoise)


print("Men In List")
Men = []
for i in range(131,150):
    Men.append(np.where(meta.SubjectId == i))

print("Projection of tagged data to 2 dim")
XYZ = np.stack((x_raw_with_tag, y_raw_with_tag, z_raw_with_tag), axis=2)
XYZ = np.reshape(XYZ, (np.shape(XYZ)[0], np.shape(XYZ)[1]*np.shape(XYZ)[2]))
HR = lmap(projGrav, XYZ)

ver_proj = np.asarray([i[0] for i in HR])
hor_proj = np.asarray([i[1] for i in HR])

print("Denoise Tagged Data")
TagverDenoise = lmap(denoise2,  ver_proj)
TaghorDenoise = lmap(denoise2,  hor_proj)

print("Fourier Tagged Data")
TagVerFilter = lmap(absfft, TagverDenoise)
TagHorFilter = lmap(absfft, TaghorDenoise)

