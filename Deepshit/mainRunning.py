# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 13:37:20 2017

@author: awagner
"""

import os
import sys
mother_path = os.path.abspath('PycharmProjects') + "/Deepshit/"
if(mother_path[2] == '\\' and mother_path[9] == 'a'):
    mother_path = 'C:/Users/awagner/Documents/Avishai_only_giti/PycharmProjects/Deepshit/'

sys.path.insert(0, mother_path)

import time
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.decomposition import PCA
import pywt
from Important_Function_For_Deep.xgboost_from_gilad import XGBoost_Classifier2
from multiprocessing import Pool
#XGBoost_Classifier2(learning_rate=0.025, n_estimators=50, max_depth=8, min_child_weight=5, gamma=0.6, subsample=0.75,
#                        colsample_bytree=1., scale_pos_weight=1)




Prepre_Data = False #If True Read the data and organize it
BuildDeepShit = True  #If True build the weights from zero

#Prepere the data
if Prepre_Data == True:    
    #execfile(mother_path + 'BeforeDeepLearning.py',globals())
    exec(open(mother_path + 'BeforeDeepLearning.py').read())
##Data for Deep Learning                        
data_for_model_Fourier =  np.stack((verFilter,horFilter ),axis=2)
data_for_model_raw_normlize =  np.stack((np.apply_along_axis(normlize_sig,1,verDenoise),np.apply_along_axis(normlize_sig,1,horDenoise) ),axis=2)
data_for_model_raw =  np.stack((verDenoise,horDenoise ),axis=2)


##Here you Calculate the weights
if BuildDeepShit == False: 
    decoded, encoded =BuildCNNNet2(np.shape(data_for_model_Fourier)[2],lost = 'binary_crossentropy')
    encoded.load_weights('encoded_weights.h5')

    decodedRaw, encodedRaw =BuildCNNNetRaw(np.shape(data_for_model)[2],lost = 'binary_crossentropy')
    encodedRaw.load_weights('encodedRaw_weights.h5')
    
    decodedRawMSE, encodedRawMSE =BuildCNNNetRaw(len(data_for_model[0]),lost = 'mse')
    encodedRawMSE.load_weights('encodedRawMSE_weights.h5')

else:
    decoded, encoded, last_layer =runnetwork(BuildCNNNet3, 10000,data = data_for_model_Fourier, num_of_epochs =3)
    decodedRaw, encodedRaw =runnetwork(BuildCNNNetRaw, 10000,data = data_for_model_raw_normlize)
    decodedRawMSE, encodedRawMSE =runnetwork(BuildCNNNetRaw, 10000,data = data_for_model_raw,lost = 'mse')
    decoded_LSTM, encoded_LSTM =runnetwork(LSTMCNN, 10000,data = data_for_model_Fourier)
 
 ##Here We build the tag data 
Tagdata = np.stack((TagVerFilter, TagHorFilter),axis=2)#,(np.shape(TagVerFilter)[0],2,np.shape(TagVerFilter)[1],1))
#encoded_home = encoded.predict(Tagdata)
Tagdeepfft = encoded.predict(Tagdata)
Tagdeepfft_LSTM = encoded_LSTM.predict(Tagdata)
Tagdata2 =  np.stack( np.stack((np.apply_along_axis(normlize_sig,1,TagverDenoise),np.apply_along_axis(normlize_sig,1,TaghorDenoise) )),axis=2)
TagdeepRaw = encodedRaw.predict(Tagdata2)
#TagdeepRaw  =  np.reshape(encoded_home,(np.shape(Tagdata2)[0],32))
Tagdata3 = np.stack((TagverDenoise,TaghorDenoise ),axis=2)
TagdeepRawMSE = encodedRaw.predict(Tagdata3)
TagLow = np.column_stack((Tagdeepfft_LSTM ,TagdeepRaw,TagdeepRawMSE,Tagdeepfft))
#xg = XGBoost_Classifier2(learning_rate=0.025, n_estimators=50, max_depth=12, min_child_weight=2, gamma=0.6, subsample=0.75,
#                colsample_bytree=1., scale_pos_weight=1)

#result = selectfeature(TagLow,trem,[1,2],Task,C=1,method = 'Forest')
#XGboost classifer
xg = XGBoost_Classifier2(learning_rate=0.025, n_estimators=50, max_depth=12, min_child_weight=2, gamma=0.6, subsample=0.75,
                        colsample_bytree=1., scale_pos_weight=3,reg_lambda1 = 0,reg_alpha1 = 0)               

pred_ALL = []
##is the sample in the cluster of the right activity 
clus = [1,2]
cond = np.asarray(map(lambda x: x in clus, Task))

print "computing probability vector for each interval"
for pateint in range(len(Men)):
    prob_per_pateint =  make_pred_withprob(cond, Men_number=pateint,data_for_pred = TagLow ,symp = Dys,meta_for_model = meta,model = xg)
    pred_ALL.append(prob_per_pateint[:,1])

pred_as_vector = np.asarray([item for sublist in pred_ALL for item in sublist] )
prob_mat_table = make_intervals_class(pred_as_vector,meta['TaskID'][cond==True],Dys[cond==True])
prob_mat_table = pd.DataFrame(prob_mat_table)
prob_mat_unique = prob_mat_table.drop_duplicates()

clf = LogisticRegression(C=1)
clf.fit(prob_mat_unique[range(3)], prob_mat_unique[3])

print "With Logistic Regression return "
tab= []
count = -1
for pateint in np.unique(meta.SubjectId):
    count = count + 1
    print(count)
    temp_vec = meta.SubjectId[cond==True].reset_index()
    final_results = clf.predict_proba(prob_mat_table[range(3)].loc[temp_vec.SubjectId == pateint])[:,1]
    final_results = np.where(final_results>0.3,1,0)
    tab.append(confusion_matrix(Dys[cond==True][temp_vec.SubjectId == pateint],final_results))
    print tab[count]
    

tabALL = pd.DataFrame(np.zeros((2,2)))
pred_total = np.zeros(1)
#colname = tabALL.columns.values
#rowname = list(tabALL.index)
for i in range(len(tab)): 
    temp = pd.DataFrame(tab[i])
    colname = temp.columns.values
    rowname = list(temp.index)
    for col in colname:
        for row in rowname:
            tabALL[row][col] = tabALL[row][col] + temp[row][col]
            
            
float(tabALL[1][1]+tabALL[0][0])/np.sum(np.sum(tabALL))
