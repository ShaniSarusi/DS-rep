#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 07:08:10 2017

@author: HealthLOB
"""
import os
import numpy as np
from future.utils import lmap
import datetime as dt
import pandas as pd
import time
from scipy.stats import ortho_group
from sklearn.model_selection import LeaveOneGroupOut
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras import utils
import Utils.Preprocessing.denoising as Denoiseing_func
from Utils.Preprocessing.other_utils import normlize_sig
from LDopa.AugmentedData.augmented_data import augment_data
from Utils.FrequencyMethods.FrequencyMethod import spectogram_and_normlize
from DeepLearning.build_network import BuildCNNClassWithActivity
data_path = '/home/lfaivish/PycharmProjects/Deepshit/DATA_FOLDER'
os.chdir(os.getcwd()+"/Documents/DataScientists")

XYZ = np.stack((lab_x ,lab_y,lab_z),axis=2)
XYZ = XYZ[cond==True]
task_clusters_for_deep = task_clusters[cond==True]

augment_XYZ, augment_symp, augment_Task, augment_SubjectId, augment_or_not = augment_data(XYZ, 
                                            labels, np.asarray(task_clusters_for_deep), np.asarray(patients))

x_denoise = lmap(Denoiseing_func.denoise, augment_XYZ[:, :, 0])
y_denoise = lmap(Denoiseing_func.denoise, augment_XYZ[:, :, 1])
z_denoise = lmap(Denoiseing_func.denoise, augment_XYZ[:, :, 2])

x_fft = lmap(spectogram_and_normlize, x_denoise)
y_fft = lmap(spectogram_and_normlize, y_denoise)
z_fft = lmap(spectogram_and_normlize, z_denoise)

x_normlize = lmap(normlize_sig, x_denoise)  # augment_XYZ[:,:,0] )#
y_normlize = lmap(normlize_sig, y_denoise)  # augment_XYZ[:,:,1] )#
z_normlize = lmap(normlize_sig, z_denoise)  # augment_XYZ[:,:,2] )#

x_normlize_real = lmap(normlize_sig, augment_XYZ[:, :, 0])  # augment_XYZ[:,:,0] )#
y_normlize_real = lmap(normlize_sig, augment_XYZ[:, :, 1])  # augment_XYZ[:,:,1] )#
z_normlize_real = lmap(normlize_sig, augment_XYZ[:, :, 2])  # augment_XYZ[:,:,2] )#


XYZ_denoise = np.stack((x_denoise, y_denoise, z_denoise), axis=2)
XYZ_normlize = np.stack((x_normlize, y_normlize, z_normlize), axis=2)
XYZ_normlize_real = np.stack((x_normlize_real, y_normlize_real, z_normlize_real), axis=2)
XYZ_normlize_fft = np.stack((x_fft, y_fft, z_fft), axis=2)

TagLow = XYZ_normlize_fft.copy()
symp = augment_symp.reshape((len(augment_symp),1))
meta_after_cond = np.asarray(augment_SubjectId)
Task_for_pred = np.where(augment_Task==2,0,1)
Task_for_pred  = Task_for_pred.reshape((len(Task_for_pred),1))
sympNew = np.where(symp==0 , 0, 0.25)
Task_for_predNew = np.where(Task_for_pred==1 , 0.5, 1)
Task_andSymp = sympNew + Task_for_predNew 
Task_andSymp = utils.to_categorical(Task_andSymp, num_classes=4)

symp_class, feature_extract = BuildCNNClassWithActivity(250, 'binary_crossentropy')  
weights_start_symp = symp_class.get_weights()

logo = LeaveOneGroupOut()
cv = logo.split(TagLow, symp, meta_after_cond)##np.random.randint(0,4,len(symp)
cv1 = list(cv)
cv = list(cv1)

#earlyStopping= EarlyStopping(monitor='loss', min_delta = 0.002,patience=3, verbose=0, mode='auto')


def scheduler(epoch):
    if(epoch == 1):
        K.set_value(symp_class.optimizer.lr,0.0001)
    if(epoch == 5):
        K.set_value(symp_class.optimizer.lr,0.00005)
    if(epoch == 10):
        K.set_value(symp_class.optimizer.lr,0.00005)
    if(epoch == 20):
        K.set_value(symp_class.optimizer.lr,0.00005)
    return K.get_value(symp_class.optimizer.lr)

change_lr = LearningRateScheduler(scheduler)
res = []
symp_cor_res = []
features_from_deep = [];
class_weight = {0 : 1,  1: 1}
for train, test in cv:
    print(test)
    symp_class.set_weights(weights_start_symp)
    xtest = TagLow[test]
    xtrain = TagLow[train]
    #weight_sample = np.abs(1 - Task_for_pred[train])
    symp_class.fit(xtrain, [symp[train],Task_andSymp[train]],epochs=50 , batch_size=128, shuffle=True, validation_data=(xtest, [symp[test],Task_andSymp[test]]),callbacks=[change_lr],class_weight =class_weight  ,verbose=2)#
    temp_res = symp_class.predict(xtest[augment_or_not[test] == 1])
    res.append(temp_res[0])
    symp_cor_res.append(symp[test])
    features_from_deep.append(feature_extract.predict(xtest[augment_or_not_after[test] == 1]))
res1 = np.vstack(res)
symp_cor_res1 = np.vstack(symp_cor_res)
plt.boxplot([res[symp[test] == 1],res[symp[test] == 0]])
confusion_matrix(labels ,np.where(res1>0.8,1,0))
import gc
gc.collect()


res1 = make_cross_val(TagLow, symp, Task_andSymp, np.random.randint(0,4,len(symp)), 
                            symp_class,  feature_extract,change_lr, class_weight, augment_or_not)