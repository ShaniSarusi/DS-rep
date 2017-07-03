#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 07:08:10 2017

@author: HealthLOB
"""
import os

import numpy as np
from future.utils import lmap
from keras import backend as K
from keras import utils
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import confusion_matrix

import Utils.Preprocessing.denoising as Denoiseing_func
from DeepLearning.build_network import BuildCNNClassWithActivity
from LDopa.AugmentedData.augmented_data import augment_data
from Utils.Preprocessing.frequency_method import spectogram_and_normlize
from Utils.Preprocessing.other_utils import normlize_sig

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

TagLow = XYZ_normlize.copy()
symp = augment_symp.reshape((len(augment_symp),1))
meta_after_cond = np.asarray(augment_SubjectId)
Task_for_pred = np.where(augment_Task==2,0,1)
Task_for_pred  = Task_for_pred.reshape((len(Task_for_pred),1))
sympNew = np.where(symp==0 , 0, 0.25)
Task_for_predNew = np.where(Task_for_pred==1 , 0.5, 1)
Task_andSymp = sympNew + Task_for_predNew 
Task_andSymp = utils.to_categorical(Task_andSymp, num_classes=4)

symp_class, feature_extract = BuildCNNClassWithActivity(TagLow.shape[1], 'binary_crossentropy')  

def scheduler(epoch):
    if(epoch == 1):
        K.set_value(symp_class.optimizer.lr,0.0005)
    if(epoch == 5):
        K.set_value(symp_class.optimizer.lr,0.0001)
    if(epoch == 10):
        K.set_value(symp_class.optimizer.lr,0.0001)
    if(epoch == 20):
        K.set_value(symp_class.optimizer.lr,0.0001)
    return K.get_value(symp_class.optimizer.lr)


deep_params = {'epochs': 30,
               'class_weight': {0 : 1,  1: 1},
               'change_lr': LearningRateScheduler(scheduler)}

Group1 = [131,134,137,140,143,146,149]
Group2 = [132,135,138,141,144,147]
Group3 = [133,136,139,142,145,148]



res1, order1 = make_cross_val(TagLow, symp, Task_andSymp, np.concatenate([np.zeros(75070),np.ones(75070)]), 
                            symp_class,  feature_extract, augment_or_not, deep_params)
results = np.asarray([x for (y,x) in sorted(zip(order1,res1))])

plt.boxplot([res1[symp[test] == 1],res1[symp[test] == 0]])
confusion_matrix(labels ,np.where(np.asarray(results) > 0.7,1,0))

import gc
secret = None
gc.collect()

