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
from LDopa.AugmentedData.augmented_data import augment_data

import Utils.Preprocessing.denoising as Denoiseing_func
#from DeepLearning.build_network import BuildCNNClassWithActivity
from LDopa.AugmentedData.augmented_data import augment_data
from Utils.Preprocessing.frequency_method import spectogram_and_normalize
from Utils.Preprocessing.other_utils import normalize_signal

data_path = '/home/lfaivish/PycharmProjects/Deepshit/DATA_FOLDER'
os.chdir(os.getcwd()+"/Documents/DataScientists")

XYZ = np.stack((lab_x ,lab_y,lab_z),axis=2)
XYZ = XYZ[cond==True]
task_clusters_for_deep = task_clusters[cond==True]

augment_XYZ, augment_dys, augment_Task, augment_SubjectId, augment_task_ids, augment_or_not = augment_data(XYZ, 
                                            labels, np.asarray(task_clusters_for_deep),
                                            np.asarray(tags_df.SubjectId[cond==True]), np.asarray(task_ids),
                                                              num_iter=20, num_iter_symp = 20)

x_denoise = lmap(lambda x: (Denoiseing_func.denoise(x)), augment_XYZ[:,:,0])
y_denoise = lmap(lambda x: (Denoiseing_func.denoise(x)), augment_XYZ[:,:,1])
z_denoise = lmap(lambda x: (Denoiseing_func.denoise(x)), augment_XYZ[:,:,2])

XYZ_denoise = np.stack((x_denoise, y_denoise, z_denoise), axis = 2)

TagLow = XYZ_denoise.copy()
augment_dys = augment_dys.reshape((len(augment_dys),1)); #augment_dys = utils.to_categorical(augment_dys, num_classes=2)
#augment_brady = augment_brady.reshape((len(augment_brady),1)); augment_brady = utils.to_categorical(augment_brady, num_classes=2)
#augment_tremor = augment_tremor.reshape((len(augment_tremor),1)); augment_tremor = utils.to_categorical(augment_tremor, num_classes=2)
meta_after_cond = np.asarray(augment_SubjectId)
Task_for_pred = np.where(augment_Task == 2, 0, 1)
Task_for_pred  = augment_Task.reshape((len(augment_Task),1))
sympNew = np.where(augment_dys == 0, 0, 0.25)
Task_for_predNew = np.where(Task_for_pred == 2, 0.5, 1)
Task_andSymp = sympNew + Task_for_pred
Task_andSymp = utils.to_categorical(Task_for_predNew, num_classes=4)
SubjectId_cat = utils.to_categorical(np.reshape(augment_SubjectId - 131, [len(augment_SubjectId), 1]), num_classes=19)

symp_class, feature_extract = BuildCNNClassWithActivity(TagLow.shape[1], 'binary_crossentropy')  


def scheduler(epoch=20):
    if epoch == 1:
        K.set_value(symp_class.optimizer.lr, 0.0001)
    if epoch == 2:
        K.set_value(symp_class.optimizer.lr, 0.0001)
    if epoch == 3:
        K.set_value(symp_class.optimizer.lr, 0.00008)
    if epoch == 5:
        K.set_value(symp_class.optimizer.lr, 0.00005)
    return K.get_value(symp_class.optimizer.lr)


deep_params = {'epochs': 10,
               'class_weight': {0 : 1,  1: 1},
               'change_lr': LearningRateScheduler(scheduler),
               'batch_size': 128}


labels_for_deep = augment_task_ids%3; 

symp = augment_dys.copy()#{'Dys': augment_dys, 'brady': augment_brady, 'trem': augment_tremor}

res1, order1, symp_res, feature_deep = make_cross_val(TagLow, augment_task_ids_bucket, symp, Task_andSymp, labels_for_deep, augment_task_ids,
                            symp_class,  feature_extract, augment_or_not, deep_params)

feature_deep1 = np.vstack(feature_deep)
feature_deep1 = np.column_stack((np.hstack(order1), feature_deep1))
feature_deep4 = feature_deep1[feature_deep1[:,0].argsort()]

plt.boxplot([res1[symp[test] == 1], res1[symp[test] == 0]])
confusion_matrix(np.vstack(symp_res), np.where(np.vstack(res1) > 0.75,1,0))

import gc
secret = None
gc.collect()

all_pred['prediction_probability'] = np.vstack(res1)
all_pred['true_label'] = np.vstack(symp_res).flatten()
all_pred['task'] = np.hstack(order1)
all_pred['patient'] = all_pred['task']%3
        
feature_deep2 = []
for i in order:
    for j in task_ids1:
        
    