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
from Utils.Preprocessing.frequency_method import spectogram_and_normalize
from Utils.Preprocessing.other_utils import normlize_sig

data_path = '/home/lfaivish/PycharmProjects/Deepshit/DATA_FOLDER'
os.chdir(os.getcwd()+"/Documents/DataScientists")

XYZ = np.stack((lab_x ,lab_y,lab_z),axis=2)
XYZ = XYZ[cond==True]
task_clusters_for_deep = task_clusters[cond==True]

augment_XYZ, augment_dys, augment_Task, augment_SubjectId, augment_task_ids, augment_or_not = augment_data(XYZ, 
                                            labels, np.asarray(task_clusters_for_deep), np.asarray(tags_df.SubjectId[cond==True]), np.asarray(task_ids))

x_denoise = lmap(Denoiseing_func.denoise, augment_XYZ[:, :, 0])
y_denoise = lmap(Denoiseing_func.denoise, augment_XYZ[:, :, 1])
z_denoise = lmap(Denoiseing_func.denoise, augment_XYZ[:, :, 2])

XYZ_denoise = np.stack((x_denoise, y_denoise, z_denoise), axis=2)

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
Task_andSymp = utils.to_categorical(Task_andSymp, num_classes=6)
SubjectId_cat = utils.to_categorical(np.reshape(augment_SubjectId - 131, [len(augment_SubjectId), 1]), num_classes=19)

symp_class, feature_extract = BuildCNNClassWithActivity(TagLow.shape[1], 'binary_crossentropy')  


def scheduler(epoch=10):
    if epoch == 1:
        K.set_value(symp_class.optimizer.lr, 0.0001)
    if epoch == 5:
        K.set_value(symp_class.optimizer.lr, 0.00005)
    if epoch == 10:
        K.set_value(symp_class.optimizer.lr, 0.00005)
    if epoch == 20:
        K.set_value(symp_class.optimizer.lr, 0.00005)
    return K.get_value(symp_class.optimizer.lr)


deep_params = {'epochs': 100,
               'class_weight': {0 : 1,  1: 1},
               'change_lr': LearningRateScheduler(scheduler),
               'batch_size': 128}


labels_for_deep = augment_task_ids%3; 

symp = augment_dys.copy()#{'Dys': augment_dys, 'brady': augment_brady, 'trem': augment_tremor}

res1, order1, symp_res = make_cross_val(TagLow, symp, Task_andSymp, labels_for_deep, augment_task_ids,
                            symp_class,  SubjectId_cat, augment_or_not, deep_params)
results = np.asarray([x for (y,x) in sorted(zip(order1,res1))])
results_feature = np.asarray([x for (y,x) in sorted(zip(order1,features_from_deep_res1))])

plt.boxplot([res1[symp[test] == 1], res1[symp[test] == 0]])
confusion_matrix(np.vstack(symp_res), np.where(np.vstack(res1) > 0.75,1,0))

import gc
secret = None
gc.collect()

all_pred['prediction_probability'] = np.vstack(res1)
all_pred['true_label'] = np.vstack(symp_res).flatten()
all_pred['task'] = order1.copy()
all_pred['patient'] = order1%3