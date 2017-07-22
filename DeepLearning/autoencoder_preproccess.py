#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 07:26:26 2017

@author: HealthLOB
"""

import pickle
import numpy as np

pkl_file = open('/home/lfaivish/PycharmProjects/Deepshit/DATA_FOLDER/unlabeld.pkl', 'rb')
unlabeld = pickle.load(pkl_file)
pkl_file.close()

unlabeld = np.stack([unlabeld[0],unlabeld[1],unlabeld[2]],axis=2)

TagLow = np.vstack([XYZ_denoise, unlabeld])
augment_dys = np.concatenate([augment_dys.flatten(), np.ones(len(unlabeld))])
augment_dys = augment_dys.reshape((len(augment_dys),1)); 
augment_task_ids = np.concatenate([augment_task_ids, np.random.randint(0,1,len(unlabeld))])
augment_or_not = np.concatenate([augment_or_not, np.zeros(len(unlabeld))])                                 
symp_class, feature_extract = BuildCNNClassWithAutoencoder(TagLow.shape[1], 'binary_crossentropy')  

home_or_not = np.concatenate([np.ones(len(XYZ_denoise)),np.zeros(len(unlabeld))])

                                 
 
def scheduler(epoch):
    if(epoch == 1):
        K.set_value(symp_class.optimizer.lr,0.001)
    if(epoch == 5):
        K.set_value(symp_class.optimizer.lr,0.0001)
    if(epoch == 10):
        K.set_value(symp_class.optimizer.lr,0.00005)
    if(epoch == 20):
        K.set_value(symp_class.optimizer.lr,0.00005)
    return K.get_value(symp_class.optimizer.lr)


deep_params = {'epochs': 10,
               'class_weight': {0 : 1,  1: 1},
               'change_lr': LearningRateScheduler(scheduler),
               'batch_size': 128}



labels_for_deep = augment_task_ids%3
labels_for_deep[range(len(XYZ_denoise),len(labels_for_deep))] = np.random.randint(0,3,len(unlabeld))

res1, order1, features_from_deep_res, symp_res = make_cross_val_with_auto(TagLow, home_or_not, augment_dys, labels_for_deep, augment_task_ids,
                            symp_class,  feature_extract, augment_or_not, deep_params)
