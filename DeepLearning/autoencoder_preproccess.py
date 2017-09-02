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


x_unlabeld = lmap(lambda x: (Denoiseing_func.denoise(x)), unlabeld[0])
y_unlabeld = lmap(lambda x: (Denoiseing_func.denoise(x)), unlabeld[1])
z_unlabeld = lmap(lambda x: (Denoiseing_func.denoise(x)), unlabeld[2])

unlabeld = np.stack([x_unlabeld,y_unlabeld ,z_unlabeld ],axis=2)

WavFeatures = WavTransform.WavTransform()
lab_x_features = WavFeatures.createWavFeatures(x_denoise)
lab_y_features = WavFeatures.createWavFeatures(y_denoise)
lab_z_features = WavFeatures.createWavFeatures(z_denoise)

features_deep_data = np.column_stack((lab_x_features, lab_y_features, lab_z_features))

home_x_features = WavFeatures.createWavFeatures(x_unlabeld )
home_y_features = WavFeatures.createWavFeatures(y_unlabeld )
home_z_features = WavFeatures.createWavFeatures(z_unlabeld )

features_unlabeld_data = np.column_stack((home_x_features, home_y_features, home_z_features))

features_autoencoder = np.concatenate([features_deep_data, features_unlabeld_data])

TagLow = np.vstack([XYZ_denoise, unlabeld])
augment_dys = np.concatenate([augment_dys.flatten(), np.ones(len(unlabeld))])
augment_dys = augment_dys.reshape((len(augment_dys),1)); 
augment_task_ids = np.concatenate([augment_task_ids, np.random.randint(0,1,len(unlabeld))])
augment_or_not = np.concatenate([augment_or_not, np.zeros(len(unlabeld))])                                 



symp_class, feature_extract = BuildCNNClassWithAutoencoder(TagLow.shape[1], 'mse')  

home_or_not = np.concatenate([np.ones(len(XYZ_denoise)),np.zeros(len(unlabeld))])

home_or_not = home_or_not.reshape((len(home_or_not),1))
                             
 
def scheduler(epoch):
    if(epoch == 1):
        K.set_value(symp_class.optimizer.lr,0.01)
    if(epoch == 2):
        K.set_value(symp_class.optimizer.lr,0.01)
    if(epoch == 3):
        K.set_value(symp_class.optimizer.lr,0.01)
    if(epoch == 4):
        K.set_value(symp_class.optimizer.lr,0.005)
    if(epoch == 5):
        K.set_value(symp_class.optimizer.lr,0.005)
    if(epoch == 15):
        K.set_value(symp_class.optimizer.lr, 0.005)
    return K.get_value(symp_class.optimizer.lr)


deep_params = {'epochs': 40,
               'class_weight': {0 : 1,  1: 1},
               'change_lr': LearningRateScheduler(scheduler),
               'batch_size': 128}



labels_for_deep = augment_task_ids%3
labels_for_deep[range(len(XYZ_denoise),len(labels_for_deep))] = np.random.randint(0,3,len(unlabeld))

res1, order1, feature_deep, symp_res = make_cross_val_with_auto(TagLow, home_or_not, augment_dys, labels_for_deep, augment_task_ids,
                            symp_class,  feature_extract, augment_or_not, deep_params)

feature_deep1 = np.vstack(feature_deep)
feature_deep1 = np.column_stack((order1, feature_deep1))
feature_deep2 = feature_deep1[feature_deep1[:,0].argsort()]