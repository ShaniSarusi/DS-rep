#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 01:08:35 2017

@author: HealthLOB
"""

from sklearn.model_selection import LeaveOneGroupOut
import numpy as np
from sklearn.metrics import accuracy_score

def make_cross_val(data, Task_params, Task_andSymp, meta_data, deep_task_ids, main_network,  second_network, augment_or_not, deep_params):

    weights_start_symp = main_network.get_weights()

    logo = LeaveOneGroupOut()
    cv = logo.split(data, Task_params, meta_data)
    cv1 = list(cv)
    cv = list(cv1)
        
    order_final = []
    res = []
    symp_cor_res = []
    features_from_deep = []
    for train, test in cv:
        print(test)
        Mytrain = train[range(len(train) - len(train)%deep_params['batch_size'])]
        test = test[range(len(test) - len(test)%deep_params['batch_size'])]
        main_network.set_weights(weights_start_symp)
        xtest = data[test]
        xtrain = data[train]
        #weight_sample = np.abs(1 - Task_for_pred[train])
        main_network.fit([xtrain], [Task_params[Mytrain], Task_andSymp[Mytrain]],
                         epochs=deep_params['epochs'] , batch_size=deep_params['batch_size'], shuffle=True, 
                         validation_data=([data[test]], [Task_params[test], Task_andSymp[test]]),
                         callbacks=[deep_params['change_lr']], verbose=2)#
        
        temp_res = main_network.predict([xtest[augment_or_not[test] == 1]])
        features_from_deep.append(second_network.predict([xtest[augment_or_not[test] == 1]]))
        res.append(temp_res[0])
        symp_cor_res.append(Task_params[test][augment_or_not[test] == 1])
        #features_from_deep.append(second_network.predict(xtest[augment_or_not[test] == 1]))
        order_final.append(deep_task_ids[test][augment_or_not[test] == 1])
        print(confusion_matrix(symp_cor_res[len(symp_cor_res)-1],np.where(temp_res[0]>0.5,1,0)))
    #res1 = np.vstack(res)
    order1 = np.hstack(order_final)
    
    return res, order1, symp_cor_res, features_from_deep




def make_cross_val_with_auto(data, home_or_not, symp, meta_data, deep_task_ids, main_network,  second_network, augment_or_not, deep_params):

    weights_start_symp = main_network.get_weights()
    
    normlise_data = lmap(normlize_sig, data)
    
    logo = LeaveOneGroupOut()
    cv = logo.split(data, symp, meta_data)
    cv1 = list(cv)
    cv = list(cv1)
        
    order_final = []
    res = []
    symp_cor_res = []
    features_from_deep = []
    for train, test in cv:
        print(test)
        main_network.set_weights(weights_start_symp)
        xtest = data[test]
        xtrain = data[train]
        main_network.fit([xtrain, home_or_not[train]], [symp[train],xtrain],epochs=deep_params['epochs'] , batch_size=deep_params['batch_size'], shuffle=True,
                         validation_data=([xtest[augment_or_not[test] == 1],home_or_not[test][augment_or_not[test] == 1]], 
                                          [symp[test][augment_or_not[test] == 1],xtest[augment_or_not[test] == 1]]),
                                          callbacks=[deep_params['change_lr']] ,verbose=2)#
        temp_res = main_network.predict([xtest[augment_or_not[test] == 1], np.ones(len(test))])
        res.append(temp_res[0])
        symp_cor_res.append(symp[test][augment_or_not[test] == 1])
        features_from_deep.append(second_network.predict(xtest[augment_or_not[test] == 1]))
        order_final.append(deep_task_ids[test][augment_or_not[test] == 1])
        print(confusion_matrix(symp_cor_res[len(symp_cor_res)-1],np.where(temp_res[0]>0.5,1,0)))
    res1 = np.vstack(res)
    order1 = np.hstack(order_final)
    
    return res1, order1, features_from_deep, symp_cor_res




def scheduler_in_func(epoch):
    if(epoch == 1):
        K.set_value(symp_class.optimizer.lr,0.0001)
    if(epoch == 5):
        K.set_value(symp_class.optimizer.lr,0.00005)
    return K.get_value(symp_class.optimizer.lr)