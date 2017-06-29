#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 01:08:35 2017

@author: HealthLOB
"""

from sklearn.model_selection import LeaveOneGroupOut
import numpy as np

def make_cross_val(data, symp, Task_andSymp, meta_data, main_network,  second_network, augment_or_not, deep_params):

    weights_start_symp = main_network.get_weights()

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
        #weight_sample = np.abs(1 - Task_for_pred[train])
        main_network.fit(xtrain, [symp[train],Task_andSymp[train],symp[train]],epochs=deep_params['epochs'] , batch_size=128, shuffle=True, validation_data=(xtest, [symp[test],Task_andSymp[test], symp[test]]),callbacks=[deep_params['change_lr']],class_weight = deep_params['class_weight']  ,verbose=2)#
        temp_res = main_network.predict(xtest[augment_or_not[test] == 1])
        res.append(temp_res[0])
        symp_cor_res.append(symp[test])
        features_from_deep.append(second_network.predict(xtest[augment_or_not[test] == 1]))
        order_final.append(test[augment_or_not[test] == 1])
    res1 = np.vstack(res)
    order1 = np.hstack(order_final)
    
    return res1, order1