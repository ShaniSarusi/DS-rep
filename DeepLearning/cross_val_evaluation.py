#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 01:08:35 2017

@author: HealthLOB
"""

from sklearn.model_selection import LeaveOneGroupOut

def make_cross_val(data, symp, Task_andSymp, meta_data, main_network,  second_network,change_lr, class_weight, augment_or_not, Order_vector):

    weights_start_symp = main_network.get_weights()

    logo = LeaveOneGroupOut()
    cv = logo.split(data, symp, meta_data)
    cv1 = list(cv)
    cv = list(cv1)
    
    res = []
    symp_cor_res = []
    features_from_deep = [Order_vector]
    for train, test in cv:
        print(test)
        main_network.set_weights(weights_start_symp)
        xtest = data[test]
        xtrain = data[train]
        order_test = 
        #weight_sample = np.abs(1 - Task_for_pred[train])
        main_network.fit(xtrain, [symp[train],Task_andSymp[train]],epochs=50 , batch_size=128, shuffle=True, validation_data=(xtest, [symp[test],Task_andSymp[test]]),callbacks=[change_lr],class_weight = class_weight  ,verbose=2)#
        temp_res = main_network.predict(xtest[augment_or_not[test] == 1])
        res.append(temp_res[0])
        symp_cor_res.append(symp[test])
        features_from_deep.append(second_network.predict(xtest[augment_or_not[test] == 1]))
    res1 = np.vstack(res)
    
    return res1