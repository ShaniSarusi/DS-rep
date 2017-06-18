#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 02:46:09 2017

@author: HealthLOB
"""
import numpy as np
import pandas as pd


def estimate_LDOPA_Rstuls_conf_matrix(data_for_pred,symp_for_pred,Cross_validation,model_with_params,thresh = 0.5):
    tab= []
    count = -1
    for train, test in Cross_validation:
        count = count + 1
        print(train)
        model_with_params.fit(data_for_pred[train] , symp_for_pred[train])
        final_results = model_with_params.predict_proba(data_for_pred[test])
        final_results = np.where(final_results[:,1] >thresh,1,0)
        tab.append(confusion_matrix(symp_for_pred[test],final_results))
    
    
    tabALL = pd.DataFrame(np.zeros((2,2)))
    for i in range(len(tab)): 
        temp = pd.DataFrame(tab[i])
        colname = temp.columns.values
        rowname = list(temp.index)
        for col in colname:
            for row in rowname:
                tabALL[row][col] = tabALL[row][col] + temp[row][col]
    print(tabALL)         
    return(tabALL)
    #print(float(tabALL[1][1]+tabALL[0][0])/np.sum(np.sum(tabALL)))
        


def estimate_LDOPA_Rstuls_corr(data_for_pred,symp_for_pred,Cross_validation,model_with_params,thresh = 0.5):
    tab= []
    tab1 = []
    count = -1
    for train, test in cv:
        count = count + 1
        print(count)
        #temp_vec = meta.SubjectId[cond==True].reset_index()
        model_with_params.fit(data_for_pred[train] , symp_for_pred[train])
        final_results = model_with_params.predict_proba(data_for_pred[test])#pd.DataFrame(pred_as_vector)[temp_vec.SubjectId == pateint]#
        final_results = np.where(final_results[:,1] >treshold_log,1,0)
        tab.append(np.sum(final_results)/len(final_results))
        tab1.append(np.sum(data_for_pred[test])/len(symp_for_pred[test]))
    
    return(pearsonr(tab1,tab))
    #print(float(tabALL[1][1]+tabALL[0][0])/np.sum(np.sum(tabALL)))
        



print("With Logistic Regression return ")
tab= []
tab1 = []
count = -1
treshold_log = 0.3
for train, test in cv:
    count = count + 1
    print(count)
    #temp_vec = meta.SubjectId[cond==True].reset_index()
    hyper_opt_clf.best_estimator_.fit(prob_mat_unique.as_matrix()[:,range(4)][train] , prob_mat_unique.as_matrix()[:,4][train])
    final_results = hyper_opt_clf.best_estimator_.predict_proba(prob_mat_unique.as_matrix()[:,range(4)][test])#pd.DataFrame(pred_as_vector)[temp_vec.SubjectId == pateint]#
    final_results = np.where(final_results[:,1] >treshold_log,1,0)
    tab.append(np.sum(final_results)/len(final_results))
    tab1.append(np.sum(prob_mat_unique.as_matrix()[:,4][test])/len(prob_mat_unique.as_matrix()[:,4][test]))

plt.scatter((tab1),(tab))
pearsonr(tab1,tab)


fig, ax = plt.subplots()
fit = np.polyfit(tab1,tab, deg=1)
ax.plot(tab1, fit[0] * np.asarray(tab1) + fit[1], color='red')
ax.scatter(tab1, tab)
fig.show()
