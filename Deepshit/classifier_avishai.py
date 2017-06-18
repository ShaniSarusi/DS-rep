# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 17:03:02 2017

@author: awagner
"""

import os
import sys

mother_path = os.getcwd()+'/DataScientists/Deepshit/'
data_path = os.getcwd()+ '/PycharmProjects/Deepshit/DATA_FOLDER/'
sys.path.insert(0, mother_path)

from hyperopt import hp
from Other.hyperopt_ATM import BayesianHyperOpt
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from Other.WavTransform import wavtransform
from xgboost.sklearn import XGBClassifier

norma = (x_raw_with_tag**2+y_raw_with_tag**2+z_raw_with_tag**2)**0.5
   
WavFeatures = wavtransform()
Ver_Feature = WavFeatures.createWavFeatures(ver_proj)
Hor_Feature = WavFeatures.createWavFeatures(hor_proj)
norma_Feature = WavFeatures.createWavFeatures(norma)

MyData = np.column_stack((Ver_Feature,Hor_Feature))
MyData = norma_Feature.copy()

pred_ALL = []
##is the sample in the cluster of the right activity 
clus = [5]
cond = np.asarray(lmap(lambda x: x in clus, Task))
symp = trem.copy()

TagLow = MyData[cond==True]
symp = symp[cond==True]
meta_after_cond = meta[cond==True]

#space = {'C': hp.uniform('C',0,5), 'gamma': hp.uniform('gamma',0,5)}
space = {'learning_rate':hp.uniform('learning_rate',0,1), 'n_estimators':hp.choice('n_estimators',range(40,50)), 'max_depth':hp.choice('max_depth',range(4,20)), 'min_child_weight':hp.choice('min_child_weight',range(3)), 'gamma':hp.uniform('gamma',0,1), 'subsample':hp.uniform('subsample',0,1)}
space = {'n_estimators':hp.choice('n_estimators',range(100,110)),
         'max_depth':hp.choice('max_depth',range(2,30)),
         'min_samples_split':hp.choice('min_samples_split',range(2,10)),
         'min_samples_leaf':hp.choice('min_samples_leaf',range(2,10))}

logo = LeaveOneGroupOut()
cv = logo.split(TagLow ,symp, meta_after_cond.SubjectId)
cv1 = list(cv)
cv = list(cv1)


hyper_opt_svm = BayesianHyperOpt(space = space,estimator=XGBClassifier(), scoring=None ,cv =  cv,max_evals = 200)
hyper_opt_svm.fit(TagLow , symp)
#rf = RandomForestClassifier(n_estimators = hyper_opt_svm.best_params_['n_estimators'],max_depth = hyper_opt_svm.best_params_['max_depth'], min_samples_split = hyper_opt_svm.best_params_['min_samples_split'], min_samples_leaf = hyper_opt_svm.best_params_['min_samples_leaf'],n_jobs=2)

print("computing probability vector for each interval")
for pateint in np.unique(meta_after_cond.SubjectId):
    prob_per_pateint =  make_pred_withprob(Men_number=pateint,data_for_pred = TagLow ,symp = symp,meta_for_model = meta_after_cond,model = hyper_opt_svm.best_estimator_)
    pred_ALL.append(prob_per_pateint[:,1])

pred_as_vector = np.asarray([(sublist[0],item) for sublist in pred_ALL for item in sublist] )
lab2 = pred_as_vector[:,0]; pred_as_vector = normlize_sig(pred_as_vector[:,1]) 
prob_mat_table = make_intervals_class(pred_as_vector,meta_after_cond['TaskID'],symp)
prob_mat_table = pd.DataFrame(prob_mat_table)
prob_mat_table['lab2'] = lab2
prob_mat_unique = prob_mat_table.drop_duplicates()


model = LogisticRegression()
space = {'C': hp.uniform('C',0,5)}
hyper_opt_clf = BayesianHyperOpt(space = space,estimator=LogisticRegression(), scoring=None ,cv =  15)
hyper_opt_clf.fit(prob_mat_unique.as_matrix()[:,range(4)] , prob_mat_unique.as_matrix()[:,4])

cv = logo.split(prob_mat_unique.as_matrix()[:,range(4)], prob_mat_unique.as_matrix()[:,4], prob_mat_unique['lab2'])
cv1 = list(cv)
cv = list(cv1)

tab_results2 = estimate_LDOPA_Rstuls_conf_matrix(prob_mat_unique.as_matrix()[:,range(4)],prob_mat_unique.as_matrix()[:,4],list(cv), hyper_opt_clf.best_estimator_,thresh = 0.49)
