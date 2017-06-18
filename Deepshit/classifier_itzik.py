# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 17:03:02 2017

@author: awagner
"""
from hyperopt import hp
from Other.hyperopt_ATM import BayesianHyperOpt
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from scipy.stats.stats import pearsonr

##is the sample in the cluster of the right activity 
clus = [1] # 1=resting, 4=periodic hand movement, 5=walking
cond = np.asarray(lmap(lambda x: x in clus, Task))
symp = trem.copy()


tagged_features = TagLow[cond==True]
tagged_labels = symp[cond==True]
patients = meta.SubjectId[cond==True]

#Generate indices to split data into training and test set:
logo = LeaveOneGroupOut()
cv = logo.split(tagged_features, tagged_labels, patients)
cv1 = list(cv)
cv = list(cv1)

'''Define the hyper-parameters space:'''
#For SVM:
#space = {'C': hp.uniform('C',0,5), 'gamma': hp.uniform('gamma',0,5)}
svm_space = {'C': hp.choice('C', [0.1, 1, 10]), 'gamma': hp.uniform('gamma',0,5)}

#For Random Forest:
rf_space = {'n_estimators':hp.choice('n_estimators', [10, 100, 1000]),
            'max_depth':hp.choice('max_depth', [2, 5, 10, 30]),
            'min_samples_split':hp.choice('min_samples_split', list(range(2, 12, 2))),
            'min_samples_leaf':hp.choice('min_samples_leaf', list(range(2, 12, 2)))}

#For xgboost:
xgb_space = {'learning_rate':hp.uniform('learning_rate',0,1),
         'n_estimators':hp.choice('n_estimators',range(40,50)),
         'max_depth':hp.choice('max_depth',range(4,20)),
         'min_child_weight':hp.choice('min_child_weight',range(3)),
         'gamma':hp.uniform('gamma',0,1),
         'subsample':hp.uniform('subsample',0,1)}


hyper_opt_svm = BayesianHyperOpt(space=svm_space, estimator=svm.SVC(kernel ='rbf'), scoring=None,
                                 cv=list(cv), max_evals = 6)
hyper_opt_svm.fit(tagged_features, tagged_labels)

hyper_opt_rf = BayesianHyperOpt(space=rf_space, estimator=RandomForestClassifier(n_jobs=2),
                                scoring=None, cv=list(cv), max_evals=20)
hyper_opt_rf.fit(tagged_features, tagged_labels)

hyper_opt_xgb = BayesianHyperOpt(space=xgb_space, estimator=XGBClassifier(), scoring=None,
                                 cv=list(cv), max_evals = 50)
hyper_opt_xgb.fit(tagged_features, tagged_labels)
xgb = hyper_opt_xgb.best_estimator_

svc = svm.SVC(C = hyper_opt_svm.best_params_['C'], kernel="rbf",
              gamma = hyper_opt_svm.best_params_['gamma'], probability=True)

#xg = XGBoost_Classifier2(learning_rate=0.025, n_estimators=50, max_depth=12, min_child_weight=2, gamma=0.6, subsample=0.75,
#                        colsample_bytree=1., scale_pos_weight=3,reg_lambda1 = 0,reg_alpha1 = 0)

rf = RandomForestClassifier(n_estimators = hyper_opt_rf.best_params_['n_estimators'],
                            max_depth = hyper_opt_rf.best_params_['max_depth'],
                            min_samples_split = hyper_opt_rf.best_params_['min_samples_split'],
                            min_samples_leaf = hyper_opt_rf.best_params_['min_samples_leaf'],
                            n_jobs=2)

print("computing probability vector for each interval")
pred_ALL = []
for pateint in range(len(Men)):
    prob_per_pateint =  make_pred_withprob(cond, Men_number=pateint, data_for_pred=TagLow,
                                           symp=symp, meta_for_model=meta, model=xgb, Men=Men)
    #prob_per_pateint returns a list of lists with [probability for '0', probability for '1']
    pred_ALL.append(prob_per_pateint[:,1])

#Take the first probability for each user as their identifier (for CV purposes)
pred_as_vector = np.asarray([(sublist[0],item) for sublist in pred_ALL for item in sublist]) 
user_label = pred_as_vector[:,0]
pred_as_vector = normlize_sig(pred_as_vector[:,1])  
prob_mat_table = make_intervals_class(pred_as_vector,meta['TaskID'][cond==True],symp[cond==True])
col_names=['min_val', 'max_val', 'mean_val', 'median_val', 'label']
prob_mat_table = pd.DataFrame(prob_mat_table, columns=col_names) 
prob_mat_table['user'] = user_label 
prob_mat_unique = prob_mat_table.drop_duplicates()
#prob_mat_unique.reset_index(inplace=True)

patients = prob_mat_unique['user']
features_col_names = col_names[0:4]
features_all = prob_mat_unique[features_col_names]
labels = prob_mat_unique['label']

'''Hyper-parameters optimization is done only on the training data.
   Thus, split here to train and test data.'''

train, test = train_test_split(prob_mat_unique, test_size = 0.25)
patients_train = train['user']
patients_test = test['user']
features_train = train[features_col_names]
features_test = test[features_col_names]
labels_train = train['label']
labels_test = test['label']


'''Define model and hyper-parameter space for final classification stage:'''
#Random Forest:
model = RandomForestClassifier(n_jobs=-1)
rf_space_small_data = {'n_estimators':hp.choice('n_estimators', [10, 100, 1000]),
                       'max_depth':hp.choice('max_depth', range(1, 5)),
                       'min_samples_split':hp.choice('min_samples_split', list(range(2, 12, 2))),
                       'min_samples_leaf':hp.choice('min_samples_leaf', list(range(1, 13, 2)))}
space = rf_space_small_data

#Logistic Regression:
model  = LogisticRegression(n_jobs=-1)
space = {'penalty': hp.choice('penalty', ['l1', 'l2']),
         'C': hp.choice('C', list(range(1,11,2)))}


'''Select hyper-parameters for model, using LOGO cross-validation:'''
logo = LeaveOneGroupOut()
#Generate indices to split data into training and test set:
cv = logo.split(features_all, labels, patients)
#cv1 = list(cv)
#cv = list(cv)

hyper_opt_clf = BayesianHyperOpt(space=space, estimator=model, scoring=None, cv=list(cv), max_evals=50)
hyper_opt_clf.fit(features_all, labels)

clf = hyper_opt_clf.best_estimator_

'''Set the model, with the best-found hyper-parameters:'''
#Random Forest:
clf = RandomForestClassifier(n_estimators = hyper_opt_clf.best_params_['n_estimators'],
                            max_depth = hyper_opt_clf.best_params_['max_depth'],
                            min_samples_split = hyper_opt_clf.best_params_['min_samples_split'],
                            min_samples_leaf = hyper_opt_clf.best_params_['min_samples_leaf'],
                            n_jobs=-1)

#Logistic Regression:
clf = LogisticRegression(penalty = hyper_opt_clf.best_params_['penalty'],
                         C = hyper_opt_clf.best_params_['C'],
                         n_jobs=-1)

'''Fit the model with LOGO cross-validation (over ALL data, rather than just the training set):'''
logo = LeaveOneGroupOut()
cv_all = logo.split(features_all, labels, patients)
cv_all_1 = list(cv_all)
cv_all = list(cv_all_1)



#conf_mat_as_scorer = make_scorer(confusion_matrix)

y_true_all = []
y_true_all_prop = []
y_pred_all = []
y_pred_all_prop = []

for train_index, test_index in list(cv_all):
    train_index = list(train_index)
    test_index = list(test_index)
#    print (test_index)
    X_train, X_test = features_all.iloc[train_index,:], features_all.iloc[test_index,:]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    true_prop = np.mean(y_test)
    y_true_all_prop.append(true_prop)
    y_true_all.extend(y_test)
    clf.fit(X_train, y_train)
    test_prediction = clf.predict(X_test)
    test_proba = clf.predict_proba(X_test)[:,1]
#    test_prediction = np.where(test_proba>=0.5, 1, 0)
    pred_prop = np.mean(test_prediction)
    y_pred_all_prop.append(pred_prop)
    y_pred_all.extend(test_prediction)
    print ('\n', 'accuracy:', accuracy_score(y_test, test_prediction))
#    print ('true prop:', true_prop)
#    print ('prediction prop:', pred_prop)
    print (confusion_matrix(y_test, test_prediction))

print ('Total accuracy:', accuracy_score(y_true_all, y_pred_all))
print ('Confusion matrix over all patients:', '\n', confusion_matrix(y_true_all, y_pred_all))

print (pearsonr(y_true_all_prop, y_pred_all_prop))
plt.scatter(y_true_all_prop, y_pred_all_prop)
plt.plot(np.unique(y_true_all_prop), \
         np.poly1d(np.polyfit(y_true_all_prop, y_pred_all_prop, 1))(np.unique(y_true_all_prop)),\
         color='r')
plt.show()
#print (np.corrcoef(y_true_all, y_pred_all))

#Finally, fit the model for the entire dataset:
clf.fit(features_all, labels)
y_prob = clf.predict_proba(features_all)[:,1]


'''
cv_scores = cross_val_score(estimator=clf, X=features_all, y=labels, scoring=conf_mat_as_scorer,
                            cv=list(cv_all))

#Fit the model to the training set:
clf.fit(features_train, labels_train)
#Show the score for the test hold-out set:
print(clf.score(features_test, labels_test))

print("With Logistic Regression return ")
tab= []
count = -1
treshold_log = 0.5
for pateint in np.unique(meta.SubjectId):
    count = count + 1
    print(count)
    temp_vec = meta.SubjectId[cond==True].reset_index()
    final_results = clf.predict_proba(prob_mat_table[list(range(4))].loc[temp_vec.SubjectId == pateint])[:,1]
    final_results = np.where(final_results >treshold_log,1,0)
    tab.append(confusion_matrix(symp[cond==True][temp_vec.SubjectId.as_matrix() == pateint],final_results))
    print(tab[count])


tabALL = pd.DataFrame(np.zeros((2,2)))
pred_total = np.zeros(1)
#colname = tabALL.columns.values
#rowname = list(tabALL.index)
for i in range(len(tab)): 
    temp = pd.DataFrame(tab[i])
    colname = temp.columns.values
    rowname = list(temp.index)
    for col in colname:
        for row in rowname:
            tabALL[row][col] = tabALL[row][col] + temp[row][col]
print(tabALL)          
float(tabALL[1][1]+tabALL[0][0])/np.sum(np.sum(tabALL))
'''