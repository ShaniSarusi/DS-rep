#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 01:36:16 2017

@author: HealthLOB
"""

logo = LeaveOneGroupOut()
cv = logo.split(features, labels, subject_ids)
cv1 = list(cv)
cv = list(cv1)


res1 = []
optimize_model_for_pat = []
for train, subj in cv:
    len(res1)
    xtest = features[subj]
    labels_test = labels[subj]
    patients_test = subject_ids[subj]
    task_ids_test = np.asarray(task_ids)[subj]
    
    optimized_model = classifier.optimize_hyper_params(xtest, labels_test, task_ids_test,
                        'xgboost', hyper_params=None, scoring_measure = None,eval_iterations = 40)
    
    all_pred_per_pateint = classifier.make_cv_predictions_prob_for_all_segments(xtest, labels_test, 
                           pd.core.series.Series(task_ids_test), optimized_model, task_ids_test)
    all_pred_per_pateint['patient'] =  np.asarray(patients_test)
    res1.append(all_pred_per_pateint)
    
    optimize_model_for_pat.append(optimized_model.fit(xtest, labels_test))
    


res_pred_on_new = []
for i in  range(len(cv)):
    xtest = features[cv[i][1]]
    labels_test = labels[cv[i][1]]
    patients_test = subject_ids[cv[i][1]]
    task_ids_test = np.asarray(task_ids)[cv[i][1]]
    
    res_pred_temp = []
    for j in range(len(cv)):
        if(j == i):
            continue
        else:
           print(j)
           res_pred_temp.append(optimize_model_for_pat[j].predict_proba(xtest)[:,1])
    
    res_pred_on_new.append(np.transpose(np.vstack(res_pred_temp)))
         

res_pred_on_new = np.vstack(res_pred_on_new)

prob_sum_squre = np.apply_along_axis(lambda x: np.mean(x**2), 1, res_pred_on_new )
prob_sum = np.apply_along_axis(lambda x: np.mean(x), 1, res_pred_on_new )

all_pred_per_usr = pd.DataFrame({'patient': subject_ids, 'task': task_ids, 'true_label': labels, 
                                 'prob_sum_squre': prob_sum_squre, 'prob_sum': prob_sum})
    
    
agg_segments_df = all_pred_per_usr.groupby(['patient', 'task', 'true_label']).agg(['min','max','mean','median', 'quantile'])
agg_segments_df.columns = agg_segments_df.columns.droplevel()
agg_segments_df.reset_index(inplace=True)

agg_segments_df.columns = ['patient', 'task',  'true_label', '1', '2', '3', '4' ,'5', '6', '7' ,'8', '9', '10' ]

'''
Use the extracted features to classify each task.
Start by optimizing the hyper-parameters. Then, make predictions for each aggregated (task) segment:
'''
agg_patients = agg_segments_df['patient']
agg_labels = agg_segments_df['true_label']
agg_features = agg_segments_df[[x for x in agg_segments_df.columns if x not in ['patient', 'true_label', 'task']]]


opt_model_for_agg_segments = classifier.optimize_hyper_params(agg_features, agg_labels, agg_patients,
                                                   model_name='logistic_regression',
                                                   hyper_params=None, scoring_measure='f1',eval_iterations = 50)
final_pred = classifier.make_cv_predictions_for_agg_segments(agg_segments_df, opt_model_for_agg_segments,  binary_class_thresh=0.65)

evaluation.global_metrics(final_pred)
