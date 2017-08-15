#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 00:40:33 2017

@author: HealthLOB
"""


logo = LeaveOneGroupOut()
cv_one_ = logo.split(TagLow[fine_tune], augment_dys[fine_tune], labels_for_deep)
cv1_fine = list(cv)
cv_fine = list(cv1)


pat_agg = [np.unique(np.asarray(tags_df.SubjectId[cond==True])[np.where(task_ids == i)[0]]) for i in agg_segments_df.task]
pat_task_clusters = [np.unique(task_clusters[cond==True][np.where(task_ids == i)[0]]) for i in agg_segments_df.task]

pat_agg = np.hstack(pat_agg)
pat_task_clusters = np.hstack(pat_task_clusters)

for i in np.unique(pat_agg):
    print(i)
    print(accuracy_score(final_pred.true_label[pat_agg == i], final_pred.binary_prediction[pat_agg == i]))
    print(confusion_matrix(final_pred.true_label[pat_agg == i], final_pred.binary_prediction[pat_agg == i]))
    
    
MyMen = 134
for i in np.unique(pat_task_clusters[pat_agg == MyMen]):
    print(i)
    print(accuracy_score(np.asarray(final_pred.true_label[np.where(pat_agg)[0] == MyMen])[pat_task_clusters[np.where(pat_agg)[0] == MyMen] == i], 
                np.asarray(final_pred.binary_prediction[pat_agg == MyMen])[pat_task_clusters[np.where(pat_agg)[0] == MyMen] == i]))
    print(confusion_matrix(np.asarray(final_pred.true_label[np.where(pat_agg)[0] == MyMen])[pat_task_clusters[pat_agg == MyMen] == i], 
                np.asarray(final_pred.binary_prediction[np.where(pat_agg)[0] == MyMen])[pat_task_clusters[np.where(pat_agg)[0] == MyMen] == i]))