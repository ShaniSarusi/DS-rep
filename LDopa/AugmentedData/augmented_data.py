#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 01:35:10 2017

@author: HealthLOB
"""

from scipy.stats import ortho_group
import numpy as np


def augment_data(XYZ, temp_labels, task_clusters, task_subjectID, task_ids,num_iter=20,
                 num_iter_symp = 20,group = [0,1,2]):
    """
    Augmented data by orthonormal transformation
    
    Input:
        XYZ (numpy): origin data
        temp_labels (1D numpy): labels per segment
        task_clasters (1D numpy): clusters per segment
        task_subjectID (1D numpy)- User ID per segment
        task_ids (1D numpy) - task_ids per segment
        num_iter (integer) - numpy of random transformation per segment
        group (list of ineger) - group clusters
    
    Output:
        augment_XYZ (numpy): augmented XYZ
        augment_symp (numpy): aegmented symptom per segment
        augment_task (numpy): aegmented task per segment
        augment_subject_id (numpy): aegmented user name per segment
        augment_task_ids (numpy): aegmented task_ids per segment
        augment_or_not (numpy): The data is augmented or from original data XYZ
    """
    
    augment_XYZ = []
    augment_symp = []
    augment_task = []
    augment_subject_id = []
    augment_or_not = []
    augment_task_ids = []
    augment_XYZ.append(XYZ)
    augment_symp.append(temp_labels)
    augment_task.append(task_clusters)
    augment_subject_id.append(task_subjectID)
    augment_task_ids.append(task_ids)
    augment_or_not.append(np.ones(len(task_subjectID)))
    for samp in range(XYZ.shape[0]):
        print(samp)
        num_of_iter = num_iter
        temp_list_perm = []; temp_list_symp = []; temp_list_Task = []; 
        temp_list_user_id = []; temp_task_ids = []; temp_aug_or_not = []
        if(task_clusters[samp] in group):
            if((temp_labels[samp] > 0)):
                num_of_iter = num_iter_symp
            for i in range(num_of_iter):
                m = ortho_group.rvs(dim=3)
                temp = m.dot(XYZ[samp].T).T
                temp_list_perm.append(temp)
                temp_list_symp.append(temp_labels[samp])
                temp_list_Task.append(task_clusters[samp])
                temp_list_user_id.append(task_subjectID[samp])
                temp_task_ids.append(task_ids[samp])
                temp_aug_or_not.append(0)
            temp_list_perm = np.stack(temp_list_perm)
            temp_list_symp = np.stack(temp_list_symp)
            temp_list_Task = np.stack(temp_list_Task)
            temp_task_ids = np.stack(temp_task_ids)
            temp_list_user_id = np.stack(temp_list_user_id)
            temp_aug_or_not = np.stack(temp_aug_or_not)
            augment_XYZ.append(temp_list_perm)
            augment_symp.append(temp_list_symp)
            augment_task.append(temp_list_Task)
            augment_subject_id.append(temp_list_user_id)
            augment_task_ids.append(temp_task_ids)
            augment_or_not.append(temp_aug_or_not)
            
    augment_XYZ = np.vstack(augment_XYZ)
    augment_symp = np.hstack(augment_symp)
    augment_task = np.hstack(augment_task)
    augment_subject_id = np.hstack(augment_subject_id)
    augment_task_ids = np.hstack(augment_task_ids)
    augment_or_not = np.hstack(augment_or_not)
    return augment_XYZ, augment_symp, augment_task, augment_subject_id, augment_task_ids, augment_or_not
