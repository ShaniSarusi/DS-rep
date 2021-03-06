# -*- coding: utf-8 -*-
"""
Created on Sun May 28 16:42:27 2017

@author: imazeh
"""

import os
import numpy as np
from future.utils import lmap
import datetime as dt

'''
Define path for data files, both from home and from the lab:
'''
data_path = 'C:\\Users\\imazeh\\Itzik\\Health_prof\\L_Dopa\\Large_data\\'
if os.getcwd() == 'C:\\Users\\imazeh':
    os.chdir(os.getcwd()+"/Itzik/Health_prof/git4/DataScientists/LDopa")

from features import WavTransform
exec(open('./Data_reading/load_from_csv.py').read())
exec(open('./preprocessing/preprocess_time_windows_data.py').read())
exec(open('./preprocessing/butter_filters.py').read())
exec(open('./Classification/classifier.py').read())
exec(open('./Evaluation/evaluation.py').read())


'''
Read data - tags data from lab + the data itself from home / lab
-----------------THE READING DATA SECTION IS TO BE CHANGED BY PLUGGING IN AVISHAI'S NEW CODE--------------------
'''
tags_df = read_tag_data(data_path)
lab_x, lab_y, lab_z = read_data_windows(data_path, read_also_home_data=False, sample_freq=50, window_size=5)

'''
Perform transformation on the data:
    -----------------IF NEEDED, CHANGE THE FUNCTIONS EXECUTING THE PROJECTION--------------------
'''
#Project data from 3 to 2 dimensions:
lab_ver_proj, lab_hor_proj = project_from_3_to_2_dims(lab_x, lab_y, lab_z)


'''
Perform signal denoising:
    -----------------IF NEEDED, CHANGE THE FUNCTIONS EXECUTING THE DENOISING--------------------
'''
lab_ver_denoised = denoise_signal(lab_ver_proj)
lab_hor_denoised = denoise_signal(lab_hor_proj)

'''
Extract features:
'''
#Create features for each projected dimension, and stack both dimensions horizontally:
WavFeatures = WavTransform.wavtransform()
lab_ver_features = WavFeatures.createWavFeatures(lab_ver_denoised)
lab_hor_features = WavFeatures.createWavFeatures(lab_hor_denoised)
features_data = np.column_stack((lab_ver_features, lab_hor_features))


'''
Prepare the data for the classification process:
'''
#Build an indicator vector, which will indicate which records are relevant for the analysis:
task_names = tags_df.Task.as_matrix()
task_clusters = tags_df.TaskClusterId.as_matrix()
relevant_task_names = []
relevant_task_clusters = [4] # 1=resting, 4=periodic hand movement, 5=walking
cond = np.asarray(lmap(lambda x: x in relevant_task_clusters, task_clusters))

#Create features and labels data frames, according to the condition indicator:
def create_labels(symptom_name, tags_data, condition_vector, binarize=True):
    if symptom_name == 'tremor':
        label_vector = tags_data.TremorGA.as_matrix()
    elif symptom_name == 'dyskinesia':
        label_vector = tags_data.DyskinesiaGA.as_matrix()
    elif symptom_name == 'bradykinesia':
        label_vector = tags_data.BradykinesiaGA.as_matrix()
    label_vector = label_vector[condition_vector==True]
    if binarize==True:
        label_vector[label_vector>0] = 1
    return label_vector

labels = create_labels('bradykinesia', tags_data=tags_df, condition_vector=cond, binarize=True)
features = features_data[cond==True]
#tags_df_after_cond = tags_df[cond==True]
patients = tags_df.SubjectId[cond==True]
task_ids = tags_df.TaskID[cond==True]


'''
Optimize the hyper-parameters of the classification model, using a leave-one-patient-out approach:
'''
optimized_model = optimize_hyper_params(features, labels, patients, 'xgboost',
                                        hyper_params=None, scoring_measure = None)

'''
Make predictions for each segment in the data.
For each user, the model is trained on all the other users:
'''
all_pred = make_cv_predictions_prob_for_all_segments(features, labels, patients, optimized_model,
                                                     task_ids)

'''
Extract features, grouped for each task:
'''
agg_segments_df = all_pred.groupby(['patient', 'task', 'true_label']).agg(['min','max','mean','median'])
agg_segments_df.columns = agg_segments_df.columns.droplevel()
agg_segments_df.reset_index(inplace=True)


'''
Use the extracted features to classify each task.
Start by optimizing the hyper-parameters. Then, make predictions for each aggregated (task) segment:
'''
agg_patients = agg_segments_df['patient']
agg_labels = agg_segments_df['true_label']
agg_features = agg_segments_df[[x for x in agg_segments_df.columns if x not in ['patient', 'true_label', 'task']]]

opt_model_for_agg_segments = optimize_hyper_params(agg_features, agg_labels, agg_patients,
                                                   model_name='random_forest_for_agg',
                                                   hyper_params=None, scoring_measure=None)
final_pred = make_cv_predictions_for_agg_segments(agg_segments_df, opt_model_for_agg_segments)


'''
------EVALUATION PHASE------
'''
#Per patient metrics:
patients_metrics = per_patient_metrics(final_pred)

#Print per patient metrics:
for patient in patients_metrics.keys():
    print ('\n', '\n', 'Metrics for patient', patient)
    print ('Confusion matrix:')
    print (patients_metrics[patient]['conf_matrix'])
    print ('Accuracy:', round(patients_metrics[patient]['accuracy'], 3))
    print ('Sensitivity (recall):', round(patients_metrics[patient]['recall'], 3))
    print ('Precision:', round(patients_metrics[patient]['precision'], 3))
    print ('AUC:', patients_metrics[patient]['auc'])

#Create and print global metrics:
global_metrics(final_pred)

#Create, print and plot correlation of the proportion of per patient classification:
per_patient_proportion_correlation(final_pred)

'''
Evaluate the classifications with the UPDRS scores provided per visit at the lab:
'''
#Read the UPDRS data:
updrs_data = pd.read_csv(data_path+'../UPDRS_WITH_DETAILS.csv')
#Remove rows with no UPDRS scores:
updrs_data_no_na = updrs_data.dropna()

#Prepare data for evaluation:
dates = pd.to_datetime(tags_df.TSStart[cond==True]).dt.date
dates = pd.DataFrame(dates).set_index(all_pred.index)
all_pred['visit_date'] = dates
        
per_visit = per_visit_compare_updrs(all_pred)
patients_without_updrs = [131, 132]
per_visit_having_updrs = per_visit[~per_visit.patient.isin(patients_without_updrs)]
per_visit_plus_updrs = pd.concat([updrs_data_no_na.reset_index(), per_visit_having_updrs.reset_index()], axis=1)

create_box_plot( per_visit_plus_updrs, updrs_measure='Rest_tremor', score_aggregated='mean')







