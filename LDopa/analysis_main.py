# -*- coding: utf-8 -*-
"""
This module is the 'main' code for running the L-Dopa analysis.
The code is run from this module, which calls various functions throughout.

@author: awagner + imazeh
"""

import os
from os.path import join, sep
import numpy as np
from future.utils import lmap
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters


'''
Avishai's settings:
'''
dat_path = join('C:', sep, 'Users', 'awagner', 'Desktop', 'Large_data')
#data_path = 'C:/Users/awagner/Desktop/Large_data/'
os.chdir(os.getcwd()+"/DataScientists")

'''
Itzik's settings:
'''
data_path = join('C:', sep, 'Users', 'imazeh', 'Itzik', 'Health_prof', 'L_Dopa', 'Large_data')
os.chdir(join('C:', sep, 'Users', 'imazeh', 'Itzik', 'Health_prof', 'git_team', 'DataScientists'))

from Utils.Features import WavTransform
from Utils.Features import ts_fresh
import Utils.Preprocessing.projections as projections
import Utils.Preprocessing.denoising as Denoiseing_func
import LDopa.DataReading.read_data_from_ldopa as data_reading
import LDopa.Classification.Classification as classifier
import LDopa.Evaluation.evaluation as evaluation

###
"""
Read with SQL
"""
res = data_reading.read_all_data("ConnectSmoove2")
res = data_reading.arrange_res(res, path=join('C:', sep, 'Users', 'awagner'))
tags_df, lab_x, lab_y, lab_z, lab_n = data_reading.MakeIntervalFromAllData(res, 25, 2, 1, 1, 50)

#######
"""
Read Itzik
"""
exec(open('./LDopa/DataReading/load_from_csv.py').read())
tags_df = read_tag_data(data_path)
lab_x, lab_y, lab_z = read_data_windows(data_path, read_also_home_data=False,
                                        sample_freq=50, window_size=5)
#####

'''
Read data - new approach:
'''
res = pd.read_csv(join(data_path, 'all_lab_data.csv'))
#res = res.drop('Unnamed: 0', 1)
res = data_reading.arrange_res(res, path=join('LDopa', 'DataReading', 'Resources', 'mapTasksClusters.csv'))
tags_df, lab_x, lab_y, lab_z, lab_n = data_reading.make_interval_from_all_data(res, 5, 2.5, 1, 1, 50)
lab_x_numpy = lab_x.as_matrix()
lab_x = lab_x_numpy[:, range(len(lab_x_numpy[0])-1)]
lab_y_numpy = lab_y.as_matrix()
lab_y = lab_y_numpy[:, range(len(lab_x_numpy[0])-1)]
lab_z_numpy = lab_z.as_matrix()
lab_z = lab_z_numpy[:, range(len(lab_x_numpy[0])-1)]


'''
Build an indicator vector, which will indicate which records are relevant for
the analysis, with regards to the specific task and symptom:
'''
task_names = tags_df.Task.as_matrix()
task_clusters = tags_df.TaskClusterId.as_matrix()
relevant_task_names = []
relevant_task_clusters = [1]  # 1=resting, 4=periodic hand movement, 5=walking
cond = np.asarray(lmap(lambda x: x in relevant_task_clusters, task_clusters))


'''
Perform transformations on the data:
'''
# Project data from 3 to 2 dimensions:
lab_ver_proj, lab_hor_proj = projections.project_from_3_to_2_dims(lab_x, lab_y,
                                                                  lab_z)


'''
Filter the data according to the condition vector:
'''
sub_lab_ver_proj = lab_ver_proj[cond == True]
sub_lab_hor_proj = lab_hor_proj[cond == True]


'''
Perform signal denoising:
'''
lab_ver_denoised = Denoiseing_func.denoise_signal(sub_lab_ver_proj)
lab_hor_denoised = Denoiseing_func.denoise_signal(sub_lab_hor_proj)


'''
Extract features:
'''
# Create wavelet features for each projected dimension,
# and stack both dimensions horizontally:
WavFeatures = WavTransform.WavTransform()
lab_ver_features = WavFeatures.createWavFeatures(lab_ver_denoised)
lab_hor_features = WavFeatures.createWavFeatures(lab_hor_denoised)
features_data = np.column_stack((lab_ver_features, lab_hor_features))

# Create TSFresh features for each projected dimension,
# and stack both dimensions horizontally:
lab_ver_for_tsf = ts_fresh.convert_signals_for_ts_fresh(sub_lab_ver_proj,
                                                       "ver")
lab_ver_tsf_features = extract_features(lab_ver_for_tsf, default_fc_parameters=ComprehensiveFCParameters(),
                                        column_id="signal_id", column_sort="time")
lab_hor_for_tsf = ts_fresh.convert_signals_for_ts_fresh(sub_lab_hor_proj,
                                                       "hor")
lab_hor_tsf_features = extract_features(lab_hor_for_tsf, default_fc_parameters=EfficientFCParameters(),
                                        column_id="signal_id", column_sort="time")
features_data = pd.concat([lab_ver_tsf_features, lab_hor_tsf_features,pd.DataFrame(lab_ver_features), pd.DataFrame(lab_hor_features)], axis=1)


'''
Prepare the data for the classification process:
'''


def create_labels(symptom_name, tags_data, condition_vector, binarize=True):
    if symptom_name == 'tremor':
        label_vector = tags_data.TremorGA.as_matrix()
    elif symptom_name == 'dyskinesia':
        label_vector = tags_data.DyskinesiaGA.as_matrix()
    elif symptom_name == 'bradykinesia':
        label_vector = tags_data.BradykinesiaGA.as_matrix()
    label_vector = label_vector[condition_vector == True]
    if binarize == True:
        label_vector[label_vector > 0] = 1
    return label_vector

labels = create_labels('tremor', tags_data=tags_df, condition_vector=cond,
                       binarize=True)
# features = features_data[cond==True]
features = features_data.copy()
# tags_df_after_cond = tags_df[cond==True]
patients = tags_df.SubjectId[cond == True]
task_ids = tags_df.TaskID[cond == True]


'''
Optimize the hyper-parameters of the classification model, using a leave-one-patient-out approach:
'''


optimized_model = classifier.optimize_hyper_params(features, labels, patients, 'xgboost',
                                                   hyper_params=None, scoring_measure=None,
                                                   eval_iterations=200)

'''
Make predictions for each segment in the data.
For each user, the model is trained on all the other users:
'''


all_pred = classifier.make_cv_predictions_prob_for_all_segments(features, labels, patients,
                                                                optimized_model, task_ids)


'''
Extract features, grouped for each task:
'''


agg_segments_df = all_pred.groupby(['patient', 'task', 'true_label']).agg(['min','max','mean','median'])
agg_segments_df.columns = agg_segments_df.columns.droplevel()
agg_segments_df.reset_index(inplace=True)


'''
Use the extracted features to classify each task.
Start by optimizing the hyper-parameters. Then, make predictions for each
aggregated (task) segment:
'''


agg_patients = agg_segments_df['patient']
agg_labels = agg_segments_df['true_label']
agg_features = agg_segments_df[[x for x in agg_segments_df.columns if x not in ['patient',
                                                                                'true_label',
                                                                                'task']]]


opt_model_for_agg_segments = classifier.optimize_hyper_params(agg_features, agg_labels, agg_patients,
                                                              model_name='random_forest_for_agg',
                                                              hyper_params=None, scoring_measure=None,
                                                              eval_iterations=50)

final_pred = classifier.make_cv_predictions_for_agg_segments(agg_segments_df, opt_model_for_agg_segments,
                                                             binary_class_thresh=0.5)


'''
------EVALUATION PHASE------
'''
# Per patient metrics:
patients_metrics = evaluation.per_patient_metrics(final_pred)

# Print per patient metrics:
for patient in patients_metrics.keys():
    print('\n', '\n', 'Metrics for patient', patient)
    print('Confusion matrix:')
    print(patients_metrics[patient]['conf_matrix'])
    print('Accuracy:', round(patients_metrics[patient]['accuracy'], 3))
    print('Sensitivity (recall):', round(patients_metrics[patient]['recall'], 3))
    print('Precision:', round(patients_metrics[patient]['precision'], 3))
    print('AUC:', patients_metrics[patient]['auc'])

evaluation.global_metrics(final_pred)

evaluation.per_patient_proportion_correlation(final_pred)


'''
Evaluate the classifications with the UPDRS scores provided per visit at the lab:
'''
# Read the UPDRS data:
updrs_data = pd.read_csv(join(data_path, 'UPDRS_WITH_DETAILS.csv'))
# Remove rows with no UPDRS scores:
updrs_data_no_na = updrs_data.dropna()

# Prepare data for evaluation:
dates = pd.to_datetime(tags_df.TSStart[cond == True]).dt.date
dates = pd.DataFrame(dates).set_index(all_pred.index)
all_pred['visit_date'] = dates

per_visit = evaluation.per_visit_compare_updrs(all_pred)
patients_without_updrs = [131, 132]
per_visit_having_updrs = per_visit[~per_visit.patient.isin(patients_without_updrs)]
per_visit_plus_updrs = pd.concat([updrs_data_no_na.reset_index(),
                                  per_visit_having_updrs.reset_index()],
                                  axis=1)

evaluation.create_box_plot(per_visit_plus_updrs, updrs_measure='Rest_tremor', score_aggregated='mean')
