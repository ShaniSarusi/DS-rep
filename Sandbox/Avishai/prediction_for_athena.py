#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 02:27:11 2017

@author: HealthLOB
"""

x_unlabeld = lmap(lambda x: (Denoiseing_func.denoise(x)), unlabeld[0])
y_unlabeld = lmap(lambda x: (Denoiseing_func.denoise(x)), unlabeld[1])
z_unlabeld = lmap(lambda x: (Denoiseing_func.denoise(x)), unlabeld[2])

lab_ver_proj, lab_hor_proj = projections.project_from_3_to_2_dims(unlabeld[0], unlabeld[1] , unlabeld[2])


lab_ver_denoised = Denoiseing_func.denoise_signal(lab_ver_proj)
lab_hor_denoised = Denoiseing_func.denoise_signal(lab_hor_proj)

WavFeatures = WavTransform.WavTransform()
lab_ver_features = WavFeatures.createWavFeatures(lab_ver_denoised)
lab_hor_features = WavFeatures.createWavFeatures(lab_hor_denoised)
features_data_unlabeld = np.column_stack((lab_ver_features, lab_hor_features))

WavFeatures = WavTransform.WavTransform()
lab_ver_features = WavFeatures.createWavFeatures(lab_ver_denoised)
lab_hor_features = WavFeatures.createWavFeatures(lab_hor_denoised)
features_data = np.column_stack((lab_ver_features, lab_hor_features))

task_names = tags_df.Task.as_matrix()
task_clusters = tags_df.TaskClusterId.as_matrix()
relevant_task_names = []
relevant_task_clusters = [0, 1, 2] # 1=resting, 4=periodic hand movement, 5=walking
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

labels = create_labels('dyskinesia', tags_data=tags_df, condition_vector=cond, binarize=True)
features = features_data[cond==True]
#tags_df_after_cond = tags_df[cond==True]
subject_ids = np.asarray((tags_df.SubjectId[cond==True]))
task_ids = tags_df.TaskID[cond==True]
#features = np.column_stack((features, feature_deep1[:,1:]   ))
patients =subject_ids.copy()

optimized_model2 = classifier.optimize_hyper_params(features, labels, np.asarray(patients), 'xgboost',
                                        hyper_params=None, scoring_measure = None ,eval_iterations = 10)


all_pred = classifier.make_cv_predictions_prob_for_all_segments(features, labels, pd.core.series.Series(patients), optimized_model,
                                                     task_ids)


agg_segments_df = all_pred.groupby(['patient', 'task', 'true_label']).agg(['min','max','mean','median','std'])
agg_segments_df.columns = agg_segments_df.columns.droplevel()
agg_segments_df.reset_index(inplace=True)

'''
Use the extracted features to classify each task.
Start by optimizing the hyper-parameters. Then, make predictions for each aggregated (task) segment:
'''
agg_patients = agg_segments_df['patient']
agg_labels = agg_segments_df['true_label']
agg_features = agg_segments_df[[x for x in agg_segments_df.columns if x not in ['patient', 'true_label', 'task']]]
#agg_features = agg_features.apply(lambda x:(x-np.mean(x))/np.std(x), 0)


opt_model_for_agg_segments = classifier.optimize_hyper_params(agg_features, agg_labels, agg_patients,
                                                   model_name='logistic_regression',
                                                   hyper_params=None, scoring_measure='f1',eval_iterations = 100)
final_pred = classifier.make_cv_predictions_for_agg_segments(agg_segments_df, opt_model_for_agg_segments,  binary_class_thresh=0.57)
evaluation.global_metrics(final_pred)

############################
raw_data =raw_data.sort_values(by = 'timestamp')
sec = 5 
freq = 50 

X_table_ses = []
Y_table_ses = []
Z_table_ses = []
raw_home_x = np.ones((int(raw_data.shape[0]/(sec*freq)), sec*freq))
raw_home_y = np.zeros((int(raw_data.shape[0]/(sec*freq)), sec*freq))
raw_home_z = np.zeros((int(raw_data.shape[0]/(sec*freq)), sec*freq))
raw_home_timestamp = np.empty([int(raw_data.shape[0]/(sec*freq)), sec*freq], dtype = str)#np.full((int(raw_data.shape[0]/(sec*freq)), sec*freq), raw_data['timestamp'][0])

for i in range(int(np.floor(raw_data.shape[0]/(sec*freq)))):
    # print(i)
    raw_home_x[i] = np.asarray(raw_data['x'].iloc[i*sec*freq:(i+1)*sec*freq])
    raw_home_y[i] = np.asarray(raw_data['y'].iloc[i*sec*freq:(i+1)*sec*freq])
    raw_home_z[i] = np.asarray(raw_data['z'].iloc[i*sec*freq:(i+1)*sec*freq])
    raw_home_timestamp[i] = np.asarray(raw_data['timestamp'].iloc[i*sec*freq:(i+1)*sec*freq])
X_table_ses.append(raw_home_x)
Y_table_ses.append(raw_home_y)
Z_table_ses.append(raw_home_z)

X_table = np.vstack(X_table_ses)
Y_table = np.vstack(Y_table_ses)
Z_table = np.vstack(Z_table_ses)

X_table = X_table/1000;
Y_table = Y_table/1000;
Z_table = Z_table/1000;

optimized_model = optimized_model.fit(features, labels)

lab_ver_proj_home, lab_hor_proj_home = projections.project_from_3_to_2_dims(X_table, Y_table, Z_table)

WavFeatures = WavTransform.WavTransform()
lab_x_features = WavFeatures.createWavFeatures(lab_ver_proj_home)
lab_y_features = WavFeatures.createWavFeatures(lab_hor_proj_home)

features_data_unlabeld = np.column_stack((lab_x_features, lab_y_features))

results = optimized_model.predict_proba(features_data_unlabeld)[:,1]
task_home_id = lmap(lambda x: int(x/6)*6.0, np.arange(len(results)))

home_results = pd.DataFrame({'prob': results, 'group_30_sec': task_home_id})
home_time = pd.DataFrame({'prob': raw_home_timestamp[:,0], 'group_30_sec': task_home_id})

agg_segments_home_results = home_results.groupby(['group_30_sec']).agg(['min','max','mean','median','std'])
agg_segments_home_time = home_time.groupby(['group_30_sec']).agg(['min'])

pred_for_home = opt_model_for_agg_segments.predict_proba(agg_segments_home_results)[:,1]

mypred = pd.DataFrame({'value': pred_for_home, 'measurement_name': 'tremor score per 30 sec', 
                      'measurement_id': 100, 'timestamp':  agg_segments_home_time['prob']['min'], ' cohort': raw_data['cohort'].iloc[0]})
    
