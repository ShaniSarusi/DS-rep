# -*- coding: utf-8 -*-
"""
Created on Thu May 11 08:44:04 2017

@author: imazeh
"""
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

meta_data_df = meta.copy()
del meta_data_df['Unnamed: 0']
#features_df = pd.read_csv(mother_path+'wavelet_features.csv')
#all_data_df = pd.concat([meta_data_df, features_df], axis=1)

task_cluster = 1
task_df = meta_data_df[meta_data_df.TaskClusterId == task_cluster].copy()
task_df['visit_date'] = pd.to_datetime(task_df.TSStart).dt.date

#Read the UPDRS data:
updrs_data = pd.read_csv(data_path+'../UPDRS_WITH_DETAILS.csv')
#Remove patient ldhpds2, which didn't have UPDRS data:
updrs_data = updrs_data[2:]

'''
First approach - aggregate all 5-sec segments in a visit, and calculate its correlation with the
UPDRS-part-3 assessment provided by the clinician.
'''

#Add the initial model's prediction for each 5-sec segment:
task_df['first_model_symp_prediction'] = pred_as_vector

#Aggregate per visit:
visit_summary_per_5_df = task_df[['SubjectId', 'IntelUsername', 'visit_date', 'first_model_symp_prediction']].groupby(['SubjectId', 'IntelUsername', 'visit_date']).agg(['mean', 'median'])
visit_summary_per_5_df.columns = visit_summary_per_5_df.columns.droplevel(0)
visit_summary_per_5_df = visit_summary_per_5_df.rename_axis(None, axis=1)
visit_summary_per_5_df.reset_index(inplace=True)
cols_for_concat = visit_summary_per_5_df[['IntelUsername', 'mean', 'median']]
cols_for_concat = cols_for_concat[4:]

all_data_5_sec = pd.concat([updrs_data.reset_index(), cols_for_concat.reset_index()], axis=1)

#Create a box-plot:
plt.boxplot([np.asarray(all_data_5_sec['mean'][all_data_5_sec.Rest_tremor==0]), \
             np.asarray(all_data_5_sec['mean'][all_data_5_sec.Rest_tremor==1]), \
             np.asarray(all_data_5_sec['mean'][all_data_5_sec.Rest_tremor==2]), \
             np.asarray(all_data_5_sec['mean'][all_data_5_sec.Rest_tremor==3])])
plt.xticks([1,2,3,4],['0', '1', '2', '3'])

#visit_summary_df.to_csv(mother_path+'visits_summary_first_approach.csv')

#Read after adding UPDRS scores:
#visits_w_updrs_first = pd.read_csv(mother_path+'visits_summary_first_approach.csv')
#visits_w_updrs_first_no_na = visits_w_updrs_first.dropna().copy()

#plot correlations:
#Mean:
#visits_w_updrs_first_no_na.sort_values(by='mean', inplace=True)
#mean_corr = np.corrcoef(visits_w_updrs_first_no_na['mean'], visits_w_updrs_first_no_na.UPDRS_part_3)


'''
Second approach - aggregate all 5-sec segments in a visit, and calculate its correlation with the
UPDRS-part-3 assessment provided by the clinician.
'''

task_df['second_model_symp_prediction'] = clf.predict_proba(prob_mat_table[features_col_names])[:,1]
task_df_sub = task_df[['TSStart', 'visit_date', 'SubjectId', 'IntelUsername', 'second_model_symp_prediction']].copy()
print (task_df_sub.shape)
task_df_sub.drop_duplicates(inplace=True)
print (task_df_sub.shape)

#Aggregate per visit:
visit_summary_second_df = task_df_sub[['SubjectId', 'IntelUsername', 'visit_date', 'second_model_symp_prediction']].groupby(['SubjectId', 'IntelUsername', 'visit_date']).agg(['mean', 'median'])
visit_summary_second_df.to_csv(mother_path+'visits_summary_second_approach.csv')

#Read after adding UPDRS scores:
visits_w_updrs_second = pd.read_csv(mother_path+'visits_summary_second_approach.csv')
visits_w_updrs_no_na = visits_w_updrs.dropna().copy()

#plot correlations:
#Mean:
visits_w_updrs_no_na.sort_values(by='mean', inplace=True)
mean_corr = np.corrcoef(visits_w_updrs_no_na['mean'], visits_w_updrs_no_na.UPDRS_part_3)


'''
Compare clinician's binary classification to the UPDRS score
'''
meta_sub = task_df[['TSStart', 'visit_date', 'SubjectId', 'IntelUsername', 'BradykinesiaGA']].copy()
print (meta_sub.shape)
meta_sub.drop_duplicates(inplace=True)
print (meta_sub.shape)
binary_avg = meta_sub[['visit_date', 'SubjectId', 'IntelUsername', 'BradykinesiaGA']].groupby(['visit_date', 'SubjectId', 'IntelUsername']).agg('mean')
binary_avg.to_csv(mother_path+'binary_avg_per_visit.csv')

binary_w_updrs = pd.read_csv(mother_path+'binary_avg_per_visit.csv')
binary_w_updrs_no_na = binary_w_updrs.dropna().copy()

#plot correlations:
#Mean:
#visits_w_updrs_no_na.sort_values(by='mean', inplace=True)
mean_corr = np.corrcoef(binary_w_updrs_no_na['BradykinesiaGA'], binary_w_updrs_no_na.UPDRS_part_3)
