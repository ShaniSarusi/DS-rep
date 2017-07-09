# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:17:35 2017

@author: imazeh
"""

import pandas as pd
from os.path import join, sep
# import imp

exec_path = join('C:', sep, 'Users', 'imazeh', 'Itzik', 'Health_prof', 'git_team', 'DataScientists', 'Teva')
import Config as c
valid_users = c.valid_users
# imp.reload(c)

clinic_df = pd.read_csv(c.clinic_file_path)
inds = pd.isnull(clinic_df).any(1).nonzero()[0]
print('len of inds:', len(inds))
print(clinic_df.user_id.unique())
clinic_df = clinic_df[clinic_df.user_id.isin(valid_users)]
# Convert relevant columns to datetime format:
clinic_df['assessment_start'] = pd.to_datetime(clinic_df['assessment_start'])
clinic_df['assessment_end'] = pd.to_datetime(clinic_df['assessment_end'])
clinic_df['timestamp'] = pd.to_datetime(clinic_df['timestamp'])

sub_clinic_df = clinic_df[['user_id', 'assessment_id', 'assessment_start', 'assessment_end',
                           'timestamp', 'x', 'y', 'z', 'patient_report_value', 'clinician_report_value']]
sub_clinic_df.to_pickle(join(c.data_path, 'clinic_processed_pickle.p'))
# sub_clinic_df = pd.read_pickle(join(c.data_path, 'clinic_processed_pickle.p'))



home_df = pd.read_csv(c.home_file_name)
inds = pd.isnull(home_df).any(1).nonzero()[0]
print('len of inds:', len(inds))
print(home_df.user_id.unique())
home_df = home_df[home_df.user_id.isin(valid_users)]
# Rename for compatibility:
home_df['patient_report_value'] = home_df['value']
# Convert relevant columns to datetime format:
home_df['assessment_start'] = pd.to_datetime(home_df['assessment_start'])
home_df['assessment_end'] = pd.to_datetime(home_df['assessment_end'])
home_df['timestamp'] = pd.to_datetime(home_df['timestamp'])
# Rearrange the columns and save:
sub_home_df = home_df[['user_id', 'assessment_id', 'assessment_start', 'assessment_end',
                       'timestamp', 'x', 'y', 'z', 'patient_report_value']]
sub_home_df.to_pickle(join(c.data_path, 'home_processed_pickle.p'))




five_mins_df = pd.read_csv(c.five_mins_file_name)
inds = pd.isnull(five_mins_df).any(1).nonzero()[0]
print('len of inds:', len(inds))
print(five_mins_df.user_id.unique())
five_mins_df = five_mins_df[five_mins_df.user_id.isin(valid_users)]
# Rename for compatibility:
five_mins_df['patient_report_value'] = five_mins_df['value']
five_mins_df['reminder_id'] = five_mins_df['id']
# Convert relevant columns to datetime format:
five_mins_df['reported_timestamp'] = pd.to_datetime(five_mins_df['reported_timestamp'])
five_mins_df['reported_minus_5'] = pd.to_datetime(five_mins_df['reported_minus_5'])
five_mins_df['timestamp'] = pd.to_datetime(five_mins_df['timestamp'])
# Rearrange the columns and save:
sub_five_mins_df = five_mins_df[['user_id', 'reminder_id', 'reported_timestamp', 'reported_minus_5',
                                 'timestamp', 'x', 'y', 'z', 'patient_report_value']]
sub_five_mins_df.to_pickle(join(c.data_path, 'reminder_five_minutes_processed_pickle.p'))
