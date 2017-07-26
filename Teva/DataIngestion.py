# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:17:35 2017

@author: imazeh
"""

import pandas as pd
from os.path import join, sep
import imp

exec_path = join('C:', sep, 'Users', 'imazeh', 'Itzik', 'Health_prof', 'git_team', 'DataScientists', 'Teva')
import Config as c
valid_users = c.valid_users
imp.reload(c)

''' Read and ingest watch accelerometer data:'''

#######################################################################
clinic_assessments_watch_df = pd.read_csv(c.clinic_assessments_watch_acc_file_path)
inds = pd.isnull(clinic_assessments_watch_df).any(1).nonzero()[0]
print('len of inds:', len(inds))
print(clinic_assessments_watch_df.user_id.unique())
clinic_assessments_watch_df = clinic_assessments_watch_df[clinic_assessments_watch_df.user_id.isin(valid_users)]
# Convert relevant columns to datetime format:
clinic_assessments_watch_df['assessment_start'] = pd.to_datetime(clinic_assessments_watch_df['assessment_start'])
clinic_assessments_watch_df['assessment_end'] = pd.to_datetime(clinic_assessments_watch_df['assessment_end'])
clinic_assessments_watch_df['timestamp'] = pd.to_datetime(clinic_assessments_watch_df['timestamp'])

sub_clinic_assessments_watch_df = clinic_assessments_watch_df[['user_id', 'assessment_id', 'assessment_start',
                                                               'assessment_end', 'timestamp', 'x', 'y', 'z',
                                                               'patient_report_value', 'clinician_report_value']]
sub_clinic_assessments_watch_df.to_pickle(c.clinic_assessments_watch_acc_pkl_file_path)


#######################################################################
clinic_steps_watch_df = pd.read_csv(c.clinic_steps_watch_acc_file_path)
inds = pd.isnull(clinic_steps_watch_df).any(1).nonzero()[0]
print('len of inds:', len(inds))
print(clinic_steps_watch_df.user_id.unique())
clinic_steps_watch_df = clinic_steps_watch_df[clinic_steps_watch_df.user_id.isin(valid_users)]
# Convert relevant columns to datetime format:
clinic_steps_watch_df['assessment_start'] = pd.to_datetime(clinic_steps_watch_df['assessment_start'])
clinic_steps_watch_df['assessment_end'] = pd.to_datetime(clinic_steps_watch_df['assessment_end'])
clinic_steps_watch_df['timestamp'] = pd.to_datetime(clinic_steps_watch_df['timestamp'])

sub_clinic_steps_watch_df = clinic_steps_watch_df[['user_id', 'assessment_id', 'assessment_start', 'assessment_end',
                                                   'timestamp', 'step_name', 'x', 'y', 'z', 'patient_report_value',
                                                   'clinician_report_value']]
sub_clinic_steps_watch_df.to_pickle(c.clinic_steps_watch_acc_pkl_file_path)


#######################################################################
home_assessments_watch_df = pd.read_csv(c.home_assessments_watch_acc_file_path)
inds = pd.isnull(home_assessments_watch_df).any(1).nonzero()[0]
print('len of inds:', len(inds))
print(home_assessments_watch_df.user_id.unique())
home_assessments_watch_df = home_assessments_watch_df[home_assessments_watch_df.user_id.isin(valid_users)]
# Rename for compatibility:
home_assessments_watch_df['patient_report_value'] = home_assessments_watch_df['value']
# Convert relevant columns to datetime format:
home_assessments_watch_df['assessment_start'] = pd.to_datetime(home_assessments_watch_df['assessment_start'])
home_assessments_watch_df['assessment_end'] = pd.to_datetime(home_assessments_watch_df['assessment_end'])
home_assessments_watch_df['timestamp'] = pd.to_datetime(home_assessments_watch_df['timestamp'])
# Rearrange the columns and save:
sub_home_assessments_watch_df = home_assessments_watch_df[['user_id', 'assessment_id', 'assessment_start',
                                                           'assessment_end', 'timestamp', 'x', 'y', 'z',
                                                           'patient_report_value']]
sub_home_assessments_watch_df.to_pickle(c.home_assessments_watch_acc_pkl_file_path)


#######################################################################
home_steps_watch_df = pd.read_csv(c.home_steps_watch_acc_file_path)
inds = pd.isnull(home_steps_watch_df).any(1).nonzero()[0]
print('len of inds:', len(inds))
print(home_steps_watch_df.user_id.unique())
home_steps_watch_df = home_steps_watch_df[home_steps_watch_df.user_id.isin(valid_users)]
# Rename for compatibility:
home_steps_watch_df['patient_report_value'] = home_steps_watch_df['value']
# Convert relevant columns to datetime format:
home_steps_watch_df['assessment_start'] = pd.to_datetime(home_steps_watch_df['assessment_start'])
home_steps_watch_df['assessment_end'] = pd.to_datetime(home_steps_watch_df['assessment_end'])
home_steps_watch_df['timestamp'] = pd.to_datetime(home_steps_watch_df['timestamp'])
# Rearrange the columns and save:
sub_home_steps_watch_df = home_steps_watch_df[['user_id', 'assessment_id', 'assessment_start', 'assessment_end',
                                               'timestamp', 'step_name', 'x', 'y', 'z', 'patient_report_value']]
sub_home_steps_watch_df.to_pickle(c.home_steps_watch_acc_pkl_file_path)


#######################################################################
home_reminders_watch_df = pd.read_csv(c.home_reminders_watch_acc_file_path)
inds = pd.isnull(home_reminders_watch_df).any(1).nonzero()[0]
print('len of inds:', len(inds))
print(home_reminders_watch_df.user_id.unique())
home_reminders_watch_df = home_reminders_watch_df[home_reminders_watch_df.user_id.isin(valid_users)]
# Rename for compatibility:
home_reminders_watch_df['patient_report_value'] = home_reminders_watch_df['value']
home_reminders_watch_df['reminder_id'] = home_reminders_watch_df['id']
# Convert relevant columns to datetime format:
home_reminders_watch_df['reported_timestamp'] = pd.to_datetime(home_reminders_watch_df['reported_timestamp'])
home_reminders_watch_df['reported_minus_5'] = pd.to_datetime(home_reminders_watch_df['reported_minus_5'])
home_reminders_watch_df['timestamp'] = pd.to_datetime(home_reminders_watch_df['timestamp'])
# Rearrange the columns and save:
sub_home_reminders_watch_df = home_reminders_watch_df[['user_id', 'reminder_id', 'reported_timestamp',
                                                       'reported_minus_5', 'timestamp', 'x', 'y', 'z',
                                                       'patient_report_value']]
sub_home_reminders_watch_df.to_pickle(c.home_reminders_watch_acc_pkl_file_path)


''' Read and ingest watch accelerometer data:'''

#######################################################################
clinic_steps_phone_df = pd.read_csv(c.clinic_steps_phone_acc_file_path)
inds = pd.isnull(clinic_steps_phone_df).any(1).nonzero()[0]
print('len of inds:', len(inds))
print(clinic_steps_phone_df.user_id.unique())
clinic_steps_phone_df = clinic_steps_phone_df[clinic_steps_phone_df.user_id.isin(valid_users)]
# Rename for compatibility:
clinic_steps_phone_df['patient_report_value'] = clinic_steps_phone_df['patient_report_desc']
clinic_steps_phone_df['clinician_report_value'] = clinic_steps_phone_df['clinician_report_desc']
# Convert relevant columns to datetime format:
clinic_steps_phone_df['assessment_start'] = pd.to_datetime(clinic_steps_phone_df['assessment_start'])
clinic_steps_phone_df['assessment_end'] = pd.to_datetime(clinic_steps_phone_df['assessment_end'])
clinic_steps_phone_df['timestamp'] = pd.to_datetime(clinic_steps_phone_df['timestamp'])

sub_clinic_steps_phone_df = clinic_steps_phone_df[['user_id', 'assessment_id', 'assessment_start', 'assessment_end',
                                                   'timestamp', 'step_name', 'x', 'y', 'z', 'patient_report_value',
                                                   'clinician_report_value']]
sub_clinic_steps_phone_df.to_pickle(c.clinic_steps_phone_acc_file_path)


#######################################################################
home_steps_phone_df = pd.read_csv(c.home_steps_phone_acc_file_path)
inds = pd.isnull(home_steps_phone_df).any(1).nonzero()[0]
print('len of inds:', len(inds))
print(home_steps_phone_df.user_id.unique())
home_steps_phone_df = home_steps_phone_df[home_steps_phone_df.user_id.isin(valid_users)]
# Rename for compatibility:
home_steps_phone_df['patient_report_value'] = home_steps_phone_df['patient_report_desc']
# Convert relevant columns to datetime format:
home_steps_phone_df['assessment_start'] = pd.to_datetime(home_steps_phone_df['assessment_start'])
home_steps_phone_df['assessment_end'] = pd.to_datetime(home_steps_phone_df['assessment_end'])
home_steps_phone_df['timestamp'] = pd.to_datetime(home_steps_phone_df['timestamp'])
# Rearrange the columns and save:
sub_home_steps_phone_df = home_steps_phone_df[['user_id', 'assessment_id', 'assessment_start', 'assessment_end',
                                               'timestamp', 'step_name', 'x', 'y', 'z', 'patient_report_value']]
sub_home_steps_phone_df.to_pickle(c.home_steps_phone_acc_pkl_file_path)
