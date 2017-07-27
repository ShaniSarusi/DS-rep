# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:12:28 2017

@author: imazeh
"""

import os
from os.path import join, sep
from scipy import constants

cloud_data_path = join(sep, 'mnt', 'intel-health-analytics', 'data', 'teva')
exec_path = join('C:', sep, 'Users', 'imazeh', 'Itzik', 'Health_prof', 'git_team', 'DataScientists', 'Teva')
#os.chdir(join('C:', sep, 'Users', 'imazeh', 'Itzik', 'Health_prof', 'git_team', 'DataScientists', 'Teva'))
data_path = join('C:', sep, 'Users', 'imazeh', 'Itzik', 'Health_prof', 'Datasets', 'Teva', 'data')

valid_users = [8, 25, 45, 51, 57, 63, 74, 82, 94, 108]
report_values = range(5)
watch_acc_baseline = 1000
phone_acc_baseline = constants.g


# Watch accelerometer data:

clinic_assessments_watch_acc_file_path = join(data_path, 'clinic_assessments_watch_acc_data.csv')
clinic_assessments_watch_acc_pkl_file_path = join(data_path, 'clinic_assessments_watch_acc_pkl.p')

clinic_steps_watch_acc_file_path = join(data_path, 'clinic_steps_watch_acc_data.csv')
clinic_steps_watch_acc_pkl_file_path = join(data_path, 'clinic_steps_watch_acc_pkl.p')

home_assessments_watch_acc_file_path = join(data_path, 'home_assessments_watch_acc_data.csv')
home_assessments_watch_acc_pkl_file_path = join(data_path, 'home_assessments_watch_acc_pkl.p')

home_steps_watch_acc_file_path = join(data_path, 'home_steps_watch_acc_data.csv')
home_steps_watch_acc_pkl_file_path = join(data_path, 'home_steps_watch_acc_pkl.p')

home_reminders_watch_acc_file_path = join(data_path, 'home_reminders_watch_acc_data.csv')
home_reminders_watch_acc_pkl_file_path = join(data_path, 'home_reminders_watch_acc_pkl.p')


# Phone accelerometer data:

clinic_steps_phone_acc_file_path = join(data_path, 'clinic_steps_phone_acc_data.csv')
clinic_steps_phone_acc_pkl_file_path = join(data_path, 'clinic_steps_phone_acc_pkl.p')

home_steps_phone_acc_file_path = join(data_path, 'home_steps_phone_acc_data.csv')
home_steps_phone_acc_pkl_file_path = join(data_path, 'home_steps_phone_acc_pkl.p')
