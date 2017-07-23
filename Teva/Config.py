# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:12:28 2017

@author: imazeh
"""

import os
from os.path import join, sep

cloud_data_path = join(sep, 'mnt', 'intel-health-analytics', 'data', 'teva')
exec_path = join('C:', sep, 'Users', 'imazeh', 'Itzik', 'Health_prof', 'git_team', 'DataScientists', 'Teva')
#os.chdir(join('C:', sep, 'Users', 'imazeh', 'Itzik', 'Health_prof', 'git_team', 'DataScientists', 'Teva'))
data_path = join('C:', sep, 'Users', 'imazeh', 'Itzik', 'Health_prof', 'Datasets', 'Teva', 'data')

clinic_file_path = join(data_path, 'clinic_assessment_watch_acc_data_view.csv')
clinic_w_steps_file_path = join(data_path, 'clinic_assessment_with_steps_watch_acc.csv')
home_file_name = join(data_path, 'home_assessments_watch_acc_data_view.csv')
home_w_steps_file_name = join(data_path, 'home_assessments_with_steps_watch_acc.csv')
five_mins_file_name = join(data_path, 'last_5_minutes_watch_acc_data_view.csv')

valid_users = [8, 25, 45, 51, 57, 63, 74, 82, 94, 108]
report_values = range(5)
#accelerometer
