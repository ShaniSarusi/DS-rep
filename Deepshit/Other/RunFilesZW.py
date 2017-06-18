# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 14:22:55 2016

@author: awagner
"""

import os
mother_path = os.path.abspath('PycharmProjects') + "/Deepshit/"
if(mother_path[2] == '\\' and mother_path[9] == 'z'):
    mother_path = 'C:/Users/zwaks/Documents/Workspaces/GitHub/DataScientists/PycharmProjects/Deepshit/'

execfile(mother_path + 'organizeData.py')
execfile(mother_path + 'butterFilter.py')#conn = pyodbc.connect("DSN=ConnectSmoove2", autocommit=True)
execfile(mother_path + 'BuildNetwork.py')
execfile(mother_path + 'RunYourNetwork.py')
execfile(mother_path + 'PredictionWithNet.py')